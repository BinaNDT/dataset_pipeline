import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools import mask as mask_utils
from typing import Dict, List, Tuple, Optional
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent))
from config import *

class BuildingDamageDataset(Dataset):
    def __init__(self, 
                 image_dir: Path,
                 annotation_file: Path,
                 transform: Optional[transforms.Compose] = None,
                 split: str = "train"):
        """
        Dataset class for building damage segmentation
        Args:
            image_dir: Directory containing images
            annotation_file: Path to COCO format annotation file
            transform: Optional transforms to apply
            split: One of ["train", "val", "test"]
        """
        self.image_dir = image_dir
        self.transform = transform
        self.split = split
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
            
        # Create image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        # Split dataset
        all_img_ids = list(self.img_to_anns.keys())
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(all_img_ids)
        
        n_total = len(all_img_ids)
        n_train = int(n_total * TRAIN_SPLIT)
        n_val = int(n_total * VAL_SPLIT)
        
        if split == "train":
            self.img_ids = all_img_ids[:n_train]
        elif split == "val":
            self.img_ids = all_img_ids[n_train:n_train+n_val]
        else:  # test
            self.img_ids = all_img_ids[n_train+n_val:]
            
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def polygon_to_mask(self, segmentation, height, width):
        """Convert polygon to binary mask"""
        import pycocotools.mask as mask_utils
        if isinstance(segmentation, list):
            # Convert polygon to RLE
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
            return mask_utils.decode(rle)
        return None
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.img_ids[idx]
        
        # Get image info and load image
        img_info = next(img for img in self.coco['images'] if img['id'] == img_id)
        img_path = self.image_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get image dimensions
        width, height = img_info.get('width', image.width), img_info.get('height', image.height)
        
        # Get annotations
        anns = self.img_to_anns[img_id]
        
        # Create instance masks
        masks = []
        labels = []
        for ann in anns:
            # Handle different segmentation formats
            if 'segmentation' in ann:
                segmentation = ann['segmentation']
                
                if isinstance(segmentation, dict):  # RLE format
                    mask = mask_utils.decode(segmentation)
                elif isinstance(segmentation, list):  # Polygon format
                    mask = self.polygon_to_mask(segmentation, height, width)
                else:
                    # Skip invalid segmentation
                    continue
                    
                if mask is not None:
                    masks.append(mask)
                    labels.append(ann['category_id'])
                    
        if not masks:  # Handle case with no valid masks
            # Create a dummy mask
            dummy_mask = np.zeros((height, width), dtype=np.uint8)
            masks = [dummy_mask]
            labels = [0]  # Background class
        
        # Convert to tensor
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            
        return {
            'image': image,
            'masks': masks,
            'labels': labels,
            'image_id': img_id
        }

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable numbers of masks per image
    """
    images = torch.stack([item['image'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    
    # Keep masks and labels as lists (not stacked)
    masks = [item['masks'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        'image': images,
        'masks': masks,
        'labels': labels,
        'image_id': image_ids
    }

def get_transform(split: str) -> transforms.Compose:
    """Get transforms for different splits"""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize before crop to maintain aspect ratio
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(
    image_dir: Path,
    annotation_file: Path,
    batch_size: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, val, and test sets"""
    
    datasets = {
        split: BuildingDamageDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            transform=get_transform(split),
            split=split
        )
        for split in ['train', 'val', 'test']
    }
    
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        for split, dataset in datasets.items()
    }
    
    return loaders['train'], loaders['val'], loaders['test'] 