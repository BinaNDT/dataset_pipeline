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
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.img_ids[idx]
        
        # Get image info and load image
        img_info = next(img for img in self.coco['images'] if img['id'] == img_id)
        img_path = self.image_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        anns = self.img_to_anns[img_id]
        
        # Create instance masks
        masks = []
        labels = []
        for ann in anns:
            mask = mask_utils.decode(ann['segmentation'])
            masks.append(mask)
            labels.append(ann['category_id'])
            
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

def get_transform(split: str) -> transforms.Compose:
    """Get transforms for different splits"""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
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
            pin_memory=True
        )
        for split, dataset in datasets.items()
    }
    
    return loaders['train'], loaders['val'], loaders['test'] 