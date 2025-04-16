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
import gc
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import *

class BuildingDamageDataset(Dataset):
    def __init__(self, 
                 image_dir: Path,
                 annotation_file: Path,
                 transform: Optional[transforms.Compose] = None,
                 split: str = "train",
                 max_samples: Optional[int] = None):
        """
        Dataset class for building damage segmentation
        Args:
            image_dir: Directory containing images
            annotation_file: Path to COCO format annotation file
            transform: Optional transforms to apply
            split: One of ["train", "val", "test"]
            max_samples: Optional limit on number of samples to use (for debugging)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.split = split
        
        # Load annotations
        logging.info(f"Loading annotations for {split} split...")
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        
        # Load class mapping
        if CLASS_MAPPING_FILE.exists():
            self.class_mapping_df = pd.read_csv(CLASS_MAPPING_FILE)
            logging.info(f"Loaded class mapping from {CLASS_MAPPING_FILE}")
            logging.info(f"Classes: {self.class_mapping_df['class_name'].tolist()}")
        
        # Create category id to new id mapping
        self.cat_id_map = {}
        if 'categories' in self.coco:
            # Map categories to ids 1-6 (0 is background)
            for i, cat in enumerate(self.coco['categories'], start=1):
                # Ensure background is 0, others are 1-6
                if cat['name'] == '__background__':
                    self.cat_id_map[cat['id']] = 0
                else:
                    self.cat_id_map[cat['id']] = i
                logging.info(f"Category mapping: {cat['id']} -> {self.cat_id_map[cat['id']]} ({cat['name']})")
        else:
            # Default mapping if no categories
            for i in range(1, 7):
                self.cat_id_map[i] = i
            self.cat_id_map[0] = 0  # Background
        
        # Create image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        # Get image ids with annotations
        all_img_ids = list(self.img_to_anns.keys())
        logging.info(f"Found {len(all_img_ids)} images with annotations")
        
        # Split dataset
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
            
        # Limit dataset size if specified
        if max_samples is not None:
            self.img_ids = self.img_ids[:max_samples]
            
        logging.info(f"{split} split contains {len(self.img_ids)} images")
            
        # Create mapping from img_id to image info for faster lookup
        self.img_id_to_info = {img['id']: img for img in self.coco['images']}
            
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
    
    def map_category_id(self, cat_id):
        """Map original category id to model category id (0-5)"""
        return self.cat_id_map.get(cat_id, 0)  # Default to background if not found
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.img_ids[idx]
        
        # Get image info and load image
        img_info = self.img_id_to_info[img_id]
        img_path = self.image_dir / img_info['file_name']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a black image as a fallback
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
        
        # Get image dimensions
        width, height = img_info.get('width', image.width), img_info.get('height', image.height)
        
        # Get annotations
        anns = self.img_to_anns[img_id]
        
        # Create instance masks
        masks = []
        labels = []
        
        # Limit number of annotations per image to save memory (optional)
        # anns = anns[:5]  # Uncomment to limit annotations
        
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
                    # Map category id to ensure it's in range 0-5
                    mapped_category = self.map_category_id(ann['category_id'])
                    labels.append(mapped_category)
                    
        if not masks:  # Handle case with no valid masks
            # Create a dummy mask
            dummy_mask = np.zeros((height, width), dtype=np.uint8)
            masks = [dummy_mask]
            labels = [0]  # Background class
        
        # Convert to tensor
        masks = torch.as_tensor(np.stack(masks), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Clear memory
        gc.collect()
            
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
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
    num_workers: int = 2,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, val, and test sets"""
    
    # For debugging, use a very small subset
    debug_mode = True
    if debug_mode:
        max_samples = 10
        logging.warning(f"DEBUG MODE: Using only {max_samples} samples per split")
    
    datasets = {
        split: BuildingDamageDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            transform=get_transform(split),
            split=split,
            max_samples=max_samples
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