import unittest
import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import shutil
import tempfile

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.dataset import BuildingDamageDataset, custom_collate_fn, get_transform
from config import *

class TestDatasetUtils(unittest.TestCase):
    """Unit tests for dataset utility functions"""
    
    def setUp(self):
        """Create temporary test data"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.image_dir = self.temp_dir / "images"
        self.image_dir.mkdir()
        
        # Create a sample image
        sample_img = Image.new('RGB', (100, 100), color='white')
        self.img_path = self.image_dir / "test_image.png"
        sample_img.save(self.img_path)
        
        # Create a sample COCO annotation
        self.annotation_file = self.temp_dir / "annotations.json"
        
        coco_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "test_image.png",
                    "width": 100,
                    "height": 100
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [
                        [10, 10, 10, 50, 50, 50, 50, 10]  # Simple square
                    ],
                    "area": 1600,
                    "bbox": [10, 10, 40, 40]
                }
            ],
            "categories": [
                {"id": 0, "name": "__background__"},
                {"id": 1, "name": "Building-No-Damage"},
                {"id": 2, "name": "Building-Minor-Damage"},
                {"id": 3, "name": "Building-Major-Damage"},
                {"id": 4, "name": "Building-Total-Destruction"}
            ]
        }
        
        with open(self.annotation_file, 'w') as f:
            json.dump(coco_data, f)
            
        # Create a sample dataset instance
        self.dataset = BuildingDamageDataset(
            image_dir=self.image_dir,
            annotation_file=self.annotation_file,
            transform=get_transform("train"),
            split="train"
        )
    
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
    
    def test_polygon_to_mask(self):
        """Test conversion from polygon to binary mask"""
        # Test with a simple square polygon
        segmentation = [[10, 10, 10, 50, 50, 50, 50, 10]]
        height, width = 100, 100
        
        mask = self.dataset.polygon_to_mask(segmentation, height, width)
        
        # Check that the mask is the expected shape
        self.assertEqual(mask.shape, (height, width))
        
        # Check that the mask contains 1s in the square region
        self.assertEqual(mask[20, 20], 1)  # Inside the square
        self.assertEqual(mask[5, 5], 0)    # Outside the square
        
        # Check that the mask area is approximately right
        mask_area = mask.sum()
        self.assertGreater(mask_area, 1000)  # Should be approximately 1600 (40x40)
        self.assertLess(mask_area, 2000)     # Allow for some rounding/approximation
    
    def test_map_category_id(self):
        """Test category ID mapping"""
        # Test mapping from original COCO ids to our model's ids
        self.assertEqual(self.dataset.map_category_id(0), 0)  # Background stays 0
        self.assertEqual(self.dataset.map_category_id(1), 1)  # Regular class
        
        # Test with invalid/unknown ID
        self.assertEqual(self.dataset.map_category_id(999), 0)  # Should default to background
    
    def test_custom_collate_fn(self):
        """Test the custom collate function for batching"""
        # Create sample batch items
        batch = [
            {
                'image': torch.rand(3, 64, 64),
                'masks': torch.zeros(2, 64, 64),
                'labels': torch.tensor([1, 2]),
                'image_id': 1
            },
            {
                'image': torch.rand(3, 64, 64),
                'masks': torch.zeros(1, 64, 64),
                'labels': torch.tensor([3]),
                'image_id': 2
            }
        ]
        
        # Apply collate function
        collated = custom_collate_fn(batch)
        
        # Check that images are stacked properly
        self.assertEqual(collated['image'].shape, (2, 3, 64, 64))
        
        # Check that masks and labels are kept as lists
        self.assertEqual(len(collated['masks']), 2)
        self.assertEqual(len(collated['labels']), 2)
        
        # Check first item's masks shape
        self.assertEqual(collated['masks'][0].shape, (2, 64, 64))
        
        # Check second item's labels
        self.assertEqual(collated['labels'][1].item(), 3)
        
        # Check image IDs
        self.assertEqual(collated['image_ids'], [1, 2])
    
    def test_dataset_getitem(self):
        """Test the __getitem__ method of BuildingDamageDataset"""
        # Get the first item from the dataset
        item = self.dataset[0]
        
        # Check that it contains all required keys
        self.assertIn('image', item)
        self.assertIn('masks', item)
        self.assertIn('labels', item)
        self.assertIn('image_id', item)
        
        # Check image shape (should be transformed to the IMAGE_SIZE defined in config)
        self.assertEqual(len(item['image'].shape), 3)  # C, H, W
        
        # Check that we have at least one mask
        self.assertGreater(item['masks'].shape[0], 0)
        
        # Check that the number of labels matches the number of masks
        self.assertEqual(item['masks'].shape[0], len(item['labels']))

if __name__ == '__main__':
    unittest.main() 