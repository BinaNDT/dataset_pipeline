import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from transformers.optimization import get_cosine_schedule_with_warmup
import wandb
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import gc  # Add garbage collection
from torch.amp import autocast, GradScaler  # Import mixed precision tools

sys.path.append(str(Path(__file__).parent))
from config import *
from utils.dataset import create_data_loaders, BuildingDamageDataset, get_transform, custom_collate_fn

# Flag to enable/disable wandb
USE_WANDB = False

# Flag to enable/disable mixed precision
USE_MIXED_PRECISION = True

# Debug mode will use a tiny subset of data to quickly test the pipeline
DEBUG_MODE = False

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'training.log'),
            logging.StreamHandler()
        ]
    )

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Trainer:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # For mixed precision training
        self.use_mixed_precision = USE_MIXED_PRECISION
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None
        
        # Initialize model
        logging.info("Initializing model...")
        config = Mask2FormerConfig.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=NUM_CLASSES,
            num_queries=NUM_QUERIES,
            ignore_mismatched_sizes=True
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MODEL_CHECKPOINT,
            config=config,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Create datasets
        logging.info("Loading datasets...")
        max_samples = 20 if DEBUG_MODE else None
        self.train_dataset = BuildingDamageDataset(
            image_dir=IMAGES_DIR,
            annotation_file=ANNOTATIONS_FILE,
            transform=get_transform('train'),
            split='train',
            max_samples=max_samples
        )
        
        self.val_dataset = BuildingDamageDataset(
            image_dir=IMAGES_DIR,
            annotation_file=ANNOTATIONS_FILE,
            transform=get_transform('val'),
            split='val',
            max_samples=max_samples
        )
        
        # Create samplers for distributed training
        if self.world_size > 1:
            self.train_sampler = DistributedSampler(self.train_dataset)
            self.val_sampler = DistributedSampler(self.val_dataset)
        else:
            self.train_sampler = None
            self.val_sampler = None
        
        # Create data loaders with samplers
        logging.info("Creating data loaders...")
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=2,  # Reduced to save memory
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=2,  # Reduced to save memory
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        num_training_steps = len(self.train_loader) * NUM_EPOCHS
        num_warmup_steps = num_training_steps // 10
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize WandB only if enabled
        self.use_wandb = USE_WANDB and rank == 0
        if self.use_wandb:
            try:
                wandb.init(
                    project="building-damage-segmentation",
                    config={
                        "learning_rate": LEARNING_RATE,
                        "epochs": NUM_EPOCHS,
                        "batch_size": BATCH_SIZE_PER_GPU * world_size,
                        "model": MODEL_CHECKPOINT,
                        "num_classes": NUM_CLASSES,
                        "mixed_precision": self.use_mixed_precision
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
                
        logging.info(f"Using mixed precision: {self.use_mixed_precision}")
        logging.info(f"Training on {len(self.train_dataset)} samples, validating on {len(self.val_dataset)} samples")
    
    def process_batch(self, batch):
        """Process batch for model input - handle variable-sized masks"""
        # Move images to device
        images = batch['image'].to(self.device)
        
        processed_batch = {
            'pixel_values': images,
            'mask_labels': [],
            'class_labels': []
        }
        
        # Process masks and labels (variable sized)
        for i in range(len(batch['masks'])):
            masks = batch['masks'][i].to(self.device)
            labels = batch['labels'][i].to(self.device)
            processed_batch['mask_labels'].append(masks)
            processed_batch['class_labels'].append(labels)
            
        return processed_batch
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        if self.world_size > 1:
            self.train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(self.train_loader, disable=self.rank != 0)
        for batch_idx, batch in enumerate(progress_bar):
            # Clear cache before processing batch
            torch.cuda.empty_cache()
            gc.collect()
            
            # Process batch for variable-sized masks
            processed_batch = self.process_batch(batch)
            
            # Forward pass with mixed precision if enabled
            if self.use_mixed_precision:
                with autocast('cuda'):
                    outputs = self.model(**processed_batch)
                    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                # Standard precision training
                outputs = self.model(**processed_batch)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if self.rank == 0:
                progress_bar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                
                if self.use_wandb and batch_idx % 100 == 0:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, disable=self.rank != 0):
            torch.cuda.empty_cache()
            gc.collect()
            
            processed_batch = self.process_batch(batch)
            
            if self.use_mixed_precision:
                with autocast('cuda'):
                    outputs = self.model(**processed_batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**processed_batch)
                loss = outputs.loss
                
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            if self.rank == 0:
                logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss_epoch": train_loss,
                        "val_loss_epoch": val_loss
                    })
                
                # Save checkpoint if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = MODEL_DIR / f"checkpoint_epoch_{epoch}.pt"
                    
                    state_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    
                    logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        if self.use_wandb:
            wandb.finish()

def main_worker(rank, world_size):
    setup(rank, world_size)
    trainer = Trainer(rank, world_size)
    trainer.train()
    cleanup()

def main():
    setup_logging()
    
    # Override the GPU device order to use GPU_ID first
    if 'GPU_ID' in globals():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
        logging.info(f"Using GPU {GPU_ID}")
    
    world_size = min(torch.cuda.device_count(), NUM_GPUS)
    
    # Print memory usage info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_info = torch.cuda.mem_get_info(i)
            free_mem = mem_info[0] / 1024**3
            total_mem = mem_info[1] / 1024**3
            logging.info(f"GPU {i}: {free_mem:.2f}GB free / {total_mem:.2f}GB total")
    
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main() 