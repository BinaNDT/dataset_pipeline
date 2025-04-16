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

sys.path.append(str(Path(__file__).parent))
from config import *
from utils.dataset import create_data_loaders

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
        
        # Initialize model
        config = Mask2FormerConfig.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=NUM_CLASSES,
            num_queries=NUM_QUERIES
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
        
        # Create data loaders
        self.train_loader, self.val_loader, _ = create_data_loaders(
            IMAGES_DIR,
            ANNOTATIONS_FILE,
            BATCH_SIZE_PER_GPU
        )
        
        if self.world_size > 1:
            self.train_loader.sampler = DistributedSampler(self.train_loader.dataset)
            self.val_loader.sampler = DistributedSampler(self.val_loader.dataset)
        
        num_training_steps = len(self.train_loader) * NUM_EPOCHS
        num_warmup_steps = num_training_steps // 10
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize WandB
        if rank == 0:
            wandb.init(
                project="building-damage-segmentation",
                config={
                    "learning_rate": LEARNING_RATE,
                    "epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE_PER_GPU * world_size,
                    "model": MODEL_CHECKPOINT,
                    "num_classes": NUM_CLASSES
                }
            )
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)
        
        progress_bar = tqdm(self.train_loader, disable=self.rank != 0)
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if self.rank == 0:
                progress_bar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                
                if batch_idx % 100 == 0:
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
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
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

def main_worker(rank, world_size):
    setup(rank, world_size)
    trainer = Trainer(rank, world_size)
    trainer.train()
    cleanup()

def main():
    setup_logging()
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main() 