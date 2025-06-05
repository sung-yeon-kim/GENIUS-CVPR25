"""
Training Code for Residual Quantization

This module implements the training pipeline for the Residual Quantization model.
It includes functions for:
- Setting up distributed training
- Creating and managing optimizers
- Training loop implementation
- Checkpoint management
- Logging and evaluation
"""

# Standard library
import argparse
import logging
import os
import random
import gc
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf
from dotenv import load_dotenv
import wandb

# Local modules
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
    build_distributed_sampler_list,
    build_dataloader_list,
)
from data.mbeir_dataset import MBEIRDictInstructioneDataset
from models.uniir_clip import utils
from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion
from models.residual_quantization.residual_quantization import RQ
from models.residual_quantization.engine import train_one_epoch, eval_engine
from models.utils import cosine_warmup_scheduler

# Set up logger
logger = logging.getLogger()

# ================ Utility Functions ================

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def filter_parameters(model: torch.nn.Module, condition_fn: callable) -> List[torch.nn.Parameter]:
    """
    Filter model parameters based on a condition function.
    
    Args:
        model: PyTorch model
        condition_fn: Function that takes (name, param) and returns bool
        
    Returns:
        List of filtered parameters
    """
    named_parameters = model.named_parameters()
    return [p for n, p in named_parameters if condition_fn(n, p) and p.requires_grad]

# ================ Model Training Functions ================

def create_optimizer(
    token_embedding_params: List[torch.nn.Parameter],
    gain_or_bias_params: List[torch.nn.Parameter],
    rest_params: List[torch.nn.Parameter],
    config: Any
) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates for different parameter groups.
    
    Args:
        token_embedding_params: Parameters for token embeddings
        gain_or_bias_params: Parameters for gain/bias terms
        rest_params: Remaining model parameters
        config: Configuration object containing training parameters
        
    Returns:
        Configured AdamW optimizer
    """
    return optim.AdamW(
        [
            {"params": token_embedding_params, "lr": config.trainer_config.learning_rate, "weight_decay": 0},
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": 0.2},
        ],
        lr=config.trainer_config.learning_rate,
        eps=1.0e-6,
    )

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    scaler: GradScaler,
    config: Any
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch: Current training epoch
        scaler: Gradient scaler for mixed precision training
        config: Configuration object
    """
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
    save_obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "scaler": scaler.state_dict(),
    }
    checkpoint_path = os.path.join(config.genir_dir, ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def log_results(
    train_stats: Optional[Dict[str, float]],
    val_stats: Optional[Dict[str, float]],
    test_stats: Optional[Dict[str, float]],
    epoch: Optional[int] = None,
    best_epoch: Optional[int] = None
) -> Dict[str, float]:
    """
    Log training, validation, and test statistics.
    
    Args:
        train_stats: Training statistics
        val_stats: Validation statistics
        test_stats: Test statistics
        epoch: Current epoch number
        best_epoch: Best epoch number
        
    Returns:
        Dictionary of logged statistics
    """
    log_stats = {}
    if train_stats:
        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
    if val_stats:
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
    if test_stats:
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
    if epoch is not None:
        log_stats["epoch"] = epoch
    if best_epoch is not None:
        log_stats["best_epoch"] = best_epoch
    return log_stats

# ================ Main Training Loop ================

def train(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    clip_model: CLIPNoFusion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    config: Any,
    epoch: int,
) -> None:
    """
    Main training loop.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Model to train
        model_without_ddp: Model without distributed data parallel wrapper
        clip_model: CLIP model for feature extraction
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision training
        config: Configuration object
        epoch: Starting epoch number
    """
    gpu_id = config.dist_config.gpu_id
    is_distributed_mode = config.dist_config.distributed_mode
    global_step = 0
    model.zero_grad()

    if epoch != 0:
        print(f"Resuming training from epoch {epoch}")
        
    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        # Set different seed for different epoch
        if is_distributed_mode:
            train_loader.sampler.set_epoch(epoch)

        # Training phase
        train_stats = train_one_epoch(
            model,
            clip_model,
            train_loader,
            optimizer,
            epoch,
            gpu_id,
            scheduler,
            global_step,
            scaler,
            config,
        )
        gc.collect()

        # Evaluation and checkpointing
        eval_freq = config.evaluator.eval_freq
        save_freq = config.evaluator.save_freq
        
        if epoch % save_freq == 0 and utils.is_main_process():
            save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
            
        if val_loader is not None and epoch % eval_freq == 0 and epoch >= config.evaluator.eval_start:
            val_status = eval_engine(model, clip_model, val_loader, gpu_id, config)
            log_stats = log_results(train_stats, val_status, None, epoch)
        else:
            log_stats = log_results(train_stats, None, None, epoch)

        # Logging
        if utils.is_main_process() and config.wandb_config.enabled:
            wandb.log(log_stats)

        dist.barrier()
        torch.cuda.empty_cache()

# ================ Main Function ================

def main(config: Any) -> None:
    """
    Main function to set up and run training.
    
    Args:
        config: Configuration object containing all training parameters
    """
    # Initialize distributed training
    is_distributed_mode = config.dist_config.distributed_mode
    seed = config.seed + utils.get_rank()
    set_seed(seed)
    cudnn.benchmark = True

    # Initialize models
    print("Creating Residual Quantization model...")
    model_config = config.model
    
    # Load CLIP model
    pretrained_clip_model_dir = os.path.join(config.genir_dir, model_config.pretrained_clip_model_dir)
    logger.info(f"Downloading CLIP model to {pretrained_clip_model_dir}...")
    clip_model = CLIPNoFusion(
        model_name=model_config.clip_vision_model_name,
        download_root=pretrained_clip_model_dir,
        config=config,
    )
    clip_model.float()
    clip_model.eval()

    # Initialize RQ model
    model = RQ(config=config)

    # Set up optimizer
    tok_embeddings = lambda n, p: '_codebook' in n or 'tok_embeddings' in n
    exclude_condition = lambda n, p: (p.ndim < 2 or any(sub in n for sub in ["bn", "ln", "bias", "logit_scale"])) and not tok_embeddings(n, p)
    include_condition = lambda n, p: not exclude_condition(n, p) and not tok_embeddings(n, p)
    
    # Filter parameters based on conditions
    tok_embeddings_params = filter_parameters(model, tok_embeddings)
    gain_or_bias_params = filter_parameters(model, exclude_condition)
    rest_params = filter_parameters(model, include_condition)
    
    # Create optimizer and gradient scaler
    optimizer = create_optimizer(tok_embeddings_params, gain_or_bias_params, rest_params, config)
    scaler = GradScaler()

    # Load pretrained weights if specified
    if model_config.pretrained_config.using_pretrained:
        pretrained_path = os.path.join(
            config.genir_dir,
            model_config.pretrained_config.pretrained_dir,
            model_config.pretrained_config.pretrained_name
        )
        assert os.path.exists(pretrained_path), f"Checkpoint file {pretrained_path} does not exist."
        logger.info(f"Loading CLIPScoreFusion checkpoint from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        clip_model.load_state_dict(checkpoint["model"])

    # Move model to GPU and wrap with DDP if needed
    model.train()
    model = model.to(config.dist_config.gpu_id)
    model_without_ddp = model
    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id], find_unused_parameters=True)
        model_without_ddp = model.module

    # Prepare datasets and dataloaders
    logger.info("Preparing dataset...")
    logger.info(f"Loading dataset from {config.mbeir_data_dir}{config.data_config.train_query_data_path}...")

    # Get preprocessing functions and tokenizers
    img_preprocess_fn = clip_model.get_img_preprocess_fn()
    clip_tokenizer = clip_model.get_tokenizer()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()    
    
    # Set up data directories
    query_dir = os.path.join(config.genir_dir, config.codebook_config.query_path)
    pool_dir = os.path.join(config.genir_dir, config.codebook_config.pool_path)
    
    # Create training dataset and dataloader
    train_dataset = MBEIRDictInstructioneDataset(
        mbeir_data_dir=config.mbeir_data_dir,
        query_data_path=config.data_config.train_query_data_path,
        cand_pool_path=config.data_config.train_cand_pool_path,
        query_instruct_path=config.data_config.query_instruct_path,
        query_dict_dir=query_dir,
        pool_dict_dir=pool_dir,
        clip_tokenizer=clip_tokenizer,
        return_instruct=True
    )
    
    # Set up distributed sampler for training
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )
    
    # Create training dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader_config.train_batch_size,
        num_workers=config.dataloader_config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
    )

    # Create validation dataset and dataloader if enabled
    valid_loader = None
    if config.evaluator.enable_eval:
        # Build validation dataset
        in_batch_val_dataset, in_batch_val_collector = build_mbeir_dataset_from_config(
            config=config,
            tokenizer=clip_tokenizer,
            img_preprocess_fn=img_preprocess_fn,
            dataset_type=DatasetType.IN_BATCH_VAL,
        )
        
        # Set up distributed sampler for validation
        in_batch_val_sampler = DistributedSampler(
            dataset=in_batch_val_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False,
        )
        
        # Create validation dataloader
        valid_loader = DataLoader(
            dataset=in_batch_val_dataset,
            batch_size=config.dataloader_config.valid_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=True,
            sampler=in_batch_val_sampler,
            shuffle=False,
            collate_fn=in_batch_val_collector,
            drop_last=True,
        )
    else:
        print("In-batch validation is disabled.")

    # Set up learning rate scheduler
    t_total = len(train_loader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=t_total, eta_min=1e-5)

    # Resume training if specified
    epoch = 0
    if model_config.ckpt_config.resume_training:
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming training from epoch {epoch}")

    # Start training
    dist.barrier()
    train(
        train_loader=train_loader,
        val_loader=valid_loader,
        model=model,
        model_without_ddp=model_without_ddp,
        clip_model=clip_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        epoch=epoch,
    )

# ================ Script Entry Point ================

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Residual Quantization model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Run training
    main(config)

    # Cleanup
    if config.wandb_config.enabled and utils.is_main_process():
        wandb.finish()
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
