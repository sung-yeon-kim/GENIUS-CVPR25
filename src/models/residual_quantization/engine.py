"""
Training and evaluation engine for Residual Quantization model.
"""

# Standard library
import gc

# Third-party
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Local modules
from models.uniir_clip import utils

# ================ Training Functions ================

def train_one_epoch(model, clip_model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config, feature_dict=None):
    """Train the model for one epoch.
    
    Args:
        model: The model to train
        clip_model: CLIP model for feature extraction
        data_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        epoch: Current epoch number
        gpu_id: GPU device ID
        scheduler: Learning rate scheduler
        global_step: Global training step counter
        scaler: Gradient scaler for mixed precision training
        config: Configuration object
        feature_dict: Optional dictionary of pre-computed features
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # Initialize metrics
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("accuracy_org", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("cl_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("mse_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("rq_loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("num_collision", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("perplexity", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))

    header = f"Train Epoch: [{epoch}]"
    print_freq = config.trainer_config.print_freq
    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0

    model.module.codebook_reset()
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Forward pass with mixed precision
        with autocast(enabled=True):
            outputs = model(batch)
            loss = outputs["loss"]

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            scheduler.step()
            accumulation_counter = 0

        # Update metrics
        metric_logger.update(accuracy=outputs["accuracy"].item())
        metric_logger.update(accuracy_org=outputs["accuracy_org"].item())
        metric_logger.update(loss=loss.item() * accumulation_steps)
        metric_logger.update(cl_loss=outputs["cl_loss"].item() * accumulation_steps)
        metric_logger.update(mse_loss=outputs["mse_loss"].item() * accumulation_steps)
        metric_logger.update(rq_loss=outputs["rq_loss"].item() * accumulation_steps)
        metric_logger.update(num_collision=outputs["num_collision"])
        metric_logger.update(perplexity=outputs["perplexity"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_e2e(model, clip_model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config, 
                    discriminator=None, opt_disc=None, scaler_disc=None, scheduler_disc=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("cl_loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("mse_loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("rq_loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("gan_loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("num_collision", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("perplexity", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0

    model.train()
    model.module.codebook_reset()

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # autocast for mixed precision
        with autocast(enabled=False):
            outputs = model(batch, logit_scale=clip_model.get_logit_scale())
            loss = outputs["loss"]
            
            if discriminator is not None:
                disc_factor = adopt_weight(1, epoch * len(data_loader)+i, threshold=500)

                disc_real = discriminator(outputs["org_feateture"].detach().clone())
                disc_fake = discriminator(outputs["decode_feature"].detach().clone())
                g_loss = -disc_factor * 0.01 * torch.mean(disc_fake)
                outputs['cl_loss'] += g_loss * 0
                loss += g_loss * 0
                
                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                gan_loss = 0.5 * disc_factor * (d_loss_real + d_loss_fake)  * 0

        # Scale the loss by the number of accumulation steps since backward averages the gradients.
        loss = loss / accumulation_steps

        # Use scaler for backward
        scaler.scale(loss).backward(retain_graph=True)
        if discriminator is not None:
            scaler_disc.scale(gan_loss).backward()
        
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        scaler_disc.unscale_(opt_disc)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if discriminator is not None:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            scaler_disc.step(opt_disc)
            scaler_disc.update()

            optimizer.zero_grad()
            opt_disc.zero_grad()
            
            scheduler.step()
            scheduler_disc.step()
            accumulation_counter = 0

        metric_logger.update(accuracy=outputs["accuracy"].item())  # We scale back the loss for logging.
        metric_logger.update(loss=loss.item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(cl_loss=outputs["cl_loss"].item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(mse_loss=outputs["mse_loss"].item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(rq_loss=outputs["rq_loss"].item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(gan_loss=gan_loss.item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(num_collision=outputs["num_collision"])
        metric_logger.update(perplexity=outputs["perplexity"])
        metric_logger.update(lr=optimizer.param_groups[1]["lr"])  # TODO: might need to loop through all param groups

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ================ Evaluation Functions ================

@torch.no_grad()
def eval_engine(model, clip_model, data_loader, gpu_id, config):
    """Evaluate the model.
    
    Args:
        model: The model to evaluate
        clip_model: CLIP model for feature extraction
        data_loader: DataLoader for evaluation data
        gpu_id: GPU device ID
        config: Configuration object
        
    Returns:
        Dictionary of evaluation metrics
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # Initialize metrics
    metric_logger.add_meter("accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("cl_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("mse_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("rq_loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("num_collision", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("perplexity", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    
    header = "Test:"
    print_freq = config.evaluator.print_freq

    model.eval()
    model.module.codebook_reset()
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move batch to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(gpu_id, non_blocking=True)

        # Forward pass with mixed precision
        with autocast():
            q_emb, p_emb = clip_model.get_embedding(batch)
            outputs = model((q_emb, p_emb))
            loss = outputs["loss"]

        # Update metrics
        metric_logger.update(accuracy=outputs["accuracy"].item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(cl_loss=outputs["cl_loss"].item())
        metric_logger.update(mse_loss=outputs["mse_loss"].item())
        metric_logger.update(rq_loss=outputs["rq_loss"].item())
        metric_logger.update(num_collision=outputs["num_collision"])
        metric_logger.update(perplexity=outputs["perplexity"])

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
