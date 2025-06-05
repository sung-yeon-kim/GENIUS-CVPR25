import torch
from torch.cuda.amp import autocast
from models.uniir_clip import utils

def train_one_epoch(model, clip_model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("R_at_1", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("Level1_acc", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("Level12_acc", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("Level123_acc", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0

    model.train()
    model.module.quantizer.eval()

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # autocast for mixed precision
        with autocast(enabled=True):
            outputs = model(batch)
            loss = outputs["loss"]

        # Scale the loss by the number of accumulation steps since backward averages the gradients.
        loss = loss / accumulation_steps

        # Use scaler for backward
        scaler.scale(loss).backward()
        
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            for param in model.parameters():
                param.grad = None
            scheduler.step()
            accumulation_counter = 0

        metric_logger.update(R_at_1=outputs["R_at_1"].item())  # We scale back the loss for logging.
        metric_logger.update(Level1_acc=outputs["Level1_acc"].item())  # We scale back the loss for logging.
        metric_logger.update(Level12_acc=outputs["Level12_acc"].item())  # We scale back the loss for logging.
        metric_logger.update(Level123_acc=outputs["Level123_acc"].item())  # We scale back the loss for logging.
        metric_logger.update(loss=outputs["loss"].item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # TODO: might need to loop through all param groups

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_engine(model, clip_model, data_loader, gpu_id, config):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("R_at_1", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("Level1_acc", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("Level12_acc", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("Level123_acc", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("lm_loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Test:"
    print_freq = config.evaluator.print_freq

    model.eval()
    num_samples = 0
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(gpu_id, non_blocking=True)  # Batch is a dictionary of tensors

        num_samples += len(batch)

        # autocast for mixed precision
        with autocast():
            outputs = model(batch)
            loss = outputs["loss"]


        metric_logger.update(R_at_1=outputs["R_at_1"].item())  # We scale back the loss for logging.
        metric_logger.update(Level1_acc=outputs["Level1_acc"].item())  # We scale back the loss for logging.
        metric_logger.update(Level12_acc=outputs["Level12_acc"].item())  # We scale back the loss for logging.
        metric_logger.update(Level123_acc=outputs["Level123_acc"].item())  # We scale back the loss for logging.
        metric_logger.update(lm_loss=outputs["lm_loss"].item())  # We scale back the loss for logging.

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
