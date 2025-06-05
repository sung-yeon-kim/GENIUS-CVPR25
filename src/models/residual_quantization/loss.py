"""
Loss functions for Residual Quantization model.
"""

# Standard library
import numpy as np

# Third-party
import torch
import torch.nn as nn
from torch.nn import functional as F

# Optional distributed training imports
try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# ================ Utility Functions ================

def gather_features(image_features,
                   text_features,
                   local_loss=False,
                   gather_with_grad=False,
                   rank=0,
                   world_size=1,
                   use_horovod=False):
    """Gather features from all workers for distributed training.
    
    Args:
        image_features: Image features tensor
        text_features: Text features tensor
        local_loss: Whether to compute loss locally
        gather_with_grad: Whether to gather with gradients
        rank: Current process rank
        world_size: Total number of processes
        use_horovod: Whether to use Horovod for distributed training
        
    Returns:
        Tuple of gathered image and text features
    """
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # Ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # Gather tensors from all GPUs
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # Ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all workers with support for backward propagation.
    
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]

def all_gather_with_grad(tensors):
    """Perform all_gather operation on the provided tensors.
    
    Graph remains connected for backward grad computation.
    
    Args:
        tensors: Input tensors to gather
        
    Returns:
        Concatenated gathered tensors
    """
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensors
    tensor_all = GatherLayer.apply(tensors)
    return torch.cat(tensor_all, dim=0)

def pdist(A, B, squared=False, eps=1e-12):
    """Compute pairwise distances between two sets of vectors.
    
    Args:
        A: First set of vectors
        B: Second set of vectors
        squared: Whether to return squared distances
        eps: Small constant for numerical stability
        
    Returns:
        Matrix of pairwise distances
    """
    D = A.pow(2).sum(1) + (-2) * B.mm(A.t())
    D = (B.pow(2).sum(1) + D.t()).clamp(min=eps)
    
    if not squared:
        D = D.sqrt()
        
    if torch.equal(A, B):
        D = D.clone()
        D[range(len(A)), range(len(A))] = 0
        
    return D

# ================ Loss Functions ================

class ClipLoss(nn.Module):
    """CLIP-style contrastive loss with optional label smoothing and margin.
    
    Args:
        local_loss: Whether to compute loss locally
        gather_with_grad: Whether to gather with gradients
        cache_labels: Whether to cache labels
        rank: Current process rank
        world_size: Total number of processes
        label_smoothing: Label smoothing factor
        margin: Margin for contrastive loss
        use_horovod: Whether to use Horovod for distributed training
    """
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        label_smoothing=0,
        margin=0.05,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.ls = label_smoothing
        self.mg = margin

        # Cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, ground_labels=None):
        """Compute CLIP loss.
        
        Args:
            image_features: Image features
            text_features: Text features
            logit_scale: Scaling factor for logits
            ground_labels: Optional ground truth labels
            
        Returns:
            Loss value
        """
        device = image_features.device
        
        # Gather features for distributed training
        if self.world_size > 1:
            all_image_features = all_gather_with_grad(image_features)
            all_text_features = all_gather_with_grad(text_features)
            ground_labels = all_gather_with_grad(ground_labels)
            
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # Calculate ground truth and cache if enabled
        num_logits = logits_per_image.shape[0]

        if ground_labels is not None:
            if self.world_size > 1:
                ground_labels_repeated = ground_labels.view(1, -1).repeat(all_image_features.shape[0], 1)
            else:
                ground_labels_repeated = ground_labels.view(1, -1).repeat(image_features.shape[0], 1)
            equal_labels = (ground_labels_repeated == ground_labels.view(-1, 1)).type(torch.float)
                
            # Apply margin
            margin_matrix = (1-equal_labels) * self.mg * logit_scale
            logits_per_image = logits_per_image + margin_matrix
            logits_per_text = logits_per_text + margin_matrix

            # Apply label smoothing
            num_pos = equal_labels.sum(1, keepdim=True)
            smooth_labels = (1-self.ls) * equal_labels / num_pos + (self.ls / (num_logits - num_pos)) * (1-equal_labels)

            # Compute softmax with numerical stability
            image_logit_exp = torch.exp(logits_per_image - torch.max(logits_per_image, dim=1, keepdim=True).values)
            text_logit_exp = torch.exp(logits_per_text - torch.max(logits_per_text, dim=1, keepdim=True).values)

            image_logit_exp = image_logit_exp / (image_logit_exp.sum(1, keepdim=True))
            text_logit_exp = text_logit_exp / (text_logit_exp.sum(1, keepdim=True))

            image_logit_exp = -torch.log(image_logit_exp + 1e-6)
            text_logit_exp = -torch.log(text_logit_exp + 1e-6)

            total_loss = (smooth_labels * image_logit_exp).sum(1).mean() \
                        + (smooth_labels * text_logit_exp).sum(1).mean()

        else:
            if self.prev_num_logits != num_logits or device not in self.labels:
                # Generate labels
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

            total_loss = (F.cross_entropy(logits_per_image, labels) + 
                         F.cross_entropy(logits_per_text, labels)) / 2

        return total_loss