"""
Residual Quantization model implementation.
"""

# Standard library
import os
import math
import time
import copy
import string
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Dict, List, Union

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import faiss
import joblib
from einops import rearrange, repeat, reduce, pack, unpack
from torch.nn.utils import weight_norm
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

# Local modules
from models.uniir_clip import utils
from models.residual_quantization.loss import ClipLoss 

# ================ Loss Functions ================

class ContrastiveLoss(nn.Module):
    """Contrastive loss for feature learning.
    
    This loss function computes the contrastive loss between two sets of embeddings,
    encouraging similar items to be close and dissimilar items to be far apart in the embedding space.
    
    Args:
        temperature: Temperature parameter for softmax scaling
        metric: Similarity metric ('cos' for cosine similarity or 'euclid' for Euclidean distance)
        gather: Whether to gather embeddings from all GPUs in distributed training
    """
    def __init__(self, temperature: float = 0.01, metric: str = 'cos', gather: bool = True):
        super().__init__()
        self.temperature = temperature
        self.metric = metric
        self.gather = gather
        
    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        """Generate ground truth labels for contrastive learning.
        
        Args:
            device: Device to create labels on
            num_logits: Number of logits (batch size)
            
        Returns:
            Tensor of labels where each item is matched with itself
        """
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def forward(self, x: torch.Tensor, y: torch.Tensor, temp: Optional[float] = None) -> torch.Tensor:
        """Compute contrastive loss between two sets of embeddings.
        
        Args:
            x: First set of embeddings
            y: Second set of embeddings
            temp: Optional temperature parameter (overrides default if provided)
            
        Returns:
            Contrastive loss value
        """
        if temp is None:
            temp = self.temperature

        # Gather embeddings from all GPUs if in distributed training
        if utils.get_world_size() > 1 and self.gather:
            x = torch.cat(utils.GatherLayer.apply(x), dim=0)
            y = torch.cat(utils.GatherLayer.apply(y), dim=0)

        labels = self.get_ground_truth(x.device, x.shape[0])

        # Compute similarity based on chosen metric
        if self.metric == 'cos':
            logits_per_x = F.linear(F.normalize(x), F.normalize(y))
        elif self.metric == 'euclid':
            logits_per_x = -torch.cdist(x,y) ** 2
        else:
            raise ValueError(f'Invalid metric: {self.metric}')

        # Scale logits and compute bidirectional loss
        logits_per_x = logits_per_x / temp
        logits_per_y = logits_per_x.T

        total_loss = (F.cross_entropy(logits_per_x, labels) 
                     + F.cross_entropy(logits_per_y, labels)) / 2

        return total_loss

def cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances between two sets of vectors.
    
    Args:
        x: First set of vectors
        y: Second set of vectors
        
    Returns:
        Matrix of pairwise distances
    """
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min=0).sqrt()

# ================ Neural Network Modules ================

class Combiner(nn.Module):
    """Combiner module for fusing textual and visual information.
    
    This module combines CLIP image and text features through a series of transformations
    and a dynamic weighting mechanism to create a unified representation. (https://github.com/ABaldrati/CLIP4Cir)
    
    Args:
        clip_feature_dim: CLIP input feature dimension
        projection_dim: Dimension for projected features
        hidden_dim: Hidden layer dimension
        drop_rate: Dropout rate
    """
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int, drop_rate=0):
        super(Combiner, self).__init__()
        # Projection layers for image and text features
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.dropout3 = nn.Dropout(drop_rate)

        # Feature combination layers
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        # Dynamic weighting mechanism
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.out_projection_layer = nn.Identity()

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, 
                img_masks: Optional[torch.Tensor] = None, txt_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine image and text features.
        
        Args:
            image_features: CLIP image features
            text_features: CLIP text features
            img_masks: Optional mask for image features
            txt_masks: Optional mask for text features
            
        Returns:
            Combined features
        """
        # Apply masks if provided
        if img_masks is not None:
            image_features = image_features * img_masks
        if txt_masks is not None:
            text_features = text_features * txt_masks

        # Project features
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features))) 

        # Combine features
        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        
        # Compute dynamic weights
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        
        # Combine with original features
        output = (self.output_layer(combined_features) + 
                 dynamic_scalar * text_features + 
                 (1 - dynamic_scalar) * image_features)
        
        return self.out_projection_layer(output)

class RQ(nn.Module):
    """Residual Quantization.
    
    Args:
        config: Model configuration
        feature_dim: Feature dimension
        clip_model: CLIP model for feature extraction
        unique_code: Whether to ensure unique codes
        modality_index: Whether to use modality-specific indexing
    """
    def __init__(self,
                 config,
                 feature_dim=768,
                 clip_model=None,
                 unique_code=False,
                 modality_index=True):
        super().__init__()
        # Initialize CLIP model if provided
        if clip_model is not None:
            self.clip_model = clip_model
            for _, param in self.clip_model.named_parameters():
                param.requires_grad = False
            self.clip_model.eval()

        # Initialize model parameters
        self.iter = 0
        self.unique_code = unique_code
        self.modality_index = modality_index
        
        # Initialize components
        self.encoder = Combiner(feature_dim, 2560, 5120)
        self.critic = nn.MSELoss(reduction='mean')
        self.contra_loss = ContrastiveLoss(temperature=0.01)

        # Set up codebook configuration
        self.codebook_vocab = config.codebook_config.codebook_vocab
        self.codebook_level = (config.codebook_config.codebook_level + 1 
                             if self.modality_index 
                             else config.codebook_config.codebook_level)
        self.level_indicators = list(string.ascii_lowercase[:self.codebook_level])

        # Initialize Residual Vector Quantization
        if self.modality_index:
            self.vq = VectorQuantize(
                dim=feature_dim,
                codebook_dim=feature_dim,  
                codebook_size=3, 
                kmeans_init=True,
                kmeans_iters=1000, 
                learnable_codebook=False,
                ema_update=True, 
                threshold_ema_dead_code=0,
                decay=0.9
            )

        self.residual_rq = ResidualVQ(
            dim=feature_dim,
            codebook_dim=feature_dim, 
            num_quantizers=self.codebook_level, 
            codebook_size=self.codebook_vocab, 
            kmeans_init=True,
            kmeans_iters=1000, 
            learnable_codebook=False,
            ema_update=True, 
            threshold_ema_dead_code=2,
            decay=0.9
        )

        if self.modality_index:
            self.residual_rq.layers[0] = self.vq 

        # Initialize tracking variables
        self.gather_embedding = True
        self.is_history = {}
        self.codebook_reset()
        
    def get_img_preprocess_fn(self):
        """Get image preprocessing function from CLIP model."""
        return self.clip_model.get_img_preprocess_fn()
    
    def get_tokenizer(self):
        """Get tokenizer from CLIP model."""
        return self.clip_model.get_tokenizer()

    def codebook_reset(self):
        """Reset codebook tracking variables."""
        self.codebag = {}
        self.num_collision = 0

    def collision_update(self, codes: torch.Tensor, id_list: List[int]):
        """Update collision tracking for codes.
        
        Args:
            codes: Generated codes
            id_list: List of IDs
        """
        for i in range(len(codes)):
            code = str(tuple(codes[i].cpu().numpy().tolist()))
            id = str(id_list[i])
            
            if code in self.codebag:
                if self.codebag[code] != id:
                    self.num_collision += 1
                    self.codebag[code] = id
            else:
                self.codebag[code] = id

    def transform_row(self, row: List[int], separator: str = '') -> str:
        """Transform a row of codes into token format.
        
        Args:
            row: List of code values
            separator: Separator string between codes
            
        Returns:
            String representation of codes
        """
        transformed_row = []
        for l, level_indicator in enumerate(self.level_indicators):
            new_value = row[l]
            transformed_row.append(f"<{level_indicator}{new_value}>")
        return separator.join(transformed_row)

    def compute_single_batch(self, batch: Tuple, logit_scale: Optional[float] = None) -> Dict:
        """Compute loss and metrics for a single batch.
        
        Args:
            batch: Input batch containing query, pool, instructions, and IDs
            logit_scale: Optional scaling factor for logits
            
        Returns:
            Dictionary containing loss values and metrics
        """
        # Unpack batch data
        query, pool, instruct, h_qid = batch
        gpu_id = utils.get_rank()

        # Prepare masks and embeddings
        h_qid = h_qid.view(-1)
        q_img_mask = query['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        q_txt_mask = query['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_img_mask = pool['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_txt_mask = pool['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)

        q_img_emb = query['img_emb'].view(-1, query['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        q_txt_emb = query['txt_emb'].view(-1, query['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_img_emb = pool['img_emb'].view(-1, pool['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_txt_emb = pool['txt_emb'].view(-1, pool['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)

        bs = len(q_img_emb)
        device = q_img_emb.device

        # Concatenate query and pool embeddings
        all_img_emb = torch.cat((q_img_emb, p_img_emb), dim=0)
        all_txt_emb = torch.cat((q_txt_emb, p_txt_emb), dim=0)
        all_img_mask = torch.cat((q_img_mask, p_img_mask), dim=0)
        all_txt_mask = torch.cat((q_txt_mask, p_txt_mask), dim=0)

        # Combine features
        encode_feature = self.encoder(all_img_emb, all_txt_emb, all_img_mask, all_txt_mask)
        encode_feature = F.normalize(encode_feature)
        q_emb, p_emb = encode_feature[:bs], encode_feature[bs:]

        # Generate hash IDs for checking uniqueness
        qid_list = [hash(q_emb[i]) for i in range(len(q_emb))]
        p_did_list = [hash(p_emb[i]) for i in range(len(p_emb))]
        all_id_list = qid_list + p_did_list

        # Perform residual quantization
        quant, Is, rq_loss = self.residual_rq(encode_feature.unsqueeze(0), rand_quantize_dropout_fixed_seed=2023)
        quant = quant.squeeze(0)
        Is = Is.squeeze(0)

        quant = F.normalize(quant)
        q_decode, p_decode = quant[:bs], quant[bs:]

        # Gather embeddings for compute contrastive score if in distributed training
        if self.gather_embedding:
            dist.barrier()
            all_p_emb = torch.cat(utils.GatherLayer.apply(p_emb), dim=0)
            all_p_decode = torch.cat(utils.GatherLayer.apply(p_decode), dim=0)
            all_h_qid = torch.cat(utils.GatherLayer.apply(h_qid), dim=0)
        else:
            all_p_emb = p_emb
            all_p_decode = p_decode
            all_h_qid = h_qid

        # Compute similarity scores
        score_org = torch.matmul(q_emb, all_p_emb.t()) # [bs, bs]
        score = torch.matmul(q_emb, all_p_decode.t()) # [bs, bs]
        sim_targets = h_qid

        # Compute accuracy
        _, max_idxs_org = torch.max(score_org, 1)
        max_idxs_org = all_h_qid[max_idxs_org]
        _, max_idxs = torch.max(score, 1)
        max_idxs = all_h_qid[max_idxs]
        accuracy_org = (max_idxs_org == sim_targets).sum() / bs
        accuracy = (max_idxs == sim_targets).sum() / bs
        
        # Compute losses
        rq_loss = 1e2 * rq_loss.mean()
        mse_loss = 1e2 * self.critic(q_decode, p_decode)
        cl_loss = self.contra_loss(q_emb, p_emb)

        # Compute perplexity
        perplexity = 0
        if Is.dim() != 1:
            for i in range(self.codebook_level):
                encodings = F.one_hot(Is[:, i].long(), self.codebook_vocab).type(quant.dtype)
                avg_probs = torch.mean(encodings, dim=0)
                perplexity_per_level = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-6)))
                perplexity += perplexity_per_level
            perplexity /= self.codebook_level
            
        loss = cl_loss + rq_loss + mse_loss

        self.collision_update(Is, all_id_list)

        if self.iter % 200 == 0:
            print('Query ID: ' + str(self.transform_row(Is[0])))
            print('Doc ID: ' + str(self.transform_row(Is[bs])))
        
        self.iter += 1

        # Prepare outputs
        outputs = {
            'Is': Is,
            'org_feateture': F.normalize(encode_feature),
            'quant': F.normalize(quant),
            'loss': loss,
            'rq_loss': rq_loss,
            'cl_loss': cl_loss,
            'mse_loss': mse_loss,
            'num_collision': self.num_collision,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'accuracy_org': accuracy_org
        }
        
        return outputs

    def encode_mbeir_batch(self, batch: Dict, code_output: bool = False, 
                          encode_output: bool = False) -> Tuple:
        """Encode a batch of MBEIR data.
        
        Args:
            batch: Input batch
            code_output: Whether to return code output
            encode_output: Whether to return encoded output
            
        Returns:
            Tuple containing outputs and IDs
        """
        # Get hashed id_list
        id_list = batch.get("did_list") or batch.get("qid_list")
        assert id_list is not None, "id_list must be provided."
        assert isinstance(id_list[0], int), "id_list must be hashed to int."

        # Compute embeddings
        img_emb, txt_emb = self.clip_model.encode_multimodal_input(
            batch["image_batched"],
            batch["txt_batched"],
        )
        img_mask = batch["image_mask_batched"].unsqueeze(-1)
        txt_mask = batch["txt_mask_batched"].unsqueeze(-1)

        assert img_emb.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."
        
        # Get outputs based on requested types
        if code_output and encode_output:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)
            return output['code'], output['encode'], id_list
        elif code_output:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)['code']
        else:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)['quant']
        return output, id_list

    def encode_extracted_batch(self, batch: Tuple, code_output: bool = False, 
                             encode_output: bool = False) -> Tuple:
        """Encode a batch of extracted features.
        
        Args:
            batch: Input batch containing pool and IDs
            code_output: Whether to return code output
            encode_output: Whether to return encoded output
            
        Returns:
            Tuple containing outputs and IDs
        """
        pool, id_list = batch
        gpu_id = utils.get_rank()

        # Prepare masks and embeddings
        img_mask = pool['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        txt_mask = pool['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        img_emb = pool['img_emb'].view(-1, pool['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        txt_emb = pool['txt_emb'].view(-1, pool['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)

        assert img_emb.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."
        
        # Get outputs based on requested types
        if code_output and encode_output:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)
            return output['code'], output['rerank'], id_list
        elif code_output:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)['code']
        else:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)['quant']
        return output, id_list
    
    @torch.no_grad()
    def inference(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, 
                 img_mask: Optional[torch.Tensor] = None, 
                 txt_mask: Optional[torch.Tensor] = None) -> Dict:
        """Perform inference on input embeddings.
        
        Args:
            img_emb: Image embeddings
            txt_emb: Text embeddings
            img_mask: Optional mask for image features
            txt_mask: Optional mask for text features
            
        Returns:
            Dictionary containing various outputs
        """
        # Encode features
        encode_feature = self.encoder(img_emb, txt_emb, img_mask, txt_mask)
        encode = F.normalize(encode_feature)
        
        # Perform quantization
        quant, Is, _ = self.residual_rq(encode)
        quant = F.normalize(quant)
        rerank_feature = F.normalize(img_emb * img_mask + txt_emb * txt_mask)

        # Handle single sample case
        if quant.shape[0] == 1:
            quant = quant.squeeze(0)
            Is = Is.squeeze(0)

        # Handle unique code tracking
        if self.unique_code:
            updated_Is = []
            for i in range(Is.shape[0]):
                is_key = tuple(Is[i].cpu().numpy().tolist())
                if is_key in self.is_history:
                    stored_encode = self.is_history[is_key]['encode'].to(encode.device)
                    if not torch.allclose(stored_encode, encode[i], atol=1e-5):
                        self.is_history[is_key]['count'] += 1
                    count = self.is_history[is_key]['count']
                else:
                    self.is_history[is_key] = {'encode': encode[i].detach().cpu(), 'count': 0}
                    count = 0   
                
                updated_Is.append(torch.cat([Is[i], torch.tensor([count], device=Is.device)]))
            Is = torch.stack(updated_Is)

        # Prepare outputs
        outputs = {
            'code': Is,
            'quant': quant,
            'encode': encode,
            'rerank': rerank_feature
        }
        return outputs
    
    def forward(self, 
                input: Optional[Tuple] = None, 
                evaluation: bool = False,
                encode_mbeir_batch: bool = True,
                code_output: bool = False, 
                encode_output: bool = False, 
                logit_scale: Optional[float] = None) -> Union[Dict, Tuple]:
        """Forward pass of the model.
        
        Args:
            input: Input batch
            evaluation: Whether in evaluation mode
            encode_mbeir_batch: Whether to encode MBEIR batch
            code_output: Whether to return code output
            encode_output: Whether to return encoded output
            logit_scale: Optional scaling factor for logits
            
        Returns:
            Model outputs
        """
        if evaluation:
            if encode_mbeir_batch:
                return self.encode_mbeir_batch(input, code_output=code_output, encode_output=encode_output)
            else:
                return self.encode_extracted_batch(input, code_output=code_output, encode_output=encode_output)
        return self.compute_single_batch(input, logit_scale=logit_scale)