"""
Generative Retriever Model Implementation
"""

# Standard library
import os
import re
import math
import random 
import string
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Union, List, Dict, Any
import pickle

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from einops import rearrange, repeat, reduce, pack, unpack
from torch.nn.utils import weight_norm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    LogitsProcessor,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.file_utils import ModelOutput
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from tqdm import tqdm

# Local modules
from models.uniir_clip import utils
from models.generative_retriever.beam import Trie, MarisaTrie
import models.generative_retriever.trie_cpp as Trie_Cpp
from models.residual_quantization.residual_quantization import RQ
from models.residual_quantization.loss import ClipLoss

# ================ Constants ================
IGNORE_INDEX = -100

# ================ Loss Functions ================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for feature learning.
    
    Args:
        temperature: Temperature parameter for softmax
        metric: Similarity metric ('cos' or 'euclid')
        bidirection: Whether to compute loss in both directions
        gather: Whether to gather embeddings from all GPUs
    """
    def __init__(self, temperature: float = 0.01, metric: str = 'cos', bidirection: bool = True, gather: bool = True):
        super().__init__()
        self.temperature = temperature
        self.metric = metric
        self.gather = gather
        self.bidirection = bidirection
        
    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        """
        Generate ground truth labels.
        
        Args:
            device: Device to create labels on
            num_logits: Number of logits
            
        Returns:
            Tensor of labels
        """
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def forward(self, x: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None, logit: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            x: First set of embeddings
            y: Second set of embeddings
            logit: Pre-computed logits (optional)
            
        Returns:
            Contrastive loss value
        """
        if logit is None:
            if utils.get_world_size() > 1 and self.gather:
                x = torch.cat(utils.GatherLayer.apply(x), dim=0)
                y = torch.cat(utils.GatherLayer.apply(y), dim=0)
            
            assert x is not None and y is not None
            if self.metric == 'cos':
                logits_per_x = F.linear(F.normalize(x), F.normalize(y))
            elif self.metric == 'euclid':
                logits_per_x = -torch.cdist(x,y) ** 2
            else:
                raise ValueError(f'Invalid metric: {self.metric}')
            labels = self.get_ground_truth(x.device, x.shape[0])
        
        else:
            logits_per_x = logit
            labels = self.get_ground_truth(logit.device, logit.shape[0])

        logits_per_x = logits_per_x / self.temperature
        logits_per_y = logits_per_x.T

        if self.bidirection:
            total_loss = (F.cross_entropy(logits_per_x, labels) + F.cross_entropy(logits_per_y, labels)) / 2
        else:
            total_loss = F.cross_entropy(logits_per_x, labels)
        return total_loss
        
# ================ Main Model ================

class T5ForGenerativeRetrieval(nn.Module):
    """
    T5-based generative retrieval model that combines T5 with residual quantization.

    Args:
        config: Model configuration
        tokenizer: Tokenizer for text processing
        clip_model: CLIP model for feature extraction
        new_tokenizer: Whether to create a new tokenizer
        init_rq_codebook: Whether to initialize the residual quantization codebook
    """
    def __init__(self, config=None, tokenizer=None, clip_model=None, new_tokenizer=True, init_rq_codebook=True):
        super().__init__()
        self.return_logits = False
        
        # Initialize CLIP model if provided
        if clip_model is not None:
            self.clip_model = clip_model
            for _, param in self.clip_model.named_parameters():
                param.requires_grad = False
            self.clip_model.eval()
            
        # Initialize model components
        self.config = config
        self.quantizer = RQ(config=config, clip_model=clip_model)
        rq_model_path = os.path.join(config.genir_dir, config.codebook_config.quantizer_path)
        self.quantizer.load_state_dict(torch.load(rq_model_path, map_location=torch.device('cpu'))["model"], strict=False)
        self.quantizer.eval()
        for _, param in self.quantizer.named_parameters():
            param.requires_grad = False
        self.modality_index = self.quantizer.modality_index
        
        # Initialize T5 model
        t5_config = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").config
        t5_config.num_layers = 0
        self.id_generator = T5ForConditionalGeneration(t5_config)
        self.id_generator.config.decoder_start_token_id = 0

        # Initialize embedding projectors
        self.num_prefix = 30
        self.embed_projector = nn.Sequential(nn.Linear(768, t5_config.d_model * self.num_prefix))
        self.embed_projector.train()
        for _, param in self.embed_projector.named_parameters():
            param.requires_grad = True

        # Initialize loss functions
        self.contra_projector = nn.Linear(768, t5_config.d_model)
        self.contra_loss = ContrastiveLoss(0.01)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0)
        
        # Set up codebook parameters
        self.codebook_vocab = self.quantizer.codebook_vocab
        self.codebook_level = self.quantizer.codebook_level
        
        # Initialize model parameters
        self.add_doc2query_task = False
        self.iter = 0
        self.alpha = getattr(getattr(config, 'hyperparameter_config', {}), 'alpha', 2)
        self.seperator = ''
        self.tokenizer = tokenizer
        self.trie_index = None
        self.trie_type = getattr(getattr(config, 'model', {}), 'trie_type', 'trie_cpp')
        self._compiled_id_gen = False

        # Ensure unique code 
        if self.quantizer.unique_code:
            self.codebook_level = self.codebook_level + 1

        # Initialize tokenizer if needed
        if new_tokenizer:
            self._initialize_tokenizer(tokenizer)

        # Initialize codebook tokens
        self._initialize_codebook_tokens()

        # Initialize codebook embeddings if needed
        if init_rq_codebook:
            self._initialize_codebook_embeddings(t5_config)

    def _initialize_tokenizer(self, tokenizer):
        """Initialize a new tokenizer with special tokens."""
        special_tokens = {'pad_token': '<pad>', 'eos_token': '</s>', 'unk_token': '<unk>'}
        special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens.values()]
        new_vocab = {tokenizer.convert_ids_to_tokens(idx): idx for idx in special_token_ids}
        start_idx = max(new_vocab.values()) + 1
        tokenizer_model = WordLevel(vocab=new_vocab, unk_token='<unk>')
        tokenizer = Tokenizer(tokenizer_model)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **special_tokens)
        self.tokenizer = tokenizer
        self.id_generator.resize_token_embeddings(len(tokenizer))

    def _initialize_codebook_tokens(self):
        """Initialize codebook tokens for different levels."""
        self.level_indicators = list(string.ascii_lowercase[:self.codebook_level])
        code_tokens = []
        for l, level in enumerate(self.level_indicators):
            if self.modality_index and l == 0:
                for i in range(3):
                    code_tokens.append(f'<{level}{i}>')
                continue
            for i in range(self.codebook_vocab):
                code_tokens.append(f'<{level}{i}>')
        num_new_tokens = self.tokenizer.add_tokens(code_tokens)
        self.id_generator.resize_token_embeddings(len(self.tokenizer))

    def _initialize_codebook_embeddings(self, t5_config):
        """Initialize codebook embeddings from residual quantization."""
        # Get the new token IDs
        code_tokens = [f'<{l}{i}>' for l in self.level_indicators for i in range(self.codebook_vocab)]
        new_token_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)

        # Map codebook vectors to embedding dimension
        embedding_dim = t5_config.d_model
        linear_layer = nn.Linear(768, embedding_dim, bias=False)

        if self.modality_index:
            # Handle modality-specific codebook initialization
            first_layer_codebook = self.quantizer.residual_rq.layers[0]._codebook.embed[0, :3, :]
            mapped_first_layer_embeddings = linear_layer(F.normalize(first_layer_codebook))

            other_layers_codebooks = [layer._codebook.embed for layer in self.quantizer.residual_rq.layers[1:]]
            other_layers_codebooks = torch.stack(other_layers_codebooks, dim=0)
            other_layers_codebooks = rearrange(other_layers_codebooks, 'q 1 c d -> q c d')
            other_layers_codebook = other_layers_codebooks.reshape(-1, other_layers_codebooks.shape[-1])
            mapped_other_layers_embeddings = linear_layer(F.normalize(other_layers_codebook))

            mapped_embeddings = torch.cat([mapped_first_layer_embeddings, mapped_other_layers_embeddings], dim=0)
        else:
            # Handle standard codebook initialization
            codebook_vectors = self.quantizer.residual_rq.codebooks.reshape(-1, 768)
            mapped_embeddings = linear_layer(F.normalize(codebook_vectors))

        # Initialize model embeddings
        self._initialize_model_embeddings(new_token_ids, mapped_embeddings)

    def _initialize_model_embeddings(self, new_token_ids, mapped_embeddings):
        """Initialize model embeddings with mapped codebook vectors."""
        # Initialize language model head
        with torch.no_grad():
            lm_head = self.id_generator.lm_head
            for idx, token_id in enumerate(new_token_ids):
                lm_head.weight[token_id] = mapped_embeddings[idx] * 100
        for param in lm_head.parameters():
            param.requires_grad = True

        # Initialize shared embeddings
        with torch.no_grad():
            shared = self.id_generator.shared
            for idx, token_id in enumerate(new_token_ids):
                shared.weight[token_id] = mapped_embeddings[idx] * 100
        for param in shared.parameters():
            param.requires_grad = True
        self.id_generator._tie_weights()

    def transform_row(self, row, separator=''):
        """Transform a row of codes into token format."""
        transformed_row = []
        for l, level_indicator in enumerate(self.level_indicators):
            new_value = row[l]
            transformed_row.append(f"<{level_indicator}{new_value}>")
        return separator.join(transformed_row)

    def detransform_row(self, row):
        """Convert token format back to code row."""
        splited_row = re.findall(r'<.*?>', row)
        detransformed_row = []
        for token in splited_row:
            match = re.match(r'<([a-z])(\d+)>', token)
            if match:
                level_indicator, value = match.groups()
                detransformed_row.append(int(value))
        return detransformed_row

    def replace_punctuation(self, text, pos_index):
        """Replace punctuation with position information."""
        def replace_func(match):
            return f" for label {pos_index}{match.group()}"
        return re.sub(r'([.?])', replace_func, text)

    def compute_single_batch(self, batch, gpu_id=None):
        """Compute loss and metrics for a single batch."""
        query, pool, instruct, h_qid = batch

        # Prepare input tensors
        h_qid = h_qid.view(-1)
        instruct = instruct.view(-1, instruct.size(-1)).to(gpu_id, non_blocking=True)

        # Extract masks
        q_img_mask = query['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        q_txt_mask = query['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_img_mask = pool['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_txt_mask = pool['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)

        # Extract embeddings
        q_img_emb = query['img_emb'].view(-1, query['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        q_txt_emb = query['txt_emb'].view(-1, query['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_img_emb = pool['img_emb'].view(-1, pool['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_txt_emb = pool['txt_emb'].view(-1, pool['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)
        
        device = q_img_emb.device
        bs = len(q_img_emb)
        
        # Process instructions
        instruct_str = self.tokenizer.batch_decode(instruct, skip_special_tokens=True)

        # Get quantized representations
        q_output = self.quantizer.inference(q_img_emb, q_txt_emb, q_img_mask, q_txt_mask)
        q_emb, q_code_list = q_output['encode'], q_output['code'].tolist()
        p_output = self.quantizer.inference(p_img_emb, p_txt_emb, p_img_mask, p_txt_mask)
        p_emb, p_code_list = p_output['encode'], p_output['code'].tolist()
        q_emb, p_emb = F.normalize(q_emb), F.normalize(p_emb)

        # Prepare target codes
        p_code_str_list = [self.transform_row(row) for row in p_code_list]
        
        # Apply augmentation
        s = torch.distributions.Beta(self.alpha, self.alpha).sample((bs, q_emb.size(1))).to(device)
        aug_emb = torch.sqrt(s) * q_emb + torch.sqrt(1-s) * p_emb
        aug_proj = self.embed_projector(F.normalize(aug_emb)).to(q_emb.dtype)
        inputs_embeds = aug_proj.reshape(bs, self.num_prefix, -1)

        # Prepare labels
        label_id = p_code_str_list
        labels = self.tokenizer(label_id, padding=True, truncation=True).input_ids
        labels = torch.LongTensor(labels).to(device)
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        
        # Forward pass
        model_outputs = self.id_generator(inputs_embeds=inputs_embeds, 
                                        labels=labels, 
                                        output_hidden_states=True)
        
        # Compute loss and metrics
        logits = model_outputs.logits
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        pred_idx = logits.argmax(-1)
        R_at_1 = (pred_idx[:bs] == labels[:bs]).all(1).sum() / bs
        Level1_acc = (pred_idx[:bs,1:2] == labels[:bs,1:2]).all(1).sum() / bs
        Level12_acc = (pred_idx[:bs,1:3] == labels[:bs,1:3]).all(1).sum() / bs
        Level123_acc = (pred_idx[:bs,1:4] == labels[:bs,1:4]).all(1).sum() / bs

        # Prepare outputs
        outputs = {
            'loss': loss,
            'R_at_1': R_at_1,
            'Level1_acc': Level1_acc,
            'Level12_acc': Level12_acc,
            'Level123_acc': Level123_acc
        }
        
        # Log examples periodically
        if self.iter % 200 == 0:
            example = self.tokenizer.decode(pred_idx[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            example_labels = self.tokenizer.decode(labels[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print('Retrieval Task: ' + 'Pred: ' + str(example) + ' Ans: ' + str(example_labels))
        
        self.iter += 1
        return outputs

    def get_img_preprocess_fn(self):
        """Get image preprocessing function from CLIP model."""
        return self.clip_model.get_img_preprocess_fn()
    
    def get_clip_tokenizer(self):
        """Get CLIP tokenizer."""
        return self.clip_model.get_tokenizer()
    
    def get_seq2seq_tokenizer(self):
        """Get sequence-to-sequence tokenizer."""
        return self.tokenizer

    def generative_index(self, cand_codes):
        """
        Construct Trie Index for efficient retrieval.
        
        Args:
            cand_codes: Candidate codes to index
            
        Returns:
            Trie index for the candidate codes
        """
        model_config = self.config.model
        ckpt_config = model_config.ckpt_config
        trie_path = os.path.join(self.config.genir_dir, ckpt_config.ckpt_dir)
        cand_codes_str = [self.transform_row(row) for row in cand_codes]
        cand_codes_ids = np.array(self.tokenizer(cand_codes_str, add_special_tokens=False).input_ids).tolist()
        
        if self.trie_type == 'marisa':
            trie = MarisaTrie(cand_codes_ids)
        elif self.trie_type == 'triecpp':
            trie = Trie_Cpp.Trie(cand_codes_ids)
        else:
            trie = Trie(cand_codes_ids)
        return trie

    def distribute_trie(self, cand_codes, trie_save_path):
        """
        Distribute Trie index across processes.
        
        Args:
            cand_codes: Candidate codes to index
            trie_save_path: Path to save the Trie index
        """
        if trie_save_path.endswith("trie.pkl"):
            suffix_map = {
                "trie": "trie.pkl",
                "triecpp": "triecpp.pkl",
                "triemarisa": "triemarisa.pkl",
            }
            base_path = trie_save_path[:-len("trie.pkl")]
            trie_save_path = base_path + suffix_map.get(self.trie_type, "trie.pkl")

        if self.trie_type == 'triecpp':
            if dist.get_rank() == 0 and not os.path.exists(trie_save_path):
                trie_index = self.generative_index(cand_codes)
                with open(trie_save_path, 'wb') as f:
                    pickle.dump(trie_index.to_dict(), f)
                    f.flush()
                    os.fsync(f.fileno())
                del trie_index
                print(f"Log: Save candidate pool Trie from {trie_save_path}.")

            dist.barrier()

            if hasattr(self, 'trie_index'):
                del self.trie_index
                torch.cuda.empty_cache()

            with open(trie_save_path, 'rb') as f:
                trie_dict = pickle.load(f)
            self.trie_index_dict = trie_dict
            self.trie_index = Trie_Cpp.Trie.from_dict(trie_dict)
            del trie_dict

        else:
            if dist.get_rank() == 0 and not os.path.exists(trie_save_path):
                trie_index = self.generative_index(cand_codes)
                with open(trie_save_path, 'wb') as f:
                    pickle.dump(trie_index, f)
                    f.flush()
                    os.fsync(f.fileno())
                del trie_index
                print(f"Log: Save candidate pool Trie from {trie_save_path}.")
            dist.barrier()

            if hasattr(self, 'trie_index'):
                del self.trie_index
                torch.cuda.empty_cache()

            with open(trie_save_path, 'rb') as f:
                self.trie_index = pickle.load(f)

        print(f"Log: Loaded candidate pool Trie from {trie_save_path}.")

    def constrained_beam_search(self, inputs_embeds, attention_mask=None, num_beams=10, cand_codes=None):
        """
        Perform constrained beam search for generation.
        
        Args:
            inputs_embeds: Input embeddings
            attention_mask: Attention mask
            num_beams: Number of beams for search
            cand_codes: Candidate codes
            
        Returns:
            Generated sequences
        """
        batch_size = inputs_embeds.size(0)
        device = inputs_embeds.device

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            prefix = input_ids.tolist()
            if prefix[0] == self.tokenizer.pad_token_id:
                prefix = prefix[1:]
            valid_tokens = self.trie_index.get(prefix)
            if valid_tokens is None:
                return []
            else:
                return valid_tokens

        with torch.no_grad():
            generated = self.id_generator.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                num_beams=num_beams, 
                num_return_sequences=num_beams,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                use_cache=True, 
                max_new_tokens=self.codebook_level,
                early_stopping=True,
            )
        return generated

    @torch.no_grad()
    def inference(self, img_emb, txt_emb, img_mask, txt_mask, inst_ids, num_beams=10, cand_codes=None):
        """
        Perform inference on input embeddings.
        
        Args:
            img_emb: Image embeddings
            txt_emb: Text embeddings
            img_mask: Image mask
            txt_mask: Text mask
            inst_ids: Instruction IDs
            num_beams: Number of beams for search
            cand_codes: Candidate codes
            
        Returns:
            Generated codes and normalized embeddings
        """
        device = img_emb.device
        bs = len(img_emb)

        if not self._compiled_id_gen:
            self.id_generator = torch.compile(self.id_generator)
            self._compiled_id_gen = True       

        # Get quantized representations
        q_output = self.quantizer.inference(img_emb, txt_emb, img_mask, txt_mask)
        emb, q_code_list = q_output['encode'], q_output['code'].tolist()
        emb = F.normalize(emb)
        instruct_str = self.tokenizer.batch_decode(inst_ids, skip_special_tokens=True)
        q_code_str_list = [self.transform_row(row) for row in q_code_list]

        # Generate outputs
        inputs_embeds = self.embed_projector(emb).reshape(len(emb), self.num_prefix, -1)
        outputs = self.constrained_beam_search(inputs_embeds, num_beams=num_beams, cand_codes=cand_codes)
        
        # Process outputs
        outputs_id = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs_id = [self.detransform_row(row) for row in outputs_id]
        outputs_id_tensor = torch.LongTensor(outputs_id).to(device)
        outputs_id_tensor = outputs_id_tensor.reshape(bs, num_beams, -1).detach()
        
        return outputs_id_tensor, F.normalize(img_emb * img_mask + txt_emb * txt_mask)

    def encode_mbeir_batch(self, batch, num_beams=10, cand_codes=None, trie_save_path=None, init_dataset=True):
        """
        Encode a batch of MBEIR data.
        
        Args:
            batch: Input batch
            num_beams: Number of beams for search
            cand_codes: Candidate codes
            trie_save_path: Path to save Trie index
            init_dataset: Whether to initialize dataset
            
        Returns:
            Generated codes, embeddings, and IDs
        """
        # Ensure all processes have the trie index
        if init_dataset:
            self.distribute_trie(cand_codes, trie_save_path)

        # Get hashed id_list
        id_list = batch.get("did_list") or batch.get("qid_list")
        assert id_list is not None, "id_list must be provided."
        assert isinstance(id_list[0], int), "id_list must be hashed to int."

        # Encode multimodal inputs
        img_emb, txt_emb = self.clip_model.encode_multimodal_input(
            batch["image_batched"],
            batch["txt_batched"],
        )
        img_mask = batch["image_mask_batched"].unsqueeze(-1)
        txt_mask = batch["txt_mask_batched"].unsqueeze(-1)
        assert img_emb.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."

        # Generate outputs
        output, embeddings = self.inference(
            img_emb, 
            txt_emb, 
            img_mask,
            txt_mask,
            batch["inst_ids"],
            num_beams=num_beams,
            cand_codes=cand_codes
        )
        return output, embeddings, torch.LongTensor(id_list)

    def forward(self, 
                input=None, 
                encode_mbeir_batch=False, 
                num_beams=10, 
                cand_codes=None, 
                init_dataset=True,
                trie_save_path=None,
                gpu_id=None):
        """
        Forward pass of the model.
        
        Args:
            input: Input data
            encode_mbeir_batch: Whether to encode MBEIR batch
            num_beams: Number of beams for search
            cand_codes: Candidate codes
            init_dataset: Whether to initialize dataset
            trie_save_path: Path to save Trie index
            gpu_id: GPU ID to use
            
        Returns:
            Model outputs
        """
        if encode_mbeir_batch:
            return self.encode_mbeir_batch(input, cand_codes=cand_codes, num_beams=num_beams, init_dataset=init_dataset, trie_save_path=trie_save_path)
        else:
            return self.compute_single_batch(input, gpu_id) 