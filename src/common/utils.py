"""
Utility functions for retrieval experiments on MBEIR.
"""

# Standard Library imports
import os
import random

# Third-party imports
import numpy as np
import torch


def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if int(relevance_score) > 0:  # Assuming only positive relevance scores indicate relevant documents
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Loaded {len(qrel)} queries from {filename}")
    print(f"Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}")
    return qrel, qid_to_taskid


# TODO: Write a hashed id to unhased id converter.
def load_runfile(filename, load_task_id=False):
    run_results = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            if qid not in run_results:
                run_results[qid] = []
            if load_task_id:
                qid, _, did, rank, score, run_id, task_id = parts
                run_results[qid].append(
                    {
                        "did": did,
                        "rank": int(rank),
                        "score": float(score),
                        "task_id": task_id,
                    }
                )
            else:
                parts = parts[:6]
                qid, _, did, rank, score, run_id = parts
                run_results[qid].append(
                    {
                        "did": did,
                        "rank": int(rank),
                        "score": float(score),
                    }
                )
    print(f"Loaded results for {len(run_results)} queries from {filename}")
    return run_results


def build_model_from_config(config):
    model_name = config.model.name

    if model_name == "CLIPScoreFusion":
        from models.uniir_clip.clip_scorefusion.clip_sf import CLIPScoreFusion

        # initialize CLIPScoreFusion model
        model_config = config.model
        uniir_dir = config.uniir_dir
        download_root = os.path.join(uniir_dir, model_config.pretrained_clip_model_dir)
        print(f"Downloading CLIP model to {download_root}...")
        model = CLIPScoreFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=download_root,
        )
        model.float()
        # The origial CLIP was in fp16 so we need to convert it to fp32

        # Load model from checkpoint
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading CLIPScoreFusion checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["model"], map_location=torch.device('cpu'))

    elif model_name == "CLIPFeatureFusion":
        from models.uniir_clip.clip_featurefusion.clip_ff import CLIPFeatureFusion

        # initialize CLIPFeatureFusion model
        model_config = config.model
        uniir_dir = config.uniir_dir
        download_root = os.path.join(uniir_dir, model_config.pretrained_clip_model_dir)
        print(f"Downloading CLIP model to {download_root}...")
        model = CLIPFeatureFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=download_root,
        )
        model.float()
        # The origial CLIP was in fp16 so we need to convert it to fp32

        # Load model from checkpoint
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading CLIPFeatureFusion checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["model"], map_location=torch.device('cpu'))

    elif model_name == "BLIPScoreFusion":
        from models.uniir_blip.blip_scorefusion.blip_sf import BLIPScoreFusion

        model_config = config.model
        model = BLIPScoreFusion(
            med_config=os.path.join("../models/uniir_blip", "backbone/configs/med_config.json"),
            image_size=model_config.image_size,
            vit=model_config.vit,
            vit_grad_ckpt=model_config.vit_grad_ckpt,
            vit_ckpt_layer=model_config.vit_ckpt_layer,
            embed_dim=model_config.embed_dim,
            queue_size=model_config.queue_size,
            config=model_config,
        )
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading BLIPScoreFusion checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["model"], map_location=torch.device('cpu'))

    elif model_name == "BLIPFeatureFusion":
        from models.uniir_blip.blip_featurefusion.blip_ff import BLIPFeatureFusion

        model_config = config.model
        model = BLIPFeatureFusion(
            med_config=os.path.join("../models/uniir_blip", "backbone/configs/med_config.json"),
            image_size=model_config.image_size,
            vit=model_config.vit,
            vit_grad_ckpt=model_config.vit_grad_ckpt,
            vit_ckpt_layer=model_config.vit_ckpt_layer,
            embed_dim=model_config.embed_dim,
            queue_size=model_config.queue_size,
            config=model_config,
        )
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading BLIPFeatureFusion checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["model"], map_location=torch.device('cpu'))

    elif model_name == "RQTokenizer":
        from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion
        from models.residual_quantization.residual_quantization import RQ

        model_config = config.model
        pretrained_clip_model_dir = os.path.join(config.genir_dir, model_config.pretrained_clip_model_dir)
        clip_model = CLIPNoFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=pretrained_clip_model_dir,
            config=config,
        )
        clip_model.float()  # The origial CLIP was in fp16 so we need to convert it to fp32
        clip_model.eval()

        pretrained_config = model_config.pretrained_config
        pretrained_path = os.path.join(config.genir_dir, pretrained_config.pretrained_dir, pretrained_config.pretrained_name)
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        clip_model.load_state_dict(checkpoint["model"])

        model_config = config.model
        model = RQ(config=config, clip_model=clip_model)
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.genir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading Residual Quantization checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["model"], strict=False)

    elif model_name == "BartGenerativeRetriever":
        from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion
        from transformers import BartTokenizer, BartTokenizerFast
        from models.generative_retriever.retriever import BartForGenerativeRetrieval

        model_config = config.model
        pretrained_clip_model_dir = os.path.join(config.genir_dir, model_config.pretrained_clip_model_dir)
        clip_model = CLIPNoFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=pretrained_clip_model_dir,
            config=config,
        )
        clip_model.float()  # The origial CLIP was in fp16 so we need to convert it to fp32
        clip_model.eval()

        pretrained_config = model_config.pretrained_config
        pretrained_path = os.path.join(config.genir_dir, pretrained_config.pretrained_dir, pretrained_config.pretrained_name)
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        clip_model.load_state_dict(checkpoint["model"])
        
        seq2seq_tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base", model_max_length=60)
        model = BartForGenerativeRetrieval(config=config, tokenizer=seq2seq_tokenizer, clip_model=clip_model)

        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.genir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading GenerativeRetriever checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["model"], strict=False)

    elif model_name == "T5GenerativeRetriever":
        from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion
        from transformers import T5Tokenizer, T5TokenizerFast
        from models.generative_retriever.retriever import T5ForGenerativeRetrieval

        model_config = config.model
        pretrained_clip_model_dir = os.path.join(config.genir_dir, model_config.pretrained_clip_model_dir)
        clip_model = CLIPNoFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=pretrained_clip_model_dir,
            config=config,
        )
        clip_model.float()  # The origial CLIP was in fp16 so we need to convert it to fp32
        clip_model.eval()

        pretrained_config = model_config.pretrained_config
        pretrained_path = os.path.join(config.genir_dir, pretrained_config.pretrained_dir, pretrained_config.pretrained_name)
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        clip_model.load_state_dict(checkpoint["model"])
        
        # seq2seq_tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-base", model_max_length=60)
        seq2seq_tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small", model_max_length=42)
        model = T5ForGenerativeRetrieval(config=config, tokenizer=seq2seq_tokenizer, clip_model=clip_model)

        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.genir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading GenerativeRetriever checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["model"], strict=False)

    elif model_name == "OPTGenerativeRetriever":
        from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion
        from transformers import GPT2TokenizerFast
        from models.generative_retriever.retriever import OptForGenerativeRetrieval

        model_config = config.model
        pretrained_clip_model_dir = os.path.join(config.genir_dir, model_config.pretrained_clip_model_dir)
        clip_model = CLIPNoFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=pretrained_clip_model_dir,
            config=config,
        )
        clip_model.float()  # The origial CLIP was in fp16 so we need to convert it to fp32
        clip_model.eval()

        pretrained_config = model_config.pretrained_config
        pretrained_path = os.path.join(config.genir_dir, pretrained_config.pretrained_dir, pretrained_config.pretrained_name)
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        clip_model.load_state_dict(checkpoint["model"])
        
        seq2seq_tokenizer = GPT2TokenizerFast.from_pretrained("facebook/opt-125m", model_max_length=60)
        model = OptForGenerativeRetrieval(config=config, tokenizer=seq2seq_tokenizer, clip_model=clip_model)

        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.genir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading GenerativeRetriever checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["model"], strict=False)

    else:
        raise NotImplementedError(f"Model {model_name} is not implemented.")
        # Notes: Add other models here
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
