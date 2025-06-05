"""
Training Code for CLIP-SF
"""

# Standard library
import argparse
import logging
import os
import random
import clip 

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
from dotenv import load_dotenv
import wandb
from tqdm import tqdm
import common.dist_utils as dist_utils
from common.dist_utils import ContiguousDistributedSampler
from torch.cuda.amp import autocast
import gc

# Local modules or packages
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
    build_distributed_sampler_list,
    build_dataloader_list,
)

from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolDataset,
    MBEIRCandidatePoolCollator,
    Mode,
)

from data.preprocessing.utils import (
    format_string,
    hash_did,
    hash_qid,
    unhash_did,
    unhash_qid,
    get_mbeir_task_id,
)

from models.uniir_clip.engine import train_one_epoch, eval_engine
from models.uniir_clip import utils
from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion


# Set up logger
logger = logging.getLogger()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_embeds_for_config(model, img_preprocess_fn, tokenizer, config):
    """This script generates embeddings for the queries and candidates(doc)"""
    genir_dir = config.genir_dir
    mbeir_data_dir = config.mbeir_data_dir
    embed_config = config.embed_config
    embed_dir_name = embed_config.embed_dir_name
    expt_dir_name = config.experiment.path_suffix
    save_embed_path = os.path.join(config.genir_dir, config.model.emb_save_path)

    # Config for dataset
    data_config = config.data_config
    query_instruct_path = data_config.query_instruct_path
    cand_pool_dir = data_config.cand_pool_dir_name
    image_size = tuple(map(int, data_config.image_size.split(",")))

    splits = []
    # Load the dataset splits to embed
    dataset_types = ["train", "val", "test"]
    all_cand_pool_name_list = [
        "visualnews_task0",
        "mscoco_task0_test",
        "fashion200k_task0",
        "webqa_task1",
        "edis_task2",
        "webqa_task2",
        "visualnews_task3",
        "mscoco_task3_test",
        "fashion200k_task3",
        "nights_task4",
        "oven_task6",
        "infoseek_task6",
        "fashioniq_task7",
        "cirr_task7",
        "oven_task8",
        "infoseek_task8"
    ]
    for split_name in dataset_types:
        split_dir_name = getattr(data_config, f"{split_name}_dir_name")
        embed_dataset_config = getattr(embed_config, f"{split_name}_datasets_config", None)
        if embed_dataset_config and embed_dataset_config.enable_embed:
            dataset_name_list = getattr(embed_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(embed_dataset_config, "correspond_cand_pools_name", None)
            splits.append((split_name, split_dir_name, dataset_name_list, cand_pool_name_list))
            assert len(dataset_name_list) == len(cand_pool_name_list), "Mismatch between datasets and candidate pools."

    # Load the candidate pool to embed
    embed_cand_pool_config = embed_config.cand_pools_config
    if embed_cand_pool_config and embed_cand_pool_config.enable_embed:
        split_name = "cand_pool"
        split_dir_name = data_config.cand_pool_dir_name
        cand_pool_name_list = embed_cand_pool_config.cand_pools_name_to_embed
        splits.append(
            (
                split_name,
                split_dir_name,
                [None] * len(cand_pool_name_list),
                cand_pool_name_list,
            )
        )

    # Pretty Print dataset and candidate pool to embed
    if dist_utils.is_main_process():
        print("-" * 30)
        for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
            if split_name == "cand_pool":
                print(f"Split: {split_name}, Split dir: {split_dir}, Candidate pools to embed: {cand_pool_name_list}")
            else:
                print(f"Split: {split_name}, Split dir: {split_dir}, Datasets to embed: {dataset_name_list}")
            print("-" * 30)

    # Generate embeddings
    for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
        for dataset_name, cand_pool_name in zip(dataset_name_list, cand_pool_name_list):
            dataset_split_dict = {
                'img': [],
                'text': [],
                'img_mask': [],
                'text_mask': [],
                'id_to_index': {}
            }
            if split_name == "cand_pool":
                cand_pool_name = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{cand_pool_name}_{split_name}.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"\nEmbedder Log: Generating embeddings for {cand_pool_data_path}...")
                    print_config = True
                # TODO: refactor this and use build dataset from Config.
                dataset = MBEIRCandidatePoolDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    cand_pool_data_path=cand_pool_data_path,
                    img_preprocess_fn=img_preprocess_fn,
                    print_config=print_config,
                )
                collator = MBEIRCandidatePoolCollator(
                    tokenizer=tokenizer,
                    image_size=image_size,
                )
            else:  # "train" or "val" or "test"
                # Construct query data path
                dataset_name = dataset_name.lower()
                query_data_name = f"mbeir_{dataset_name}_{split_name}.jsonl"
                query_data_path = os.path.join(split_dir, query_data_name)

                # Construct the candidate pool path
                cand_pool_name = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{cand_pool_name}_cand_pool.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"\nEmbedder Log: Generating embeddings for {query_data_path} with {cand_pool_data_path}...")
                    print_config = True
                mode = Mode.EVAL
                dataset = MBEIRMainDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    query_data_path=query_data_path,
                    cand_pool_path=cand_pool_data_path,
                    query_instruct_path=query_instruct_path,
                    img_preprocess_fn=img_preprocess_fn,
                    mode=mode,
                    enable_query_instruct=data_config.enable_query_instruct,
                    shuffle_cand=data_config.shuffle_cand,
                    print_config=print_config,
                )
                collator = MBEIRMainCollator(
                    tokenizer=tokenizer,
                    image_size=image_size,
                    mode=mode,
                )

            # Config for data loader
            batch_size = config.dataloader_config.batch_size
            num_workers = config.dataloader_config.num_workers

            # Set up distributed data parallel
            num_tasks = dist_utils.get_world_size()
            global_rank = dist_utils.get_rank()
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
            )  # Note: assume the dataset is in sorted order.
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=False,
                sampler=sampler,
                shuffle=False,  # Since we have distributed sampler, we don't need to shuffle the data here.
                collate_fn=collator,
                drop_last=False,
            )
            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.
            if dist_utils.is_main_process():
                if split_name == "cand_pool":
                    print(f"Embedder Log: Data loader for {cand_pool_data_path} is set up.")
                    print(f"Embedder Log: Generating embeddings for {cand_pool_data_path}...")
                else:
                    print(
                        f"Embedder Log: Data loader for {query_data_path} with candidate pool {cand_pool_data_path} is set up."
                    )
                    print(f"Embedder Log: Generating embeddings for {query_data_path} ...")
                print(f"Inference with half precision: {config.embed_config.use_fp16}")

            index = 0
            for i, batch in enumerate(tqdm(data_loader)):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(config.dist_config.gpu_id, non_blocking=True)

                # Extract ids, txt, image and masks
                did_list = batch.get("did_list")
                txt_batched = batch["txt_batched"]
                image_batched = batch["image_batched"]
                txt_mask_batched = batch["txt_mask_batched"]
                image_mask_batched = batch["image_mask_batched"]

                # Encode inputs using the model
                with autocast(enabled=config.embed_config.use_fp16):
                    img_emb, txt_emb = model.module.encode_multimodal_input(image_batched, txt_batched)

                # Gather outputs if distributed training is enabled
                dist.barrier()
                if utils.get_world_size() > 1:
                    did_list = torch.cat(utils.GatherLayer.apply(torch.LongTensor(did_list).to(img_emb.device)), dim=0)
                    txt_emb = torch.cat(utils.GatherLayer.apply(txt_emb), dim=0)
                    img_emb = torch.cat(utils.GatherLayer.apply(img_emb), dim=0)
                    txt_mask_batched = torch.cat(utils.GatherLayer.apply(txt_mask_batched), dim=0)
                    image_mask_batched = torch.cat(utils.GatherLayer.apply(image_mask_batched), dim=0)

                dist.barrier()
                # Save to dictionary on the main process
                if utils.is_main_process():
                    for j, did in enumerate(did_list):
                        if utils.get_world_size() > 1:
                            did = did.item()
                        if did not in dataset_split_dict['id_to_index']:
                            dataset_split_dict['id_to_index'][did] = index
                            dataset_split_dict['img'].append(img_emb[j].detach().cpu())
                            dataset_split_dict['text'].append(txt_emb[j].detach().cpu())
                            dataset_split_dict['img_mask'].append(image_mask_batched[j].detach().cpu())
                            dataset_split_dict['text_mask'].append(txt_mask_batched[j].detach().cpu())
                            index += 1

            # Save the split dictionary to a pt file
            dist.barrier()
            if utils.is_main_process():
                # Convert lists to tensors for storage efficiency
                dataset_split_dict['img'] = torch.stack(dataset_split_dict['img'], dim=0)
                dataset_split_dict['text'] = torch.stack(dataset_split_dict['text'], dim=0)
                dataset_split_dict['img_mask'] = torch.stack(dataset_split_dict['img_mask'], dim=0)
                dataset_split_dict['text_mask'] = torch.stack(dataset_split_dict['text_mask'], dim=0)

                # Create directory if not exists
                os.makedirs(save_embed_path, exist_ok=True)
                
                # Save the dictionary using pt
                file_name = f"{split_name}_{cand_pool_name}_IT_dict.pt"
                file_path = os.path.join(save_embed_path, file_name)
                torch.save(dataset_split_dict, file_path)
                print(f"Embedder Log: Saved embeddings and masks to {file_path}")

            # Barrier to ensure rank 0 saves before continuing
            if utils.get_world_size() > 1:
                torch.distributed.barrier()

            # Free memory after each dataset
            del dataset_split_dict
            del dataset
            del collator
            del data_loader
            gc.collect()
            torch.cuda.empty_cache()

        # Union pool embeddings
        if split_name == "cand_pool" and embed_cand_pool_config.embed_union_pool:
            # To efficiently generate embeddings for the union(global) pool,
            # We concat previously saved embeddings and ids from single(local) pool
            # Instead of embed the union pool directly.
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"\nEmbedder Log: Generating embeddings for union pool...")

                # Increase number of threads for rank 0
                world_size = utils.get_world_size()
                total_cores = os.cpu_count()
                initial_threads_per_process = total_cores // world_size
                torch.set_num_threads(total_cores)

                all_img = []
                all_text = []
                all_img_mask = []
                all_text_mask = []
                all_ids = []
                for cand_pool_name in all_cand_pool_name_list:
                    cand_pool_name = cand_pool_name.lower()
                    embed_data_name = f"{split_name}_{cand_pool_name}_IT_dict.pt"
                    file_path = os.path.join(save_embed_path, embed_data_name)
                    dataset_split_dict = torch.load(file_path)
                    all_img.append(dataset_split_dict['img'])
                    all_text.append(dataset_split_dict['text'])
                    all_img_mask.append(dataset_split_dict['img_mask'])
                    all_text_mask.append(dataset_split_dict['text_mask'])
                    all_ids.extend(list(dataset_split_dict['id_to_index'].keys()))
                    print(f"Embedder Log: Concatenating embeddings from {file_path}.")

                all_img = torch.cat(all_img, dim=0)
                all_text = torch.cat(all_text, dim=0)
                all_img_mask = torch.cat(all_img_mask, dim=0)
                all_text_mask = torch.cat(all_text_mask, dim=0)
                all_ids = torch.LongTensor(all_ids)
                assert len(all_img) == len(all_ids), "Mismatch between embeddings and IDs length."
                print(f"Embedder Log: all_img length: {len(all_img)}, all_text length: {len(all_text)}, all_ids length: {len(all_ids)}.")

                # Save the embeddings and ids to .pt
                union_data_name = f"mbeir_union_{split_name}_IT_dict.pt"
                union_path = os.path.join(save_embed_path, union_data_name)
                union_dict = {
                        'img': all_img,
                        'text': all_text,
                        'img_mask': all_img_mask,
                        'text_mask': all_text_mask,
                        'id_to_index': {i: idx for idx, i in enumerate(all_ids)}
                    }
                torch.save(union_dict, union_path)
                print(f"Embedder Log: Saved union embeddings and ids to {union_path}.")

                # Delete the embeddings and IDs to free up memory
                del all_img
                del all_text
                del all_img_mask
                del all_text_mask
                del all_ids

                # Explicitly call the garbage collector
                gc.collect()

                # Reset number of threads to initial value after conversion
                torch.set_num_threads(initial_threads_per_process)

            if utils.get_world_size() > 1:
                torch.distributed.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.


def main(config):
    is_distributed_mode = config.dist_config.distributed_mode

    # Set up seed for reproducibility
    seed = config.seed + utils.get_rank()
    set_seed(seed)

    gpu_id = config.dist_config.gpu_id
    cudnn.benchmark = True

    # Initialize and load model
    print("Creating CLIP model...")
    model_config = config.model
    pretrained_clip_model_dir = os.path.join(config.genir_dir, model_config.pretrained_clip_model_dir)
    logger.info(f"Downloading CLIP model to {pretrained_clip_model_dir}...")
    
    model = CLIPNoFusion(
        model_name=model_config.clip_vision_model_name,
        download_root=pretrained_clip_model_dir,
        config=config,
    )
    model.float()  # The origial CLIP was in fp16 so we need to convert it to fp32

    ckpt_config = model_config.ckpt_config
    if ckpt_config.using_pretrained:
        checkpoint_path = os.path.join(config.genir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading CLIPScoreFusion checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model"])

    # Move model to GPUs
    model.eval()
    model = model.to(config.dist_config.gpu_id)

    model_without_ddp = model
    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id])
        model_without_ddp = model.module

    img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()
    tokenizer = model_without_ddp.get_tokenizer()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    # Generate embeddings for the configuration
    generate_embeds_for_config(model, img_preprocess_fn, tokenizer, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--genir_dir",
        type=str,
        default="/data/GENIUS",
        help="Path to GENIUS directory to save checkpoints, embeddings, etc.",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/GENIUS/mbeir_data",
        help="Path to mbeir dataset directory",
    )
    args = parser.parse_args()
    print(f"Loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.genir_dir = args.genir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    # Initialize distributed training
    args.dist_url = config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    main(config)

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
