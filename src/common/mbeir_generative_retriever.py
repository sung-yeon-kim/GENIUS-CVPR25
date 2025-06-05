import os
import argparse
from omegaconf import OmegaConf
import tqdm
import gc
import random
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.cuda.amp import autocast
import transformers
from datetime import datetime
import torch.nn.functional as F

import dist_utils
from dist_utils import ContiguousDistributedSampler
from utils import build_model_from_config
from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolDataset,
    MBEIRCandidatePoolCollator,
    MBEIRDictCandDataset,
    Mode,
)
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


from data.preprocessing.utils import (
    load_jsonl_as_list,
    save_list_as_jsonl,
    count_entries_in_file,
    load_mbeir_format_pool_file_as_dict,
    print_mbeir_format_dataset_stats,
    unhash_did,
    unhash_qid,
    get_mbeir_task_name,
)
import sys
import csv

sys.setrecursionlimit(10000)

EXCEPTIONAL_CAND_POOLS = ['mscoco_task0', 'mscoco_task3', 'flickr30k_task0', 'flickr30k_task3']\

@torch.no_grad()
def generate_codes_for_dataset(
    model, 
    data_loader, 
    device, 
    use_fp16=True, 
    is_quantizer=True, 
    num_beams=10, 
    cand_codes=None, 
    is_extracted=False,
    use_embedding=True,
    trie_save_path = None
):
    codes_tensor = []
    id_tensor = []
    if use_embedding:
        encode_tensor = []

    total_cores = os.cpu_count()
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        initial_threads_per_process = total_cores // world_size
        torch.set_num_threads(initial_threads_per_process)
        data_loader = tqdm.tqdm(data_loader, desc=f"Rank {rank}")

    init_dataset = True
    for batch in data_loader:
        if not is_extracted:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                    for k, v in value.items():
                        batch[key][k] = v.to(device)

        with autocast(enabled=use_fp16):
            if is_quantizer:
                codes_batched, encode_batched, ids_list_batched = model.module.quantizer(
                    batch,
                    evaluation=True,   
                    encode_mbeir_batch=(not is_extracted), 
                    code_output=True, 
                    encode_output=True
                )
            else:
                codes_batched, encode_batched, ids_list_batched = model(
                    batch, 
                    cand_codes=cand_codes, 
                    num_beams=num_beams, 
                    encode_mbeir_batch=True,
                    init_dataset=init_dataset,
                    trie_save_path=trie_save_path
                )

        codes_tensor.append(codes_batched)
        id_tensor.append(ids_list_batched.view(-1,1).to(device))
        if use_embedding:
            encode_tensor.append(encode_batched.half())

        if init_dataset:
            init_dataset = False

    codes_tensor = torch.cat(codes_tensor, dim=0)
    id_tensor = torch.cat(id_tensor, dim=0)
    if use_embedding:
        encode_tensor = torch.cat(encode_tensor, dim=0)

    codes_list = None
    id_list = None
    if use_embedding:
        encodes_list = None

    if dist.is_initialized():
        size_tensor = torch.tensor([codes_tensor.size(0)], dtype=torch.long, device=device)

        if dist.get_rank() == 0:
            gathered_codes = [torch.empty_like(codes_tensor) for _ in range(dist.get_world_size())]
            gathered_ids = [torch.empty_like(id_tensor) for _ in range(dist.get_world_size())]
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
            if use_embedding:
                gathered_encodes = [torch.empty_like(encode_tensor) for _ in range(dist.get_world_size())]
        else:
            gathered_codes = None
            gathered_encodes = None
            gathered_ids = None
            sizes = None

        dist.gather(codes_tensor, gather_list=gathered_codes, dst=0)
        if use_embedding:
            dist.gather(encode_tensor, gather_list=gathered_encodes, dst=0)
        dist.gather(id_tensor, gather_list=gathered_ids, dst=0)
        dist.gather(size_tensor, gather_list=sizes, dst=0)

        if dist.get_rank() == 0:
            torch.set_num_threads(total_cores)
            for i in range(dist.get_world_size()):
                gathered_codes[i] = gathered_codes[i][: sizes[i][0]]
                gathered_ids[i] = gathered_ids[i][: sizes[i][0]]
                if use_embedding:
                    gathered_encodes[i] = gathered_encodes[i][: sizes[i][0]]

            codes_list = torch.cat(gathered_codes, dim=0).cpu().numpy()
            if use_embedding:
                encodes_list = torch.cat(gathered_encodes, dim=0).cpu().numpy()
            id_tensor_all = torch.cat(gathered_ids, dim=0).squeeze(1)
            id_list = id_tensor_all.cpu().tolist()

            assert len(id_list) == codes_list.shape[0]
            assert len(set(id_list)) == len(id_list), "Hashed IDs should be unique"
            print(f"Log: Finished processing embeddings and ids on rank 0.")
            print(f"Log: Converted codes_list to cpu numpy array of type {codes_list.dtype}.")
            torch.set_num_threads(initial_threads_per_process)

    else:
        codes_list = codes_tensor.detach().cpu().numpy()
        id_list = id_tensor.squeeze(1).cpu().tolist()
        if use_embedding:
            encodes_list = encode_tensor.detach().cpu().numpy()

    if use_embedding:
        return codes_list, encodes_list, id_list
    else:
        return codes_list, id_list

@torch.no_grad()
def generate_noembed_codes_for_dataset(
                                model, 
                                data_loader, 
                                device, 
                                use_fp16=True, 
                                is_quantizer=True, 
                                num_beams=10, 
                                cand_codes=None, 
                                is_extracted=False,
                                trie_save_path = None
                                ):
    codes_tensor = []
    id_list = []

    total_cores = os.cpu_count()
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        initial_threads_per_process = total_cores // world_size
        torch.set_num_threads(initial_threads_per_process)
        data_loader = tqdm.tqdm(data_loader, desc=f"Rank {rank}")

    init_dataset = True
    for batch in data_loader:
        if not is_extracted:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                    for k, v in value.items():
                        batch[key][k] = v.to(device)
                    
        # Enable autocast to FP16
        with autocast(enabled=use_fp16):
            if is_quantizer:
                codes_batched, encode_batched, ids_list_batched = model.module.quantizer(batch, 
                                                                                        evaluation=True,   
                                                                                        encode_mbeir_batch=(not is_extracted), 
                                                                                        code_output=True, 
                                                                                        encode_output=True)

            else:
                codes_batched, encode_batched, ids_list_batched = model(batch, 
                                                                        cand_codes=cand_codes, 
                                                                        num_beams=num_beams, 
                                                                        encode_mbeir_batch=True,
                                                                        init_dataset=init_dataset,
                                                                        trie_save_path=trie_save_path)
                
        codes_tensor.append(codes_batched)
        id_list.extend(ids_list_batched)

        if init_dataset:
            init_dataset = False
        
    codes_tensor = torch.cat(codes_tensor, dim=0)
    codes_list = None
    encodes_list= None

    if dist.is_initialized():
        # First, share the sizes of tensors across processes
        size_tensor = torch.tensor([codes_tensor.size(0)], dtype=torch.long, device=device)
        
        # Allocate tensors to gather results on rank 0
        if dist.get_rank() == 0:
            # Allocate tensors with the correct sizes on rank 0
            gathered_codes = [torch.empty_like(codes_tensor.clone().detach()) for _ in range(dist.get_world_size())]
            id_list_gathered = [list() for _ in range(dist.get_world_size())]
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
        else:
            gathered_codes = None
            gathered_encodes = None
            id_list_gathered = None
            sizes = None

        # Synchronize all processes before gathering data
        dist.barrier()

        # Gather codes from all processes
        dist.gather(codes_tensor, gather_list=gathered_codes, dst=0)
        # Gather ids from all processes
        dist.gather_object(id_list, object_gather_list=id_list_gathered, dst=0)
        # Gather sizes from all processes
        dist.gather(size_tensor, gather_list=sizes, dst=0)

        # Synchronize all processes after gathering data
        dist.barrier()

        if dist.get_rank() == 0:
            torch.set_num_threads(total_cores)
            gathered_codes[-1] = gathered_codes[-1][: sizes[-1][0]]
            codes_list = torch.cat(gathered_codes, dim=0)

            # Flatten gathered ids
            id_list = [id for sublist in id_list_gathered for id in sublist]
            assert len(id_list) == codes_list.size(0)
            # Check unique ids
            assert len(id_list) == len(set(id_list)), "Hashed IDs should be unique"
            print(f"Log: Finished processing embeddings and ids on rank 0.")

            # Note: we are using float16 to save space, and the precision loss is negligible.
            codes_list = codes_list.detach().cpu().numpy()
            print(f"Log: Converted codes_list to cpu numpy array of type {codes_list.dtype}.")

            # Reset number of threads to initial value after conversion
            torch.set_num_threads(initial_threads_per_process)

        dist.barrier()
    else:
        codes_list = codes_tensor.detach().cpu().numpy()
        
    return codes_list, id_list




@torch.no_grad()
def generate_codes_for_config(model, img_preprocess_fn, clip_tokenizer, seq2seq_tokenizer, config):
    genir_dir = config.genir_dir
    mbeir_data_dir = config.mbeir_data_dir
    retrieval_config = config.retrieval_config
    gen_code_dir_name = retrieval_config.gen_code_dir_name
    expt_dir_name = config.experiment.path_suffix

    data_config = config.data_config
    query_instruct_path = data_config.query_instruct_path
    cand_pool_dir = data_config.cand_pool_dir_name
    union_cand_pool_dir = data_config.union_cand_pool_dir_name
    image_size = tuple(map(int, data_config.image_size.split(",")))
    is_extracted = data_config.is_extracted

    splits = []

    gen_code_cand_pool_config = retrieval_config.cand_pools_config
    if gen_code_cand_pool_config and gen_code_cand_pool_config.enable_gen_code:
        split_name = "cand_pool"
        split_dir_name = data_config.cand_pool_dir_name
        cand_pool_name_list = gen_code_cand_pool_config.cand_pools_name_to_gen_code
        splits.append(
            (
                split_name,
                split_dir_name,
                [None] * len(cand_pool_name_list),
                cand_pool_name_list,
            )
        )
        
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
        gen_code_dataset_config = getattr(retrieval_config, f"{split_name}_datasets_config", None)
        if gen_code_dataset_config and gen_code_dataset_config.enable_gen_code:
            dataset_name_list = getattr(gen_code_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(gen_code_dataset_config, "correspond_cand_pools_name", None)
            splits.append((split_name, split_dir_name, dataset_name_list, cand_pool_name_list))
            assert len(dataset_name_list) == len(cand_pool_name_list), "Mismatch between datasets and candidate pools."

    if dist_utils.is_main_process():
        print("-" * 30)
        for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
            if split_name == "cand_pool":
                print(f"Split: {split_name}, Split dir: {split_dir}, Candidate pools to gen_code: {cand_pool_name_list}")
            else:
                print(f"Split: {split_name}, Split dir: {split_dir}, Datasets to retrieval: {dataset_name_list}")
            print("-" * 30)

    for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
        for dataset_name, cand_pool_name in zip(dataset_name_list, cand_pool_name_list):
            if split_name == "cand_pool" and is_extracted:
                cand_pool_name = cand_pool_name.lower()
                if cand_pool_name in EXCEPTIONAL_CAND_POOLS:
                    cand_pool_name = cand_pool_name + '_test'
                cand_pool_file_name = f"mbeir_{cand_pool_name}_{split_name}.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)
                cand_pool_extracted_path = os.path.join(genir_dir, data_config.extracted_dir)
                extracted_file_name = f"{split_name}_{cand_pool_name}_IT_dict.pt"
                cand_pool_extracted_path = os.path.join(cand_pool_extracted_path, extracted_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"\Generate Code Log: Generating codes from {cand_pool_extracted_path}...")
                    print_config = True

                dataset = MBEIRDictCandDataset(
                    mbeir_data_dir=config.mbeir_data_dir,
                    cand_pool_path = cand_pool_data_path,
                    pool_dict_dir = cand_pool_extracted_path,
                )
                collator = None

            elif split_name == "cand_pool" and not is_extracted:
                cand_pool_name = cand_pool_name.lower()
                if cand_pool_name in EXCEPTIONAL_CAND_POOLS:
                    cand_pool_name = cand_pool_name + '_test'
                cand_pool_file_name = f"mbeir_{cand_pool_name}_{split_name}.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"Generate Code Log: Generating codes for {cand_pool_data_path}...")
                    print_config = True
                dataset = MBEIRCandidatePoolDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    cand_pool_data_path=cand_pool_data_path,
                    img_preprocess_fn=img_preprocess_fn,
                    print_config=print_config,
                )
                collator = MBEIRCandidatePoolCollator(
                    tokenizer=clip_tokenizer,
                    image_size=image_size,
                )

            else:
                dataset_name = dataset_name.lower()
                query_data_name = f"mbeir_{dataset_name}_{split_name}.jsonl"
                query_data_path = os.path.join(split_dir, query_data_name)

                cand_pool_name = cand_pool_name.lower()
                if cand_pool_name in EXCEPTIONAL_CAND_POOLS:
                    cand_pool_name = cand_pool_name + '_test'
                cand_pool_file_name = f"mbeir_{cand_pool_name}_cand_pool.jsonl"
                if cand_pool_name == "union":
                    cand_pool_data_path = os.path.join(union_cand_pool_dir, f"mbeir_{cand_pool_name}_{split_name}_cand_pool.jsonl")
                else:
                    cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"Generate Code Log: Generating codes for {query_data_path} with {cand_pool_data_path}...")
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
                    tokenizer=clip_tokenizer,
                    image_size=image_size,
                    mode=mode,
                    seq2seq_tokenizer=seq2seq_tokenizer,
                )

            batch_size = config.dataloader_config.batch_size
            num_workers = config.dataloader_config.num_workers

            num_tasks = dist_utils.get_world_size()
            global_rank = dist_utils.get_rank()
            sampler = ContiguousDistributedSampler(
                dataset,
                num_replicas=num_tasks,
                rank=global_rank,
            )
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=False,
                collate_fn=collator,
                drop_last=False,
            )

            if dist.is_initialized():
                dist.barrier()

            cand_trie_path = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, "cand_pool", f"mbeir_{cand_pool_name}_cand_pool_trie.pkl")
            
            if dist_utils.is_main_process():
                if split_name == "cand_pool":
                    print(f"Log: Data loader for {cand_pool_data_path} is set up.")
                    print(f"Log: Generating codes for {cand_pool_data_path}...")
                else:
                    print(
                        f"Log: Data loader for {query_data_path} with candidate pool {cand_pool_data_path} is set up."
                    )
                    print(f"Log: Generating codes for {query_data_path} ...")
                print(f"Inference with half precision: {config.retrieval_config.use_fp16}")

            if split_name == "cand_pool" and is_extracted:
                codes, embeddings, id_list = generate_codes_for_dataset(
                    model=model,
                    data_loader=data_loader,
                    device=config.dist_config.gpu_id,
                    use_fp16=config.retrieval_config.use_fp16,
                    is_extracted=True,
                )
            elif split_name == "cand_pool" and not is_extracted:
                codes, embeddings, id_list = generate_codes_for_dataset(
                    model=model,
                    data_loader=data_loader,
                    device=config.dist_config.gpu_id,
                    use_fp16=config.retrieval_config.use_fp16,
                )
            else:
                # load cand_codes per each dataset
                cand_code_path = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, "cand_pool", f"mbeir_{cand_pool_name}_cand_pool_codes.npy")
                if os.path.exists(cand_code_path):
                    cand_codes = np.load(cand_code_path)
                    print(f"Log: Loaded candidate pool codes from {cand_code_path}.")
                    cand_codes = np.load(cand_code_path)
                    print(f"Log: Loaded candidate pool codes from {cand_code_path}.")

                else:
                    print(f"Warning: Candidate pool codes not found at {cand_code_path}. Using None for cand_codes.")
                    cand_codes = None

                if retrieval_config.rerank:
                    codes, embeddings, id_list = generate_codes_for_dataset(
                        model=model,
                        data_loader=data_loader,
                        device=config.dist_config.gpu_id,
                        use_fp16=config.retrieval_config.use_fp16,
                        is_quantizer=False,
                        cand_codes=cand_codes,
                        num_beams = config.retrieval_config.num_beams,
                        trie_save_path = cand_trie_path
                    )
                else:
                    codes, id_list = generate_noembed_codes_for_dataset(
                        model=model,
                        data_loader=data_loader,
                        device=config.dist_config.gpu_id,
                        use_fp16=config.retrieval_config.use_fp16,
                        is_quantizer=False,
                        cand_codes=cand_codes,
                        num_beams = config.retrieval_config.num_beams,
                        trie_save_path = cand_trie_path
                    )

            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Log: Codes list length: {len(codes)}")
                if retrieval_config.rerank:
                    print(f"Log: Embedding list length: {len(embeddings)}")
                print(f"Log: ID list length: {len(id_list)}")

                mid_name = cand_pool_name if split_name == "cand_pool" else dataset_name
                save_path = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, split_name)

                code_data_name = f"mbeir_{mid_name}_{split_name}_codes.npy"
                code_path = os.path.join(save_path, code_data_name)
                os.makedirs(os.path.dirname(code_path), exist_ok=True)
                np.save(code_path, codes)
                print(f"Log: Saved codes to {code_path}.")

                if retrieval_config.rerank:
                    embedding_data_name = f"mbeir_{mid_name}_{split_name}_embeddings.npy"
                    embeddings_path = os.path.join(save_path, embedding_data_name)
                    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
                    np.save(embeddings_path, embeddings)
                    print(f"Log: Saved Embeddings to {embeddings_path}.")

                id_data_name = f"mbeir_{mid_name}_{split_name}_ids.npy"
                id_path = os.path.join(save_path, id_data_name)
                os.makedirs(os.path.dirname(id_path), exist_ok=True)
                np.save(id_path, id_list)
                print(f"Log: Saved ids to {id_path}.")

            if dist.is_initialized():
                dist.barrier()

            del codes
            del id_list
            del data_loader
            del dataset
            del collator
            del sampler
            try:
                del embeddings
            except:
                None

            gc.collect()
            torch.cuda.empty_cache()

        # Union pool embeddings

        if split_name == "cand_pool" and gen_code_cand_pool_config.gen_code_union_pool:
            # To efficiently generate codes for the union(global) pool,
            # We concat previously saved codes and ids from single(local) pool
            # Instead of embed the union pool directly.
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\nLog: Generating codes for union pool...")

                # Increase number of threads for rank 0
                world_size = dist.get_world_size()
                total_cores = os.cpu_count()
                initial_threads_per_process = total_cores // world_size
                torch.set_num_threads(total_cores)

                all_codes = []
                all_embeddings = []
                all_ids = []
                for cand_pool_name in cand_pool_name_list:
                    cand_pool_name = cand_pool_name.lower()
                    if cand_pool_name in EXCEPTIONAL_CAND_POOLS:
                        cand_pool_name = cand_pool_name + '_test'
                    cand_pool_name = f"mbeir_{cand_pool_name}_{split_name}"
                    code_data_name = f"{cand_pool_name}_codes.npy"
                    embedding_data_name = f"{cand_pool_name}_embeddings.npy"
                    id_data_name = f"{cand_pool_name}_ids.npy"
                    code_path = os.path.join(
                        genir_dir,
                        gen_code_dir_name,
                        expt_dir_name,
                        split_name,
                        code_data_name,
                    )
                    embedding_path = os.path.join(
                        genir_dir,
                        gen_code_dir_name,
                        expt_dir_name,
                        split_name,
                        embedding_data_name,
                    )
                    id_path = os.path.join(
                        genir_dir,
                        gen_code_dir_name,
                        expt_dir_name,
                        split_name,
                        id_data_name,
                    )
                    all_codes.append(np.load(code_path))
                    if retrieval_config.rerank:
                        all_embeddings.append(np.load(embedding_path))
                    all_ids.append(np.load(id_path))
                    print(f"Log: Concatenating codes from {code_path} and ids from {id_path}.") 
                    if retrieval_config.rerank:
                        print("All Num Code: {}, All Num Emb: {}, All Num ID: {}".format(len(np.concatenate(all_codes, axis=0)), len(np.concatenate(all_embeddings, axis=0)), len(np.concatenate(all_ids, axis=0))))
                    else:
                        print("All Num Code: {}, All Num ID: {}".format(len(np.concatenate(all_codes, axis=0)), len(np.concatenate(all_ids, axis=0))))

                all_codes = np.concatenate(all_codes, axis=0)
                if retrieval_config.rerank:
                    all_embeddings = np.concatenate(all_embeddings, axis=0)
                all_ids = np.concatenate(all_ids, axis=0)
                assert len(all_codes) == len(all_ids), "Mismatch between codes and IDs length."
                if retrieval_config.rerank:
                    assert len(all_embeddings) == len(all_ids), "Mismatch between embeddings and IDs length."
                print(f"Log: all_codes length: {len(all_codes)} and all_ids length: {len(all_ids)}.")

                # Save the codes to .npy
                code_data_name = f"mbeir_union_{split_name}_codes.npy"
                code_path = os.path.join(
                    genir_dir,
                    gen_code_dir_name,
                    expt_dir_name,
                    split_name,
                    code_data_name,
                )
                os.makedirs(os.path.dirname(code_path), exist_ok=True)
                np.save(code_path, all_codes)
                print(f"Log: Saved codes to {code_path}.")

                if retrieval_config.rerank:
                    # Save the embeddings to .npy
                    embeddings_data_name = f"mbeir_union_{split_name}_embeddings.npy"
                    embeddings_path = os.path.join(
                        genir_dir,
                        gen_code_dir_name,
                        expt_dir_name,
                        split_name,
                        embeddings_data_name,
                    )
                    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
                    np.save(embeddings_path, all_embeddings)
                    print(f"Log: Saved embeddings to {embeddings_path}.")

                # Save the IDs to .npy
                id_data_name = f"mbeir_union_{split_name}_ids.npy"
                id_path = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, split_name, id_data_name)
                os.makedirs(os.path.dirname(id_path), exist_ok=True)
                np.save(id_path, all_ids)
                print(f"Log: Saved ids to {id_path}.")


                # Delete the codes and IDs to free up memory
                del all_codes
                if retrieval_config.rerank:
                    del all_embeddings
                del all_ids

                # Explicitly call the garbage collector
                gc.collect()

                # Reset number of threads to initial value after conversion
                torch.set_num_threads(initial_threads_per_process)

            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish saving the codes and ids.



def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0

    top_k_retrieved_indices_set = set(retrieved_indices[:k])
    relevant_docs_set = set(relevant_docs)

    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0


def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if int(relevance_score) > 0:
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(
        f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid


# Create a hash map for candidate pool codes
def create_hash_map(codes, ids):
    hash_map = defaultdict(list)
    for code, id in zip(codes, ids):
        hash_map[tuple(code)].append(id)
    return hash_map

def create_id_to_index_map(ids):
    return {id: index for index, id in enumerate(ids)}

# Function to retrieve indices for a query using hash map
def retrieve_indices_for_query(query_code, cand_pool_hash_map, k):
    unique_candidates = []
    for code in query_code:
        code_tuple = tuple(code)
        if code_tuple in cand_pool_hash_map:
            unique_candidates.extend(cand_pool_hash_map[code_tuple])

    # Ensure the order of unique_candidates matches the order in query_code
    # unique_candidates = sorted(set(unique_candidates), key=unique_candidates.index)\
    
    # If the number of unique candidates exceeds k, trim to k
    if len(unique_candidates) > k:
        unique_candidates = unique_candidates[:k]

    return unique_candidates

def compute_cosine_similarities(query_embedding, candidate_embeddings):
    query_embedding = F.normalize(torch.tensor(query_embedding).unsqueeze(0), p=2, dim=1)
    candidate_embeddings = F.normalize(torch.tensor(candidate_embeddings), p=2, dim=1)
    return F.linear(query_embedding, candidate_embeddings).squeeze(0)

def retrieve_and_rank_for_query(query_code, 
                                cand_pool_hash_map, 
                                query_embedding,
                                cand_pool_embeddings, 
                                cand_pool_ids, 
                                id_to_index_map, 
                                k):
    unique_candidates = []
    for code in query_code:
        code_tuple = tuple(code)
        if code_tuple in cand_pool_hash_map:
            unique_candidates.extend(cand_pool_hash_map[code_tuple])
    
    # Remove duplicates while preserving order
    unique_candidates = list(dict.fromkeys(unique_candidates))

    # Convert to PyTorch tensors
    query_embedding_tensor = torch.tensor(query_embedding, dtype=torch.float32)
    candidate_indices = [id_to_index_map[cand_id] for cand_id in unique_candidates]
    candidate_embeddings_tensor = torch.tensor(cand_pool_embeddings[candidate_indices], dtype=torch.float32)

    # Calculate cosine similarities
    # similarities = F.cosine_similarity(query_embedding_tensor.unsqueeze(0), candidate_embeddings_tensor)
    similarities = compute_cosine_similarities(query_embedding_tensor, candidate_embeddings_tensor)
    
    # Get top k indices
    top_k_indices = similarities.argsort(descending=True)[:k]
    
    # Map back to candidate IDs
    top_k_candidates = [unique_candidates[i] for i in top_k_indices.tolist()]
    
    return top_k_candidates


def generative_retrieve(config):
    genir_dir = config.genir_dir
    mbeir_data_dir = config.mbeir_data_dir
    retrieval_config = config.retrieval_config
    qrel_dir_name = retrieval_config.qrel_dir_name
    gen_code_dir_name = retrieval_config.gen_code_dir_name
    expt_dir_name = config.experiment.path_suffix

    results_dir_name = retrieval_config.results_dir_name
    exp_results_dir = os.path.join(genir_dir, results_dir_name, expt_dir_name)
    os.makedirs(exp_results_dir, exist_ok=True)
    exp_run_file_dir = os.path.join(exp_results_dir, "run_files")
    os.makedirs(exp_run_file_dir, exist_ok=True)
    exp_tsv_results_dir = os.path.join(exp_results_dir, "final_tsv")
    os.makedirs(exp_tsv_results_dir, exist_ok=True)

    splits = []
    dataset_types = ["train", "val", "test"]
    for split_name in dataset_types:
        retrieval_dataset_config = getattr(retrieval_config, f"{split_name}_datasets_config", None)
        if retrieval_dataset_config and retrieval_dataset_config.enable_retrieve:
            dataset_name_list = getattr(retrieval_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(retrieval_dataset_config, "correspond_cand_pools_name", None)
            qrel_name_list = getattr(retrieval_dataset_config, "correspond_qrels_name", None)
            metric_names_list = getattr(retrieval_dataset_config, "correspond_metrics_name", None)
            dataset_gen_code_dir = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, split_name)
            splits.append(
                (
                    split_name,
                    dataset_gen_code_dir,
                    dataset_name_list,
                    cand_pool_name_list,
                    qrel_name_list,
                    metric_names_list,
                )
            )
            assert (
                len(dataset_name_list) == len(cand_pool_name_list) == len(qrel_name_list) == len(metric_names_list)
            ), "Mismatch between datasets and candidate pools and qrels."

    print("-" * 30)
    for (
        split_name,
        dataset_gen_code_dir,
        dataset_name_list,
        cand_pool_name_list,
        qrel_name_list,
        metric_names_list,
    ) in splits:
        print(
            f"Split: {split_name}, Retrieval Datasets: {dataset_name_list}, Candidate Pools: {cand_pool_name_list}, Metric: {metric_names_list})"
        )
        print("-" * 30)

    eval_results = []
    qrel_dir = os.path.join(mbeir_data_dir, qrel_dir_name)
    cand_dir = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, "cand_pool")
    for (
        split,
        dataset_gen_code_dir,
        dataset_name_list,
        cand_pool_name_list,
        qrel_name_list,
        metric_names_list,
    ) in splits:
        for dataset_name, cand_pool_name, qrel_name, metric_names in zip(
            dataset_name_list, cand_pool_name_list, qrel_name_list, metric_names_list
        ):
            print("\n" + "-" * 30)
            print(f"Retriever: Retrieving for query:{dataset_name} | split:{split} | from cand_pool:{cand_pool_name}")

            dataset_name = dataset_name.lower()
            cand_pool_name = cand_pool_name.lower()
            if cand_pool_name in EXCEPTIONAL_CAND_POOLS:
                cand_pool_name = cand_pool_name + '_test'
            qrel_name = qrel_name.lower()

            qrel_path = os.path.join(qrel_dir, split, f"mbeir_{qrel_name}_{split}_qrels.txt")
            qrel, qid_to_taskid = load_qrel(qrel_path)

            gen_code_query_id_path = os.path.join(dataset_gen_code_dir, f"mbeir_{dataset_name}_{split}_ids.npy")
            query_ids = np.load(gen_code_query_id_path)
            gen_code_query_path = os.path.join(dataset_gen_code_dir, f"mbeir_{dataset_name}_{split}_codes.npy")
            query_codes = np.load(gen_code_query_path)
            if retrieval_config.rerank:
                embeddings_query_path = os.path.join(dataset_gen_code_dir, f"mbeir_{dataset_name}_{split}_embeddings.npy")
                query_embeddings = np.load(embeddings_query_path)

            cand_pool_code_path = os.path.join(cand_dir, f"mbeir_{cand_pool_name}_cand_pool_codes.npy")
            if retrieval_config.rerank:
                cand_pool_embeddings_path = os.path.join(cand_dir, f"mbeir_{cand_pool_name}_cand_pool_embeddings.npy")
            cand_pool_id_path = os.path.join(cand_dir, f"mbeir_{cand_pool_name}_cand_pool_ids.npy")
            cand_pool_codes = np.load(cand_pool_code_path)
            if retrieval_config.rerank:
                cand_pool_embeddings = np.load(cand_pool_embeddings_path)
            cand_pool_ids = np.load(cand_pool_id_path)

            metric_list = [metric.strip() for metric in metric_names.split(",")]
            metric_recall_list = [metric for metric in metric_list if "recall" in metric.lower()]

            cand_pool_hash_map = create_hash_map(cand_pool_codes, cand_pool_ids)
            id_to_index_map = create_id_to_index_map(cand_pool_ids)

            k = max([int(metric.split("@")[1]) for metric in metric_recall_list])

            # Retrieve indices for all queries
            retrieved_indices = []
            for i, query_code in enumerate(query_codes):
                if retrieval_config.rerank:
                    retrieved_indices.append(retrieve_and_rank_for_query(
                        query_code, 
                        cand_pool_hash_map, 
                        query_embeddings[i], 
                        cand_pool_embeddings, 
                        cand_pool_ids,
                        id_to_index_map, 
                        k
                    ))
                else:
                    retrieved_indices.append(retrieve_indices_for_query(query_code, cand_pool_hash_map, k))


            if not os.path.exists(exp_run_file_dir):
                os.makedirs(exp_run_file_dir)
            run_id = f"mbeir_{dataset_name}_single_pool_{split}"
            run_file_name = f"{run_id}_run.txt"
            run_file_path = os.path.join(exp_run_file_dir, run_file_name)
            with open(run_file_path, "w") as run_file:
                for idx, indices in enumerate(retrieved_indices):
                    qid = unhash_qid(query_ids[idx])
                    task_id = qid_to_taskid[qid]
                    for rank, retrieved_id in enumerate(indices, start=1):
                        run_file.write(f"{qid} Q0 {retrieved_id} {rank} {1.0} {run_id} {task_id}\n")
            print(f"Retriever: Run file saved to {run_file_path}")

            recall_values_by_task = defaultdict(lambda: defaultdict(list))
            for i, retrieved_indices_for_qid in enumerate(retrieved_indices):
                retrieved_indices_for_qid = [unhash_did(idx) for idx in retrieved_indices_for_qid]
                qid = unhash_qid(query_ids[i])
                relevant_docs = qrel[qid]
                task_id = qid_to_taskid[qid]
                
                # Compute Recall@k for each metric
                for metric in metric_recall_list:
                    k = int(metric.split("@")[1])
                    recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                    recall_values_by_task[task_id][metric].append(recall_at_k)
                                
            for task_id, recalls in recall_values_by_task.items():
                task_name = get_mbeir_task_name(int(task_id))
                result = {
                    "TaskID": int(task_id),
                    "Task": task_name,
                    "Dataset": dataset_name,
                    "Split": split,
                    "CandPool": cand_pool_name,
                }
                for metric in metric_recall_list:
                    mean_recall_at_k = round(sum(recalls[metric]) / len(recalls[metric]), 4)
                    result[metric] = mean_recall_at_k
                    print(f"Retriever: Mean {metric}: {mean_recall_at_k}")
                eval_results.append(result)

    dataset_order = {
        "visualnews_task0": 1,
        "mscoco_task0": 2,
        "fashion200k_task0": 3,
        "webqa_task1": 4,
        "edis_task2": 5,
        "webqa_task2": 6,
        "visualnews_task3": 7,
        "mscoco_task3": 8,
        "fashion200k_task3": 9,
        "nights_task4": 10,
        "oven_task6": 11,
        "infoseek_task6": 12,
        "fashioniq_task7": 13,
        "cirr_task7": 14,
        "oven_task8": 15,
        "infoseek_task8": 16,
    }
    split_order = {"val": 1, "test": 2}
    cand_pool_order = {"union": 99}
    eval_results_sorted = sorted(
        eval_results,
        key=lambda x: (
            x["TaskID"],
            dataset_order.get(x["Dataset"].lower(), 99),
            split_order.get(x["Split"].lower(), 99),
            cand_pool_order.get(x["CandPool"].lower(), 0),
        ),
    )

    grouped_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    available_recall_metrics = [
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "Recall@50",
    ]
    for result in eval_results_sorted:
        key = (result["TaskID"], result["Task"], result["Dataset"], result["Split"])
        for metric in available_recall_metrics:
            grouped_results[key][result["CandPool"]].update({metric: result.get(metric, None)})

    if retrieval_config.write_to_tsv:
        date_time = datetime.now().strftime("%m-%d-%H")
        tsv_file_name = f"eval_results_{date_time}.tsv"
        tsv_file_path = os.path.join(exp_tsv_results_dir, tsv_file_name)
        tsv_data = []
        header = [
            "TaskID",
            "Task",
            "Dataset",
            "Split",
            "Metric",
            "CandPool",
            "Value",
            "UnionPool",
            "UnionValue",
        ]
        tsv_data.append(header)

        for (task_id, task, dataset, split), cand_pools in grouped_results.items():
            union_results = cand_pools.get("union", {})
            for metric in available_recall_metrics:
                for cand_pool, metrics in cand_pools.items():
                    if cand_pool != "union":
                        row = [
                            task_id,
                            task,
                            dataset,
                            split,
                            metric,
                            cand_pool,
                            metrics.get(metric, None),
                        ]
                        if row[-1] is None:
                            continue
                        if union_results:
                            row.extend(["union", union_results.get(metric, "N/A")])
                        else:
                            row.extend(["", ""])
                        tsv_data.append(row)

        with open(tsv_file_path, "w", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t")
            for row in tsv_data:
                writer.writerow(row)

        print(f"Retriever: Results saved to {tsv_file_path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config):
    seed = config.seed + dist_utils.get_rank()
    set_seed(seed)

    model = build_model_from_config(config)

    if not callable(getattr(model, "encode_mbeir_batch")):
        raise AttributeError("The provided model does not have a callable 'encode_mbeir_batch' method.")
    if not callable(getattr(model, "get_img_preprocess_fn")):
        raise AttributeError("The provided model does not have an 'get_img_preprocess_fn' attribute.")
    if not callable(getattr(model, "get_clip_tokenizer")):
        raise AttributeError("The provided model does not have a 'get_clip_tokenizer' attribute.")
    if not callable(getattr(model, "get_seq2seq_tokenizer")):
        raise AttributeError("The provided model does not have a 'get_seq2seq_tokenizer' attribute.")

    img_preprocess_fn = model.get_img_preprocess_fn()
    clip_tokenizer = model.get_clip_tokenizer()
    seq2seq_tokenizer = model.get_seq2seq_tokenizer()

    model = model.to(config.dist_config.gpu_id)
    if config.dist_config.distributed_mode:
        model = DDP(
            model,
            device_ids=[config.dist_config.gpu_id],
            broadcast_buffers=False,
            find_unused_parameters=False
        )
    model.eval()
    
    print(f"Models are set up on GPU {config.dist_config.gpu_id}.")
    
    with torch.inference_mode():
        generate_codes_for_config(
            model=model,
            img_preprocess_fn=img_preprocess_fn,
            clip_tokenizer=clip_tokenizer,
            seq2seq_tokenizer=seq2seq_tokenizer,
            config=config,
        )
        
        if dist_utils.is_main_process():
            generative_retrieve(config)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Codes for MBEIR")
    parser.add_argument("--genir_dir", type=str, default="/home/ubuntu/tjddus/GenIR/GENIUS")
    parser.add_argument("--mbeir_data_dir", type=str, default="/home/ubuntu/tjddus/GenIR/GENIUS/mbeir_data")
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)

    config.genir_dir = args.genir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    args.dist_url = config.dist_config.dist_url
    dist_utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    if dist_utils.is_main_process():
        print(OmegaConf.to_yaml(config, sort_keys=False))

    main(config)


    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
