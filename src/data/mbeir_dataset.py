# Standard library
import base64
import io
import json
import os
import random
from enum import Enum
from typing import Callable, List, Union, Any

# Third-party
import numpy as np
import torch
from PIL import Image

from collections import defaultdict
from torch.utils.data import Dataset
from typeguard import typechecked
import pickle
from concurrent.futures import ThreadPoolExecutor

# Project files
from data.preprocessing.utils import (
    format_string,
    hash_did,
    hash_qid,
    get_mbeir_task_id,
)


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    DICT = 'dict'


class MBIERDatasetFromDict(Dataset):
    def __init__(self, query_dir, pool_dir):
        print(f"\n---Loading Mbeir Dataset from Dictionary---")
        with open(query_dir, "rb") as f:
            self.query_dict = pickle.load(f)
        with open(pool_dir, "rb") as f:
            self.pool_dict = pickle.load(f)

        self.all_dict = {}
        for (key, value) in self.query_dict.items():
            self.all_dict['q_' + str(key)] = torch.from_numpy(value)
        for (key, value) in self.pool_dict.items():
            self.all_dict['p_' + str(key)] = torch.from_numpy(value)
        self.all_keys = list(self.all_dict.keys())

    def __len__(self):
        return len(self.all_keys)

    def __getitem__(self, idx):
        embedding = self.all_dict[self.all_keys[idx]]
        return embedding, self.all_keys[idx]

class MBIERDatasetFromList(Dataset):
    def __init__(self, query_dir, pool_dir):
        print(f"\n---Loading Mbeir Dataset from List---")
        with open(query_dir, "rb") as f:
            self.query_list = pickle.load(f)
        with open(pool_dir, "rb") as f:
            self.pool_list = pickle.load(f)

    def __len__(self):
        assert len(self.query_list) == len(self.pool_list)
        return len(self.query_list)

    def __getitem__(self, idx):
        q_emb = self.query_list[idx]
        p_emb = self.pool_list[idx]
        return q_emb, p_emb


class MBEIRDatasetBase(Dataset):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        img_preprocess_fn,
    ):
        """
        Initialize the MBEIRDataset.

        Args:
        - datapath (str): Path to the data.
        - img_preprocess_fn (function): Image preprocessing function.
        - mbeir_data_dir (str): Root directory of the MBEIR dataset.
        - training (bool): Indicator if the dataset is for training.
        """
        self.mbeir_data_dir = mbeir_data_dir
        self.img_preprocess_fn = img_preprocess_fn or (lambda x: x)

    def __len__(self):
        raise NotImplementedError("This method should be implemented in derived classes.")

    def _load_data_jsonl(self, datapath):
        data_entries = []
        with open(datapath, "r") as fin:
            for line in fin:
                data_entry = json.loads(line)
                data_entries.append(data_entry)
        return data_entries

    def _load_data(self, data_path):
        """Validate and load data."""
        full_data_path = os.path.join(self.mbeir_data_dir, data_path)
        assert os.path.exists(full_data_path), f"Data Path {full_data_path} does not exist"
        assert full_data_path.endswith(".jsonl"), f"Data Path {full_data_path} is not a jsonl file"
        data_entries = self._load_data_jsonl(full_data_path)
        return data_entries

    def _load_query_data(self, query_data_path):
        self.query_data = self._load_data(query_data_path)

    def _load_cand_pool(self, cand_pool_data_path):
        self.cand_pool = self._load_data(cand_pool_data_path)

    def _load_query_instructions(self, instructions_path):
        """Validate and load instructions."""
        full_instructions_path = os.path.join(self.mbeir_data_dir, instructions_path)
        # Validate the path and file extension
        assert os.path.exists(full_instructions_path), f"Instructions Path {full_instructions_path} does not exist"
        assert full_instructions_path.endswith(".tsv"), f"Instructions Path {full_instructions_path} is not a tsv file"
        prompts_dict = {}
        with open(full_instructions_path, "r") as f:
            next(f)  # Skip the header line
            for line in f.readlines():
                parts = line.strip().split("\t")
                # Construct the key to be dataset_id, query_modality, cand_modality
                key = f"{parts[3]}, {parts[0]}, {parts[1]}"
                prompts = [p for p in parts[4:] if p]  # Filters out any empty prompts
                prompts_dict[key] = prompts
        self.query_instructions = prompts_dict

    def _load_and_preprocess_image(self, query_img_path):
        """Load an image given a path"""
        if not query_img_path:
            return None
        full_query_img_path = os.path.join(self.mbeir_data_dir, query_img_path)
        assert os.path.exists(full_query_img_path), f"Image Path {full_query_img_path} does not exist"
        image = Image.open(full_query_img_path).convert("RGB")
        image = self.img_preprocess_fn(image)
        return image

    def _get_random_query_prompt(self, dataset_id, query_modality, cand_modality):
        key = f"{dataset_id}, {query_modality}, {cand_modality}"
        prompts = self.query_instructions.get(key, [])
        assert prompts, f"Cannot find prompts for {key}"
        prompt = format_string(random.choice(prompts))
        assert prompt, f"Prompt is empty for {key}"
        return prompt

    def __getitem__(self, index):
        raise NotImplementedError("This method should be implemented in derived classes.")


class MBEIRListInstructioneDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        query_data_path,  # Relate path to the query data
        cand_pool_path,  # Relate path to the candidate pool data
        query_instruct_path,  # Relate path to the query instructions
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
        query_list_dir=None,
        pool_list_dir=None,
        tokenizer=None,
    ):
        super().__init__(mbeir_data_dir, None)

        print(f"\n---Loading Mbeir Dataset from List with Instruction---")
        with open(query_list_dir, "rb") as f:
            self.query_list = pickle.load(f)
        with open(pool_list_dir, "rb") as f:
            self.pool_list = pickle.load(f)
            
        self.tokenizer = tokenizer

        self._load_query_data(query_data_path)
        self._load_cand_pool_as_dict(cand_pool_path)
        self._load_query_instructions(query_instruct_path)

        if print_config:
            self.query_data_path = query_data_path
            self.cand_pool_path = cand_pool_path
            self.query_instruct_path = query_instruct_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Dataset Config---")
        print(f"Query Data Path: {self.query_data_path}")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"--------------------------\n")

    def _load_cand_pool_as_dict(self, cand_pool_data_path):
        self._load_cand_pool(cand_pool_data_path)
        cand_pool_dict = {}
        for cand_pool_entry in self.cand_pool:
            did = cand_pool_entry.get("did")
            assert did, f"Cannot find did for {cand_pool_entry}"
            cand_pool_dict[did] = cand_pool_entry
        self.cand_pool = cand_pool_dict

    def __len__(self):
        return len(self.query_data)

    def _get_random_cand(self, cand_list):
        return random.choice(cand_list)

    def _get_first_cand(self, cand_list):
        return cand_list[0]

    def __getitem__(self, index):
        """Retrieve an item from the dataset by index."""
        mbeir_entry = self.query_data[index]

        query_txt = mbeir_entry.get("query_txt") or ""
        query_img_path = mbeir_entry.get("query_img_path", None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None

        # Randomly sample a positive example
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        assert len(pos_cand_list) > 0, f"Cannot find positive candidates for {mbeir_entry}"

        selected_pos_cand_did = self._get_first_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        assert pos_cand, f"Cannot find positive candidate {selected_pos_cand_did} for {mbeir_entry}"
        # Note: pos_cand_dataset_id should be the same as query_dataset_id but for OVEN and INFOSEEK it is not.
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        # Randomly sample a query prompt
        # Note:query_modality and pos_cand_modality should define the golden modalities of the current mbeir_entry task.
        # neg_cand_modality could be different from pos_cand_modality.
        query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality)
        # query_prompt = self.tokenizer(query_prompt, return_tensors="pt", max_length=30, padding=True).input_ids
        query_prompt_ids = torch.LongTensor(self.tokenizer(query_prompt, padding="max_length", max_length=42, truncation=True, add_special_tokens=False).input_ids)
        
        query_txt = format_string(query_txt)
        
        q_emb = self.query_list[index]
        p_emb = self.pool_list[index]
        
        return q_emb, p_emb, query_prompt_ids
    
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
class MBEIRDictInstructioneDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        query_data_path,  # Relate path to the query data
        cand_pool_path,  # Relate path to the candidate pool data
        query_instruct_path,  # Relate path to the query instructions
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
        query_dict_dir=None,
        pool_dict_dir=None,
        shuffle_cand=True,  # Whether to shuffle the candidates
        return_instruct=False,
        tokenizer=None,
    ):
        super().__init__(mbeir_data_dir, None)
        print(f"\n---Loading Mbeir Dataset from Dictionary with Instruction---")
        self.query_dict = torch.load(query_dict_dir)
        self.qid_to_index = self.query_dict['id_to_index']
        self.pool_dict =  torch.load(pool_dict_dir, map_location=torch.device('cpu'))
        self.did_to_index = self.pool_dict['id_to_index']
        self.shuffle_cand = shuffle_cand
        self.return_instruct = return_instruct
            
        self.tokenizer = tokenizer
        self.select_cand = self._get_random_cand if self.shuffle_cand else self._get_first_cand

        self._load_query_data(query_data_path)
        self._load_cand_pool_as_dict(cand_pool_path)
        self._load_query_instructions(query_instruct_path)

        if print_config:
            self.query_data_path = query_data_path
            self.cand_pool_path = cand_pool_path
            self.query_instruct_path = query_instruct_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Dataset Config---")
        print(f"Query Data Path: {self.query_data_path}")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"--------------------------\n")

    def _load_cand_pool_as_dict(self, cand_pool_data_path):
        self._load_cand_pool(cand_pool_data_path)
        cand_pool_dict = {}
        for cand_pool_entry in self.cand_pool:
            did = cand_pool_entry.get("did")
            assert did, f"Cannot find did for {cand_pool_entry}"
            cand_pool_dict[did] = cand_pool_entry
        self.cand_pool = cand_pool_dict

    def __len__(self):
        return len(self.query_data)

    def _get_random_cand(self, cand_list):
        return random.choice(cand_list)

    def _get_first_cand(self, cand_list):
        return cand_list[0]

    def __getitem__(self, index):
        """Retrieve an item from the dataset by index."""
        mbeir_entry = self.query_data[index]

        query_txt = mbeir_entry.get("query_txt") or ""
        query_img_path = mbeir_entry.get("query_img_path", None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        h_qid = hash_qid(qid)
        query_dataset_id = qid.split(":")[0] if qid else None

        # Randomly sample a positive example
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        assert len(pos_cand_list) > 0, f"Cannot find positive candidates for {mbeir_entry}"

        selected_pos_cand_did = self.select_cand(pos_cand_list)
        h_did = hash_did(selected_pos_cand_did)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        assert pos_cand, f"Cannot find positive candidate {selected_pos_cand_did} for {mbeir_entry}"
        # Note: pos_cand_dataset_id should be the same as query_dataset_id but for OVEN and INFOSEEK it is not.
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        # Randomly sample a query prompt
        # Note:query_modality and pos_cand_modality should define the golden modalities of the current mbeir_entry task.
        # neg_cand_modality could be different from pos_cand_modality.
        if self.tokenizer is not None:
            query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality)
            query_prompt_ids = torch.LongTensor(self.tokenizer(query_prompt, padding="max_length", truncation=True, add_special_tokens=False).input_ids)
            query_txt = format_string(query_txt)
            
        q_emb = self.query_dict['embedding'][self.qid_to_index[h_qid]]
        p_emb = self.pool_dict['embedding'][self.did_to_index[h_did]]
        
        if self.return_instruct and self.tokenizer is not None:
            return q_emb, p_emb, query_prompt_ids
        else:
            return q_emb, p_emb

class MBEIRDictInstructioneDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,
        query_data_path,
        cand_pool_path,
        query_instruct_path,
        returns=None,
        print_config=True,
        query_dict_dir=None,
        pool_dict_dir=None,
        shuffle_cand=True,
        return_instruct=False,
        tokenizer=None,
        clip_tokenizer=None,
        num_positives=1,  # Number of positives per query
    ):
        super().__init__(mbeir_data_dir, None)
        print(f"\n---Loading Mbeir Dataset from Dictionary with Instruction---")
        self.query_dict = torch.load(query_dict_dir)
        self.qid_to_index = self.query_dict['id_to_index']
        self.pool_dict = torch.load(pool_dict_dir, map_location=torch.device('cpu'))
        self.did_to_index = self.pool_dict['id_to_index']
        self.shuffle_cand = shuffle_cand
        self.return_instruct = return_instruct
        self.tokenizer = tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.num_positives = num_positives

        self._load_query_data(query_data_path)
        self._load_cand_pool_as_dict(cand_pool_path)
        self._load_query_instructions(query_instruct_path)
        self._prepare_query_positive_pairs()

        if print_config:
            self.query_data_path = query_data_path
            self.cand_pool_path = cand_pool_path
            self.query_instruct_path = query_instruct_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Dataset Config---")
        print(f"Query Data Path: {self.query_data_path}")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"--------------------------\n")

    def _load_cand_pool_as_dict(self, cand_pool_data_path):
        self._load_cand_pool(cand_pool_data_path)
        cand_pool_dict = {}
        for cand_pool_entry in self.cand_pool:
            did = cand_pool_entry.get("did")
            assert did, f"Cannot find did for {cand_pool_entry}"
            cand_pool_dict[did] = cand_pool_entry
        self.cand_pool = cand_pool_dict

    def _prepare_query_positive_pairs(self):
        self.query_positive_pairs = []
        for query_index, mbeir_entry in enumerate(self.query_data):
            pos_cand_list = mbeir_entry.get("pos_cand_list", [])
            if not pos_cand_list:
                continue

            # Create a mapping of unique DIDs to their first occurrence index in the original list
            unique_dids = {did: i for i, did in enumerate(dict.fromkeys(pos_cand_list))}

            # If we have fewer positives than required, duplicate randomly
            if len(pos_cand_list) < self.num_positives:
                extended_pos_cand_list = pos_cand_list + random.choices(pos_cand_list, k=self.num_positives - len(pos_cand_list))
            else:
                if self.shuffle_cand:
                    extended_pos_cand_list = random.sample(pos_cand_list, self.num_positives)
                else:
                    extended_pos_cand_list = pos_cand_list[:self.num_positives]

            # Add the positive order information, using the mapping from the original list
            self.query_positive_pairs.append((query_index, [(did, unique_dids[did]) for did in extended_pos_cand_list]))

    def __len__(self):
        return len(self.query_positive_pairs)

    def __getitem__(self, index):
        query_index, pos_cand_dids_with_order = self.query_positive_pairs[index]
        mbeir_entry = self.query_data[query_index]

        query_txt = mbeir_entry.get("query_txt") or "<pad>"
        qid = mbeir_entry.get("qid", None)
        h_qid = hash_qid(qid)
        query_dataset_id = qid.split(":")[0] if qid else None
        query_modality = mbeir_entry.get("query_modality", None)

        query = {
            'img_emb': self.query_dict['img'][self.qid_to_index[h_qid]].repeat(self.num_positives, 1),
            'txt_emb': self.query_dict['text'][self.qid_to_index[h_qid]].repeat(self.num_positives, 1),
            'img_mask': self.query_dict['img_mask'][self.qid_to_index[h_qid]].repeat(self.num_positives),
            'txt_mask': self.query_dict['text_mask'][self.qid_to_index[h_qid]].repeat(self.num_positives)
        }

        if self.tokenizer is not None:
            query_txt_id = torch.LongTensor(self.tokenizer(query_txt, padding="max_length", truncation=True, add_special_tokens=False).input_ids)
            query['txt_id'] = query_txt_id.repeat(self.num_positives, 1)

        pool = {
            'img_emb': torch.empty(0),
            'txt_emb': torch.empty(0),
            'img_mask': torch.empty(0),
            'txt_mask': torch.empty(0)
        }
        if self.tokenizer is not None:
            pool['txt_id'] = torch.empty(0, dtype=torch.long)

        h_qids = []
        positive_orders = []

        for pos_cand_did, order in pos_cand_dids_with_order:
            h_did = hash_did(pos_cand_did)
            pos_cand = self.cand_pool.get(pos_cand_did)
            assert pos_cand, f"Cannot find positive candidate {pos_cand_did} for {mbeir_entry}"

            pos_cand_txt = pos_cand.get("txt") or "<pad>"
            pos_cand_txt = format_string(pos_cand_txt)

            pool['img_emb'] = torch.cat([pool['img_emb'], self.pool_dict['img'][self.did_to_index[h_did]].unsqueeze(0)])
            pool['txt_emb'] = torch.cat([pool['txt_emb'], self.pool_dict['text'][self.did_to_index[h_did]].unsqueeze(0)])
            pool['img_mask'] = torch.cat([pool['img_mask'], self.pool_dict['img_mask'][self.did_to_index[h_did]].unsqueeze(0)])
            pool['txt_mask'] = torch.cat([pool['txt_mask'], self.pool_dict['text_mask'][self.did_to_index[h_did]].unsqueeze(0)])

            if self.tokenizer is not None:
                pool_txt_id = torch.LongTensor(self.tokenizer(pos_cand_txt, padding="max_length", truncation=True, add_special_tokens=False).input_ids)
                pool['txt_id'] = torch.cat([pool['txt_id'], pool_txt_id.unsqueeze(0)])

            h_qids.append(h_qid)

        h_qids = torch.tensor(h_qids)

        if self.return_instruct and self.tokenizer is not None:
            query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand.get("modality", None))
            instruct = torch.LongTensor(self.tokenizer(query_prompt, padding="max_length", truncation=True, add_special_tokens=False).input_ids)
            if self.num_positives > 1:
                instruct = instruct.repeat(self.num_positives, 1)
            return query, pool, instruct, h_qids
        elif self.return_instruct and self.clip_tokenizer is not None:
            query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand.get("modality", None))
            instruct = self.clip_tokenizer(query_prompt)
            if self.num_positives > 1:
                instruct = instruct.repeat(self.num_positives, 1)
            return query, pool, instruct, h_qids
        else:
            return query, pool, h_qids

class MBEIRDictCandDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,
        cand_pool_path,
        pool_dict_dir=None,
        returns=None,
        print_config=False,
    ):
        super().__init__(mbeir_data_dir, None)
        print(f"\n---Loading Mbeir Dataset from Dictionary---")
        self.pool_dict = torch.load(pool_dict_dir, map_location=torch.device('cpu'))
        self.did_to_index = self.pool_dict['id_to_index']
        self._load_cand_pool(cand_pool_path)

        if print_config:
            self.cand_pool_path = cand_pool_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Dataset Config---")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"--------------------------\n")

    def __len__(self):
        return len(self.cand_pool)

    def __getitem__(self, index):
        """Retrieve an item from the dataset by index."""

        txt_list, txt_mask_list, img_list, img_mask_list, did_list = [], [], [], [], []
        # Candidate can be indexed directly from the batch

        mbeir_cand_pool_entry = self.cand_pool[index]
        did = mbeir_cand_pool_entry.get("did", None)
        dataset_id = did.split(":")[0] if did else None
        cand_txt = mbeir_cand_pool_entry.get("txt") or ""
        cand_txt = format_string(f"{cand_txt}")
        cand_modality = mbeir_cand_pool_entry.get("modality", None)

        h_did = hash_did(did)
        pool = {
            'img_emb': self.pool_dict['img'][self.did_to_index[h_did]],
            'txt_emb': self.pool_dict['text'][self.did_to_index[h_did]],
            'img_mask': self.pool_dict['img_mask'][self.did_to_index[h_did]],
            'txt_mask': self.pool_dict['text_mask'][self.did_to_index[h_did]]
        }


        return pool, h_did


class MBEIRMainDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        query_data_path,  # Relate path to the query data
        cand_pool_path,  # Relate path to the candidate pool data
        query_instruct_path,  # Relate path to the query instructions
        img_preprocess_fn,
        mode=Mode.TRAIN,
        enable_query_instruct=True,  # Whether to enable instructions
        shuffle_cand=True,  # Whether to shuffle the candidates
        hard_neg_num=0,  # Number of negative examples in the batch
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        super().__init__(mbeir_data_dir, img_preprocess_fn)

        self._load_query_data(query_data_path)
        self._load_cand_pool_as_dict(cand_pool_path)
        self._load_query_instructions(query_instruct_path)

        self.mode = mode
        self.shuffle_cand = shuffle_cand
        self.select_cand = self._get_random_cand if self.shuffle_cand else self._get_first_cand
        self.enable_query_instruct = enable_query_instruct
        self.enable_instruct_with_text = True
        self.hard_neg_num = hard_neg_num

        returns = {} if returns is None else returns
        self.returns = {
            "hashed_qid": True,  # default value
            "task_id": True,  # default value
            "hashed_p_did": True,  # default value
            **returns,  # Overwrite defaults with any values provided in returns
        }
        if print_config:
            self.query_data_path = query_data_path
            self.cand_pool_path = cand_pool_path
            self.query_instruct_path = query_instruct_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Dataset Config---")
        print(f"Mode: {self.mode}")
        print(f"Query Data Path: {self.query_data_path}")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"Enable Query Instructions: {self.enable_query_instruct}")
        if self.enable_query_instruct:
            print(f"Query Instructions Path: {self.query_instruct_path}")
        print(f"Shuffle Candidates: {self.shuffle_cand}")
        print(f"Hard Negative Number: {self.hard_neg_num}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")

    def _load_cand_pool_as_dict(self, cand_pool_data_path):
        self._load_cand_pool(cand_pool_data_path)
        cand_pool_dict = {}
        for cand_pool_entry in self.cand_pool:
            did = cand_pool_entry.get("did")
            assert did, f"Cannot find did for {cand_pool_entry}"
            cand_pool_dict[did] = cand_pool_entry
        self.cand_pool = cand_pool_dict

    def __len__(self):
        return len(self.query_data)

    def _get_random_cand(self, cand_list):
        return random.choice(cand_list)

    def _get_first_cand(self, cand_list):
        return cand_list[0]

    def __getitem__(self, index):
        """Retrieve an item from the dataset by index."""
        mbeir_entry = self.query_data[index]

        query_txt = mbeir_entry.get("query_txt") or ""
        query_img_path = mbeir_entry.get("query_img_path", None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None

        # Randomly sample a positive example
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        assert len(pos_cand_list) > 0, f"Cannot find positive candidates for {mbeir_entry}"

        # TODO: Fix this hack for OVEN and INFOSEEK
        # We only choose the one matched with the query dataset_id due to OVEN and INFOSEEK
        if self.mode == Mode.EVAL:
            pos_cand_list = [
                pos_cand_did for pos_cand_did in pos_cand_list if pos_cand_did.split(":")[0] == query_dataset_id
            ]

        selected_pos_cand_did = self.select_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        assert pos_cand, f"Cannot find positive candidate {selected_pos_cand_did} for {mbeir_entry}"
        # Note: pos_cand_dataset_id should be the same as query_dataset_id but for OVEN and INFOSEEK it is not.
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        # Randomly sample a query prompt
        # Note:query_modality and pos_cand_modality should define the golden modalities of the current mbeir_entry task.
        # neg_cand_modality could be different from pos_cand_modality.
        query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality)
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        query_txt_without_prompt = format_string(query_txt)

        # Sample negative examples
        selected_neg_cand_list = []
        if self.mode == Mode.TRAIN:
            neg_cand_id_list = mbeir_entry.get("neg_cand_list", [])
            if self.hard_neg_num > 0:
                assert len(neg_cand_id_list) > 0, f"Cannot find negative candidates for {mbeir_entry}"
                if self.shuffle_cand:
                    random.shuffle(neg_cand_id_list)
                selected_neg_cand_id_list = []
                for i in range(self.hard_neg_num):
                    selected_neg_cand_id_list.append(
                        neg_cand_id_list[i % len(neg_cand_id_list)]
                    )  # % Wrap around from idx 0.
                for neg_cand_did in selected_neg_cand_id_list:
                    neg_cand = self.cand_pool.get(neg_cand_did, None)
                    neg_cand_txt = neg_cand.get("txt") or ""
                    neg_cand_txt = format_string(neg_cand_txt)
                    neg_cand["txt"] = neg_cand_txt
                    selected_neg_cand_list.append(neg_cand)

        def _prepare_data_dict(txt, img_path, instruct=None):
            img = self._load_and_preprocess_image(img_path)
            return {"txt": txt, "img": img, "instruct": instruct}

        query = _prepare_data_dict(
            (query_txt_with_prompt if self.enable_query_instruct else query_txt_without_prompt),
            query_img_path,
            query_prompt,
        )
        instance = {"query": query}

        if self.mode == Mode.EVAL:
            if self.returns.get("hashed_qid"):
                instance.update({"qid": hash_qid(qid)})
            if self.returns.get("task_id"):
                instance.update({"task_id": get_mbeir_task_id(query_modality, pos_cand_modality)})
            # TODO: add src_content if needed

        if self.mode == Mode.TRAIN:
            if self.returns.get("hashed_qid"):
                instance.update({"qid": hash_qid(qid)})
            if self.returns.get("hashed_p_did"):
                instance.update({"p_did": hash_did(selected_pos_cand_did)})

            pos_cand = _prepare_data_dict(
                pos_cand_txt,
                pos_cand.get("img_path", None),
            )
            instance.update({"pos_cand": pos_cand})

            neg_cand_list = [
                _prepare_data_dict(
                    neg_cand["txt"],
                    neg_cand.get("img_path", None),
                )
                for neg_cand in selected_neg_cand_list
            ]
            if len(neg_cand_list) > 0:
                instance.update({"neg_cand_list": neg_cand_list})
                
        return instance
    
class MBEIRQueryDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        query_data_path,  # Relate path to the candidate pool data
        img_preprocess_fn,
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        super().__init__(mbeir_data_dir, img_preprocess_fn)
        self._load_query_data(query_data_path)

        returns = {} if returns is None else returns
        self.returns = {
            "src_content": False,  # default value
            "hashed_qid": True,  # default value for query id
            **returns,
        }

        # Print dataset config
        if print_config:
            self.query_data_path = query_data_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Query Dataset Config---")
        print(f"Query Data Path: {self.query_data_path}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, index):
        mbeir_entry = self.query_data[index]
        query_txt = mbeir_entry.get("query_txt") or ""
        query_img_path = mbeir_entry.get("query_img_path", None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None
        img = self._load_and_preprocess_image(query_img_path)
        
        # Randomly sample a query prompt
        # Note:query_modality and pos_cand_modality should define the golden modalities of the current mbeir_entry task.
        # neg_cand_modality could be different from pos_cand_modality.
        query_txt_without_prompt = format_string(query_txt)

        instance = {
            "txt": (query_txt_without_prompt),
            "img": img,
            "modality": query_modality,
        }
        if self.returns.get("hashed_qid"):
            instance.update({"qid": hash_qid(qid)})
        if self.returns.get("src_content"):
            instance.update({"src_content": mbeir_cand_pool_entry.get("src_content", None)})
        return instance


class MBEIRCandidatePoolDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        cand_pool_data_path,  # Relate path to the candidate pool data
        img_preprocess_fn,
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        super().__init__(mbeir_data_dir, img_preprocess_fn)
        self._load_cand_pool(cand_pool_data_path)

        returns = {} if returns is None else returns
        self.returns = {
            "src_content": False,  # default value
            "hashed_did": True,  # default value for candidate id
            **returns,
        }

        # Print dataset config
        if print_config:
            self.cand_pool_path = cand_pool_data_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Candidate Pool Dataset Config---")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")

    def __len__(self):
        return len(self.cand_pool)

    def __getitem__(self, index):
        mbeir_cand_pool_entry = self.cand_pool[index]
        img_path = mbeir_cand_pool_entry.get("img_path", None)
        img = self._load_and_preprocess_image(img_path)

        did = mbeir_cand_pool_entry.get("did", None)
        dataset_id = did.split(":")[0] if did else None
        cand_txt = mbeir_cand_pool_entry.get("txt") or ""
        cand_txt = format_string(f"{cand_txt}")
        cand_modality = mbeir_cand_pool_entry.get("modality", None)

        instance = {
            "txt": cand_txt,
            "img": img,
            "modality": cand_modality,
        }
        if self.returns.get("hashed_did"):
            instance.update({"did": hash_did(did)})
        if self.returns.get("src_content"):
            instance.update({"src_content": mbeir_cand_pool_entry.get("src_content", None)})
        return instance


class MBEIRCollatorBase(object):
    @typechecked
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: Union[tuple, int]):
        """
        :param tokenizer: The tokenizer function to be used for text.
               It should take in a list of strings and return a corresponding tensor.
               Note: Pre-set properties like max_length, padding, and truncation
               should be configured before passing the tokenizer to this function.
        :param image_size: The size of the image to be used, should set in the config file.
        """
        self.tokenizer = tokenizer
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.H, self.W = image_size
        self.padded_image = torch.zeros((3, self.H, self.W))  # Note: this is a black image
        self.padded_txt = ""  # Note: this is an empty string

    def _get_padded_text_with_mask(self, txt):
        return (txt, 1) if txt not in [None, ""] else (self.padded_txt, 0)

    def _get_padded_image_with_mask(self, img):
        return (img, 1) if img is not None else (self.padded_image, 0)

    def __call__(self, batch):
        raise NotImplementedError("This method should be implemented in derived classes.")


class MBEIRMainCollator(MBEIRCollatorBase):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: tuple, mode=Mode.TRAIN, seq2seq_tokenizer=None):
        super().__init__(tokenizer, image_size)
        self.mode = mode
        self.seq2seq_tokenizer = seq2seq_tokenizer

    def __call__(self, batch):
        # Note: I group txt/image from queries and candidates together to form a single tensor.
        # Allowing for efficient GPU-based processing.

        txt_list, txt_mask_list, img_list, img_mask_list = [], [], [], []

        index_mapping = {
            "query": [[] for _ in range(len(batch))],
        }
        instance_keys = ["query"]

        # Handle EVAL mode-specific operations
        qid_list, task_id_list, instrcut_list = [], [], []
        if self.mode == Mode.EVAL:
            for instance in batch:
                qid = instance.pop("qid")
                task_id = instance.pop("task_id")
                if qid is not None:
                    qid_list.append(qid)
                if task_id is not None:
                    task_id_list.append(task_id)

        # Handle TRAIN mode-specific operations
        p_did_list = []
        if self.mode == Mode.TRAIN:
            for instance in batch:
                qid = instance.get("qid", None)
                if qid is not None:
                    qid_list.append(qid)
                p_did = instance.get("p_did", None)
                if p_did is not None:
                    p_did_list.append(p_did)

            index_mapping.update({"pos_cand": [[] for _ in range(len(batch))]})
            instance_keys.extend(["pos_cand"])

            if "neg_cand_list" in batch[0]:
                index_mapping.update({"neg_cand_list": [[] for _ in range(len(batch))]})
                instance_keys.extend(["neg_cand_list"])

        # Generate Index Mapping
        counter = 0
        inst_ids = []
        txt_ids = []

        for inst_idx, instance in enumerate(batch):
            for instance_key in instance_keys:
                items = [instance[instance_key]] if instance_key != "neg_cand_list" else instance[instance_key]  # list
                for item in items:
                    txt = item["txt"]
                    img = item["img"]
                    inst = item["instruct"]

                    if inst is not None and self.seq2seq_tokenizer is not None:
                        inst_id = self.seq2seq_tokenizer(inst, padding="max_length", max_length=75, truncation=True, add_special_tokens=False).input_ids
                        txt_id = self.seq2seq_tokenizer(txt, padding="max_length", max_length=75, truncation=True, add_special_tokens=False).input_ids
                    elif self.seq2seq_tokenizer is not None:
                        inst_id = self.seq2seq_tokenizer(" ", padding="max_length", max_length=75, truncation=True, add_special_tokens=False).input_ids
                        txt_id = self.seq2seq_tokenizer(" ", padding="max_length", max_length=75, truncation=True, add_special_tokens=False).input_ids
                    else:
                        inst_id = None
                        txt_id = None

                    index_mapping[instance_key][inst_idx].append(counter)  # Track current index
                    counter += 1
                    padded_txt, txt_mask = self._get_padded_text_with_mask(txt)
                    padded_img, img_mask = self._get_padded_image_with_mask(img)
                    txt_list.append(padded_txt)
                    img_list.append(padded_img)
                    txt_mask_list.append(txt_mask)
                    img_mask_list.append(img_mask)
                    inst_ids.append(inst_id)
                    txt_ids.append(txt_id)


        processed_batch = {
            "txt_batched": self.tokenizer(txt_list),
            "image_batched": torch.stack(img_list, dim=0),
            "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
            "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
            "index_mapping": index_mapping,
            "inst_ids": inst_ids,
            "txt_ids": txt_ids
        }

        if self.mode == Mode.EVAL:
            if qid_list:
                processed_batch.update({"qid_list": qid_list})
            if task_id_list:
                processed_batch.update({"task_id_list": task_id_list})

        if self.mode == Mode.TRAIN:
            if qid_list:
                processed_batch.update({"qid_list": torch.tensor(qid_list)})
            if p_did_list:
                processed_batch.update({"p_did_list": torch.tensor(p_did_list)})

        # TODO: Fix this hack for BLIP tokenizer.
        if hasattr(processed_batch["txt_batched"], "input_ids"):
            bs = processed_batch["txt_batched"]["input_ids"].size(0)
        else:
            bs = len(processed_batch["txt_batched"])
        assert bs == processed_batch["image_batched"].size(0)
        assert bs == processed_batch["txt_mask_batched"].size(0)
        assert bs == processed_batch["image_mask_batched"].size(0)
        return processed_batch


class MBEIRQueryPoolCollator(MBEIRCollatorBase):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: tuple, seq2seq_tokenizer=None):
        super().__init__(tokenizer, image_size)
        self.seq2seq_tokenizer = seq2seq_tokenizer

    def __call__(self, batch):
        # Note: I group txt/image from queries and candidates together to form a single tensor.
        # Allowing for efficient GPU-based processing.

        txt_list, txt_mask_list, img_list, img_mask_list = [], [], [], []

        index_mapping = {
            "query": [[] for _ in range(len(batch))],
        }
        instance_keys = ["query"]

        qid_list, task_id_list, instrcut_list = [], [], []
        # Handle TRAIN mode-specific operations
        for instance in batch:
            qid = instance.get("qid", None)
            if qid is not None:
                qid_list.append(qid)

        # Generate Index Mapping
        counter = 0
        inst_ids = []
        txt_ids = []

        for inst_idx, instance in enumerate(batch):
            for instance_key in instance_keys:
                items = [instance[instance_key]] if instance_key != "neg_cand_list" else instance[instance_key]  # list
                for item in items:
                    txt = item["txt"]
                    img = item["img"]
                    inst = item["instruct"]

                    if inst is not None and self.seq2seq_tokenizer is not None:
                        inst_id = self.seq2seq_tokenizer(inst, padding="max_length", max_length=75, truncation=True, add_special_tokens=False).input_ids
                        txt_id = self.seq2seq_tokenizer(txt, padding="max_length", max_length=75, truncation=True, add_special_tokens=False).input_ids
                    elif self.seq2seq_tokenizer is not None:
                        inst_id = self.seq2seq_tokenizer(" ", padding="max_length", max_length=75, truncation=True, add_special_tokens=False).input_ids
                        txt_id = self.seq2seq_tokenizer(" ", padding="max_length", max_length=75, truncation=True, add_special_tokens=False).input_ids
                    else:
                        inst_id = None
                        txt_id = None

                    index_mapping[instance_key][inst_idx].append(counter)  # Track current index
                    counter += 1
                    padded_txt, txt_mask = self._get_padded_text_with_mask(txt)
                    padded_img, img_mask = self._get_padded_image_with_mask(img)
                    txt_list.append(padded_txt)
                    img_list.append(padded_img)
                    txt_mask_list.append(txt_mask)
                    img_mask_list.append(img_mask)
                    inst_ids.append(inst_id)
                    txt_ids.append(txt_id)


        processed_batch = {
            "txt_batched": self.tokenizer(txt_list),
            "image_batched": torch.stack(img_list, dim=0),
            "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
            "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
            "index_mapping": index_mapping,
            "inst_ids": inst_ids,
            "txt_ids": txt_ids
        }

        if qid_list:
            processed_batch.update({"qid_list": torch.tensor(qid_list)})

        # TODO: Fix this hack for BLIP tokenizer.
        if hasattr(processed_batch["txt_batched"], "input_ids"):
            bs = processed_batch["txt_batched"]["input_ids"].size(0)
        else:
            bs = len(processed_batch["txt_batched"])
        assert bs == processed_batch["image_batched"].size(0)
        assert bs == processed_batch["txt_mask_batched"].size(0)
        assert bs == processed_batch["image_mask_batched"].size(0)
        return processed_batch


class MBEIRCandidatePoolCollator(MBEIRCollatorBase):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: tuple):
        super().__init__(tokenizer, image_size)

    def __call__(self, batch):
        txt_list, txt_mask_list, img_list, img_mask_list, did_list = [], [], [], [], []
        # Candidate can be indexed directly from the batch
        for instance in batch:
            txt = instance["txt"]
            img = instance["img"]
            padded_txt, txt_mask = self._get_padded_text_with_mask(txt)
            padded_img, img_mask = self._get_padded_image_with_mask(img)
            txt_list.append(padded_txt)
            img_list.append(padded_img)
            txt_mask_list.append(txt_mask)
            img_mask_list.append(img_mask)

            did = instance.get("did", None)
            if did is not None:
                did_list.append(did)

        processed_batch = {
            "txt_batched": self.tokenizer(txt_list),
            "image_batched": torch.stack(img_list, dim=0),
            "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
            "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
        }

        if did_list:
            processed_batch.update({"did_list": did_list})

        if hasattr(processed_batch["txt_batched"], "input_ids"):
            bs = processed_batch["txt_batched"]["input_ids"].size(0)
        else:
            bs = len(processed_batch["txt_batched"])
        assert bs == processed_batch["image_batched"].size(0)
        assert bs == processed_batch["txt_mask_batched"].size(0)
        assert bs == processed_batch["image_mask_batched"].size(0)
        return processed_batch
