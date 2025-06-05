"""
flickr30k_data_preprocessor.py

Module description:
    1. Convert all image files to the JPG format and resize the smaller dimension to 256.
    2. Generate candidate pool.
    3. Convert the dataset to the MBEIR format.
"""

import os
import json
import argparse
from multiprocessing import cpu_count

from utils import (
    resize_and_convert_image_to_jpg,
    is_valid_image,
    get_dataset_id,
    format_string,
    count_entries_in_file,
    generate_mbeir_format_doc_key,
    load_mbeir_format_pool_file_as_dict,
    print_mbeir_format_cand_pool_stats,
    save_list_as_jsonl,
    load_jsonl_as_list,
    print_mbeir_format_dataset_stats,
    parallel_process_image_directory,
    aggregate_candidates_for_mbeir_format_dataset,
)

FLICKR30K_QUERY_MODALITY_IMAGE = "image"
FLICKR30K_QUERY_MODALITY_TEXT = "text"
FLICKR30K_CANDIDATE_MODALITY_IMAGE = "image"
FLICKR30K_CANDIDATE_MODALITY_TEXT = "text"
FLICKR30K_DATASET_ID = get_dataset_id("FLICKR30K")
assert FLICKR30K_DATASET_ID is not None, "Unknown dataset name!"


def flickr30k_to_mbeir_entry(
    flickr30k_entry,
    candidate_pool,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Convert Flickr30k data format to MBEIR format.
    """
    mbeir_entries = []
    img_filename = flickr30k_entry["filename"]
    img_path = os.path.join("mbeir_images", "flickr30k", "Images", img_filename)

    if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
        print(f"Warning: Invalid image: {img_path}")  # if the image is invalid, skip it
        return None

    # Each image has a list of sentences
    captions = [sentence["raw"] for sentence in flickr30k_entry["sentences"]]

    # Generate image to text MBEIR entry
    mbeir_entry_img2txt = {
        "qid": None,
        "query_txt": None,
        "query_img_path": img_path,
        "query_modality": FLICKR30K_QUERY_MODALITY_IMAGE,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    for caption in captions[:5]:  # Only use the first 5 captions
        txt = format_string(caption)
        if not txt:
            print(f"Warning: Empty caption: {flickr30k_entry}")
            continue

        # Add positive candidate to img2txt entry
        _img2txt_candidate = {
            "txt": txt,
            "modality": FLICKR30K_CANDIDATE_MODALITY_TEXT,
        }
        doc_key = generate_mbeir_format_doc_key(_img2txt_candidate)
        img2txt_candidate = candidate_pool.get(doc_key, None)
        if not img2txt_candidate:
            print(f"Cannot find candidate for {doc_key}")
            continue  # Skip if candidate not found
        mbeir_entry_img2txt["pos_cand_list"].append(img2txt_candidate["did"])  # Store the document ID

        # Generate text to image MBEIR entry
        mbeir_entry_txt2img = {
            "qid": None,
            "query_txt": txt,
            "query_img_path": None,
            "query_modality": FLICKR30K_QUERY_MODALITY_TEXT,
            "query_src_content": None,
            "pos_cand_list": [],
            "neg_cand_list": [],
        }

        # Add positive candidates to txt2img entry
        _txt2img_candidate = {
            "img_path": img_path,
            "modality": FLICKR30K_CANDIDATE_MODALITY_IMAGE,
        }
        doc_key = generate_mbeir_format_doc_key(_txt2img_candidate)
        txt2img_candidate = candidate_pool.get(doc_key, None)
        if not txt2img_candidate:
            print(f"Cannot find candidate for {doc_key}")
            continue  # Skip if candidate not found
        mbeir_entry_txt2img["pos_cand_list"].append(txt2img_candidate["did"])

        mbeir_entries.append(mbeir_entry_txt2img)
    mbeir_entries.append(mbeir_entry_img2txt)
    if not mbeir_entries:
        print(f"Cannot find positive image facts for {captions}")
    return mbeir_entries


def flickr30k_to_mbeir(flickr30k_data, candidate_pool_file_path, mbeir_data_dir, include_src_content=True):
    """
    Flickr30k dataset to MBEIR format.
    """
    mbeir_entries_merged = []

    # Load candidate pool
    cand_pool_dict = load_mbeir_format_pool_file_as_dict(candidate_pool_file_path, doc_key_to_content=True)

    for flickr30k_entry in flickr30k_data["images"]:
        mbeir_entries = flickr30k_to_mbeir_entry(
            flickr30k_entry,
            cand_pool_dict,
            mbeir_data_dir,
            include_src_content,
        )
        if mbeir_entries:  # Skip invalid entries
            mbeir_entries_merged.extend(mbeir_entries)
    return mbeir_entries_merged


def generate_flickr30k_candidate_pool(
    flickr30k_dir,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Generate Flickr30k candidate pool in MBEIR format and save it to jsonl files.
    """
    flickr30k_data_file = os.path.join(flickr30k_dir, "dataset_flickr30k.json")
    assert os.path.exists(flickr30k_data_file), f"{flickr30k_data_file} does not exist."

    document_id = 1  # Start from 1 for document IDs
    seen_txts_all = {}  # To store descriptions we've already seen
    seen_image_paths_all = {}  # To store image paths we've already seen

    # Dictionaries to store seen texts and images for each split
    seen_txts_split = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    seen_image_paths_split = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    # Open files for candidate pools
    all_cand_pool_path = os.path.join(flickr30k_dir, "mbeir_flickr30k_cand_pool.jsonl")
    task0_cand_pool_files = {}  # For images only
    task3_cand_pool_files = {}  # For texts only

    # For val and test splits, open task0 and task3 candidate pool files
    for split in ["val", "test"]:
        task0_cand_pool_files[split] = open(
            os.path.join(flickr30k_dir, f"mbeir_flickr30k_task0_{split}_cand_pool.jsonl"), "w"
        )
        task3_cand_pool_files[split] = open(
            os.path.join(flickr30k_dir, f"mbeir_flickr30k_task3_{split}_cand_pool.jsonl"), "w"
        )

    # For train split, open a single candidate pool file
    train_cand_pool_file_path = os.path.join(flickr30k_dir, "mbeir_flickr30k_train_cand_pool.jsonl")
    train_cand_pool_file = open(train_cand_pool_file_path, "w")

    with open(all_cand_pool_path, "w") as allfile:
        with open(flickr30k_data_file, "r") as source:
            flickr30k_data = json.load(source)

            for flickr30k_entry in flickr30k_data["images"]:
                split = flickr30k_entry["split"]  # "train", "val", or "test"
                img_filename = flickr30k_entry["filename"]
                img_path = os.path.join("mbeir_images", "flickr30k", "Images", img_filename)
                full_img_path = os.path.join(mbeir_data_dir, img_path)

                if not is_valid_image(full_img_path):
                    print(f"Warning: Invalid image: {img_path}")
                else:
                    candidate_pool_entry_img = {
                        "txt": None,
                        "img_path": img_path,
                        "modality": FLICKR30K_CANDIDATE_MODALITY_IMAGE,
                        "did": f"{FLICKR30K_DATASET_ID}:{document_id}",
                        "src_content": None,
                    }
                    # If image path hasn't been seen, create image entry
                    if img_path not in seen_image_paths_all:
                        allfile.write(json.dumps(candidate_pool_entry_img) + "\n")
                        seen_image_paths_all[img_path] = candidate_pool_entry_img
                        document_id += 1  # Increment for next entry
                    else:
                        candidate_pool_entry_img = seen_image_paths_all[img_path]

                    if split in ["val", "test"]:
                        if img_path not in seen_image_paths_split[split]:
                            task0_cand_pool_files[split].write(json.dumps(candidate_pool_entry_img) + "\n")
                            seen_image_paths_split[split].add(img_path)
                    elif split == "train":
                        if img_path not in seen_image_paths_split[split]:
                            train_cand_pool_file.write(json.dumps(candidate_pool_entry_img) + "\n")
                            seen_image_paths_split[split].add(img_path)

                captions = [sentence["raw"] for sentence in flickr30k_entry["sentences"]]
                for caption in captions[:5]:  # Only use the first 5 captions
                    txt = format_string(caption)
                    if not txt:
                        print(f"Warning: Empty caption: {flickr30k_entry}")  # Skip if caption is empty
                        continue

                    candidate_pool_entry_txt = {
                        "txt": txt,
                        "img_path": None,
                        "modality": FLICKR30K_CANDIDATE_MODALITY_TEXT,
                        "did": f"{FLICKR30K_DATASET_ID}:{document_id}",
                        "src_content": None,
                    }
                    # If description hasn't been seen, create text entry
                    if txt not in seen_txts_all:
                        allfile.write(json.dumps(candidate_pool_entry_txt) + "\n")
                        seen_txts_all[txt] = candidate_pool_entry_txt
                        document_id += 1  # Increment for next entry
                    else:
                        candidate_pool_entry_txt = seen_txts_all[txt]

                    if split in ["val", "test"]:
                        if txt not in seen_txts_split[split]:
                            task3_cand_pool_files[split].write(json.dumps(candidate_pool_entry_txt) + "\n")
                            seen_txts_split[split].add(txt)
                    elif split == "train":
                        if txt not in seen_txts_split[split]:
                            train_cand_pool_file.write(json.dumps(candidate_pool_entry_txt) + "\n")
                            seen_txts_split[split].add(txt)

    # Close the train candidate pool file
    train_cand_pool_file.close()

    # Close files for val and test splits
    for split in ["val", "test"]:
        task0_cand_pool_files[split].close()
        task3_cand_pool_files[split].close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Format Flickr30k images and refactor dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--flickr30k_images_dir",
        type=str,
        default="mbeir_images/flickr30k/Images/",
        help="Relative directory path to save Flickr30k images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--flickr30k_dir",
        type=str,
        default="src_data/flickr30k",
        help="Relative directory path of Flickr30k files folder under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="1. Filter out corrupt images. 2. Resize images. 3. Convert images to JPEG format.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating Flickr30k candidate pool in MBEIR format.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting Flickr30k data to MBEIR format.",
    )
    parser.add_argument(
        "--trim_train_data",
        action="store_true",
        help="Trim the training data queries.",
    )
    parser.add_argument(
        "--enable_training_candidate_pool",
        action="store_true",
        help="Enable generating training candidate pool in MBEIR format.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and unzip Flickr30k image files and the JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Construct full paths
    flickr30k_dir = os.path.join(args.mbeir_data_dir, args.flickr30k_dir)
    flickr30k_images_dir = os.path.join(args.mbeir_data_dir, args.flickr30k_images_dir)
    flickr30k_candidate_pool_path = os.path.join(flickr30k_dir, "mbeir_flickr30k_cand_pool.jsonl")

    if args.enable_image_processing:
        print(f"Processing images in {flickr30k_images_dir}...")
        parallel_process_image_directory(flickr30k_images_dir, num_processes=cpu_count())

    # Generate candidate pool
    if args.enable_candidate_pool:
        print("Generating Flickr30k candidate pool in MBEIR format...")
        generate_flickr30k_candidate_pool(
            flickr30k_dir,
            args.mbeir_data_dir,
        )
        print(f"Candidate pool saved to {flickr30k_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(flickr30k_candidate_pool_path)
        for split in ["val", "test"]:
            task0_cand_pool_path = os.path.join(flickr30k_dir, f"mbeir_flickr30k_task0_{split}_cand_pool.jsonl")
            task3_cand_pool_path = os.path.join(flickr30k_dir, f"mbeir_flickr30k_task3_{split}_cand_pool.jsonl")
            print(f"Task0 (image) candidate pool for {split} saved to {task0_cand_pool_path}")
            print_mbeir_format_cand_pool_stats(task0_cand_pool_path)
            print(f"Task3 (text) candidate pool for {split} saved to {task3_cand_pool_path}")
            print_mbeir_format_cand_pool_stats(task3_cand_pool_path)
        # Print statistics for the train candidate pool
        train_cand_pool_path = os.path.join(flickr30k_dir, "mbeir_flickr30k_train_cand_pool.jsonl")
        print(f"Train candidate pool saved to {train_cand_pool_path}")
        print_mbeir_format_cand_pool_stats(train_cand_pool_path)

    # Convert Flickr30k data to MBEIR format
    if args.enable_mbeir_conversion:
        print("Converting Flickr30k data to MBEIR format...")

        data_splits = ["train", "val", "test"]
        flickr30k_data_file = os.path.join(flickr30k_dir, "dataset_flickr30k.json")
        assert os.path.exists(flickr30k_data_file), f"{flickr30k_data_file} does not exist."
        with open(flickr30k_data_file, "r") as source:
            flickr30k_data = json.load(source)

        for split in data_splits:
            flickr30k_data_split = {
                "images": [entry for entry in flickr30k_data["images"] if entry["split"] == split]
            }

            if split == "train":
                candidate_pool_file_path = os.path.join(flickr30k_dir, "mbeir_flickr30k_train_cand_pool.jsonl")
            else:
                candidate_pool_file_path = flickr30k_candidate_pool_path  # Use the full candidate pool

            mbeir_entries = flickr30k_to_mbeir(
                flickr30k_data_split,
                candidate_pool_file_path,
                args.mbeir_data_dir,
            )

            # Aggregate data
            mbeir_entries = aggregate_candidates_for_mbeir_format_dataset(mbeir_entries, print_duplicate=False)

            # Generate query IDs
            for i, entry in enumerate(mbeir_entries):
                entry.update({"qid": f"{FLICKR30K_DATASET_ID}:{i + 1}"})

            if split == "train":
                mbeir_format_flickr30k_train_file_path = os.path.join(flickr30k_dir, f"mbeir_flickr30k_train.jsonl")
                save_list_as_jsonl(mbeir_entries, mbeir_format_flickr30k_train_file_path, mode="w")
                # Print statistics
                total_entries, _data = count_entries_in_file(mbeir_format_flickr30k_train_file_path)
                print(f"MBEIR format Flickr30k train data saved to {mbeir_format_flickr30k_train_file_path}")
                print(f"Total number of entries in {mbeir_format_flickr30k_train_file_path}: {total_entries}")
                flickr30k_cand_pool = load_mbeir_format_pool_file_as_dict(
                    candidate_pool_file_path, doc_key_to_content=True, key_type="did"
                )
                print_mbeir_format_dataset_stats(_data, flickr30k_cand_pool)
            else:
                # For val and test splits, separate into task0 and task3
                mbeir_entries_task0 = []
                mbeir_entries_task3 = []
                for entry in mbeir_entries:
                    if entry["query_modality"] == "text":
                        mbeir_entries_task0.append(entry)
                    elif entry["query_modality"] == "image":
                        mbeir_entries_task3.append(entry)
                # Save task0 (text queries)
                mbeir_format_task0_file_path = os.path.join(flickr30k_dir, f"mbeir_flickr30k_task0_{split}.jsonl")
                save_list_as_jsonl(mbeir_entries_task0, mbeir_format_task0_file_path, mode="w")
                # Save task3 (image queries)
                mbeir_format_task3_file_path = os.path.join(flickr30k_dir, f"mbeir_flickr30k_task3_{split}.jsonl")
                save_list_as_jsonl(mbeir_entries_task3, mbeir_format_task3_file_path, mode="w")
                print(f"Saved {split} text queries to {mbeir_format_task0_file_path}")
                print(f"Saved {split} image queries to {mbeir_format_task3_file_path}")

                # Print statistics
                task0_cand_pool_path = os.path.join(flickr30k_dir, f"mbeir_flickr30k_task0_{split}_cand_pool.jsonl")
                task3_cand_pool_path = os.path.join(flickr30k_dir, f"mbeir_flickr30k_task3_{split}_cand_pool.jsonl")
                if mbeir_entries_task0:
                    total_entries, _data = count_entries_in_file(mbeir_format_task0_file_path)
                    print(f"Total number of entries in {mbeir_format_task0_file_path}: {total_entries}")
                    flickr30k_cand_pool = load_mbeir_format_pool_file_as_dict(
                        task0_cand_pool_path, doc_key_to_content=True, key_type="did"
                    )
                    print_mbeir_format_dataset_stats(_data, flickr30k_cand_pool)
                if mbeir_entries_task3:
                    total_entries, _data = count_entries_in_file(mbeir_format_task3_file_path)
                    print(f"Total number of entries in {mbeir_format_task3_file_path}: {total_entries}")
                    flickr30k_cand_pool = load_mbeir_format_pool_file_as_dict(
                        task3_cand_pool_path, doc_key_to_content=True, key_type="did"
                    )
                    print_mbeir_format_dataset_stats(_data, flickr30k_cand_pool)


if __name__ == "__main__":
    main()
