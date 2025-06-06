# GENIUS

This repo contains the codebase for the CVPR 2025 paper "[GENIUS: A Generative Framework for Universal Multimodal Search](https://arxiv.org/pdf/2503.19868)"


<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/pdf/2503.19868" target="_blank" style="margin: 2px;">
    <img alt="arXiv" src="https://img.shields.io/badge/📄%20arXiv-2503.19868-b31b1b?color=b31b1b&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://sung-yeon-kim.github.io/project_pages/GENIUS/index.html" target="_blank" style="margin: 2px;">
    <img alt="Project Page" src="https://img.shields.io/badge/🌐%20Project%20Page-GENIUS-ff6b6b?color=ff6b6b&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/sung-yeon-kim/GENIUS" target="_blank" style="margin: 2px;">
    <img alt="GitHub" src="https://img.shields.io/badge/💻%20GitHub-GENIUS-2ea44f?color=2ea44f&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/Sungyeon/GENIUS" target="_blank" style="margin: 2px;">
    <img alt="HuggingFace" src="https://img.shields.io/badge/🤗%20Checkpoint-GENIUS-ffd700?color=ffd700&logoColor=black" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="LICENSE" target="_blank" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/📜%20License-MIT-4b0082?color=4b0082&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction

We propose **GENIUS**, a **universal generative retrieval framework** that supports diverse tasks across multiple modalities. By learning discrete, modality‐decoupled IDs via **semantic quantization**, GENIUS encodes multimodal data into compact identifiers and performs constant‐time retrieval with competitive accuracy. 

<p align="center">
  <img src="misc/cvpr_25_genius.png" alt="GENIUS Overview" width="55%">
</p>

### ✨ Key Advantages

- **Universal Retrieval**  
  Single model handles various retrieval tasks including image‐to‐image, text‐to‐text, image‐to‐text, text‐to‐image, and their combinations.

- **Fast Retrieval**  
  Constant‐time lookup via discrete ID matching, independent of candidate pool size.

- **Competitive Accuracy**  
  Comparable to—and sometimes better than—embedding‐based methods, while significatnly reducing inference cost.

## Overview

GENIUS consists of three key components that work together in a three-stage training pipeline:

1. **Multimodal Encoder (CLIP-SF)**  
   Extracts joint image/text features using a shared backbone. We leverage **UniIR's score‐fusion CLIP model** to learn cross‐modal relations without extra pretraining, with pretrained checkpoints available on [Hugging Face](https://huggingface.co/TIGER-Lab/UniIR/blob/main/checkpoint/CLIP_SF/clip_sf_large.pth).

2. **Modality-Decoupled Quantizer**  
   Compresses continuous embeddings into discrete, layered ID codes including modality and semantic information. Through **residual quantization training**, it learns to encode both images and text into layered, discrete IDs:  
   - First code: **modality indicator** (0 = image, 1 = text, 2 = image‐text)  
   - Subsequent codes: **semantic features** (objects → attributes → context)

<p align="center">
  <img src="misc/semantic_quantization.png" alt="Semantic Quantization" width="95%">
</p>

3. **ID Generator**  
   **Sequence‐to‐sequence decoder** that generates target IDs based on query embeddings. During training, it learns to predict discrete IDs from various input queries (images, text, or pairs with instruction). We employ **query augmentation** (query-target mixing) to improve generalization and incorporate **constrained beam search** during decoding to enforce valid ID sequences.

---

## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/sung-yeon-kim/GENIUS.git
cd GENIUS
conda env create -f genius_env.yml
```


## Usage

### Download Pretrained CLIP-SF (Stage 0)

We utilize UniIR's score fusion model as a replacement for the encoder pretraining stage.

```bash
mkdir -p checkpoint/CLIP_SF
wget https://huggingface.co/TIGER-Lab/UniIR/resolve/main/checkpoint/CLIP_SF/clip_sf_large.pth -O checkpoint/CLIP_SF/clip_sf_large.pth
```


### Feature Extraction (Preprocessing)


#### - Training Data

Extracts CLIP features for training set → `extracted_embed/CLIP_SF/train`.

```bash
# Navigate to feature extraction directory
cd feature_extraction

# Run feature extraction for training data
bash run_feature_extraction_train.sh
```

#### - Candidate Pool

Extracts CLIP features for the retrieval candidate pool → `extracted_embed/CLIP_SF/cand`.

```bash
# Run feature extraction for candidate pool
bash run_feature_extraction_cand.sh
```

### Residual Quantization (Stage 1)

```bash
cd models/residual_quantization
vim configs_scripts/large/train/inbatch/inbatch.yaml  # Edit config like data path, batch size, etc.
bash configs_scripts/large/train/inbatch/run_inbatch.sh
```

### Generator Training (Stage 2)

```bash
cd models/generative_retriever
vim configs_scripts/large/train/inbatch/inbatch.yaml  # Edit config like data path, batch size, etc.
bash configs_scripts/large/train/inbatch/run_inbatch.sh
```

### Inference

1. Extract CLIP features for candidate pool (if not already done):
```bash
cd feature_extraction
bash run_feature_extraction_cand.sh
```

2. Compile trie_cpp (recommended for faster inference):
```bash
cd models/generative_retriever/trie_cpp
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    trie_cpp.cpp -o trie_cpp$(python3-config --extension-suffix)
```

3. Run evaluation:
```bash
cd models/generative_retriever
bash configs_scripts/large/eval/inbatch/run_eval.sh
```
> For inference, you can choose between three trie implementations: `trie_cpp` (fastest), `trie` (Python), `marisa` (alternative).


## Model Checkpoints

We provide GENIUS model checkpoints in the 🤗 [Hugging Face repository](https://huggingface.co/Sungyeon/GENIUS):

### Stage 1: Residual Quantization Model
- **Model**: [`rq_clip_large.pth`](https://huggingface.co/Sungyeon/GENIUS/blob/main/checkpoint/rq_clip_large.pth)
- **Description**: Learns to encode multimodal data into discrete IDs through residual quantization
- **Size**: ~1.2GB

### Stage 2: Generator Model
- **Model**: [`GENIUS_t5small.pth`](https://huggingface.co/Sungyeon/GENIUS/blob/main/checkpoint/GENIUS_t5small.pth)
- **Description**: T5-based sequence-to-sequence model that generates target IDs for retrieval
- **Size**: ~500MB

### Stage 0: CLIP-SF Model
- **Model**: [`clip_sf_large.pth`](https://huggingface.co/TIGER-Lab/UniIR/blob/main/checkpoint/CLIP_SF/clip_sf_large.pth)
- **Source**: [TIGER-Lab/UniIR](https://huggingface.co/TIGER-Lab/UniIR)
- **Description**: Score-fusion CLIP model for multimodal feature extraction

```bash
# Clone the repository
git clone https://huggingface.co/Sungyeon/GENIUS

# Download CLIP-SF model
wget https://huggingface.co/TIGER-Lab/UniIR/resolve/main/checkpoint/CLIP_SF/clip_sf_large.pth -O checkpoint/CLIP_SF/clip_sf_large.pth
```

> Note: All three models are required for full functionality. The CLIP-SF model is used for feature extraction, the Residual Quantization model for ID encoding, and the Generator model for retrieval.

## 📈 Performance

> The results in parentheses denote scores from our reimplemented checkpoints, as the originals were lost during server migration. While close to the paper, slight variations may occur due to retraining randomness.


### Universal Information Retrieval

| Task | Dataset | CLIP_SF | BLIP_FF | GENIUS (checkpoint) | GENIUSᴿ (checkpoint) |
|:-----|:--------|:-------:|:-------:|:------:|:-------:|
| **T→I** | VisualNews | 42.6 | 23.0 | 18.5 (18.5) | 27.3  (27.3)|
| | MSCOCO | 77.9 | 75.6 | 55.1 (55.3) | 68.0 (68.0) |
| | Fashion200K | 17.8 | 25.4 | 13.7 (14.0) | 16.2 (15.9)|
| **T→T** | WebQA | 84.7 | 79.5 | 31.1 (31.9) | 42.9 (43.6)|
| **T→(I,T)** | EDIS | 59.4 | 50.3 | 36.6 (37.0) | 44.1 (44.1)|
| | WebQA | 78.8 | 79.7 | 49.0 (49.0) | 59.7 (59.3) |
| **I→T** | VisualNews | 42.8 | 21.1 | 18.4 (18.2) | 26.8 (26.8)|
| | MSCOCO | 92.3 | 88.8 | 82.7 (83.0) | 90.6 (90.7) |
| | Fashion200K | 17.9 | 27.6 | 12.8 (12.9) | 16.2 (16.6) |
| **I→I** | NIGHTS | 33.4 | 33.0 | 8.1 (8.1) | 30.2 (30.0) |
| | OVEN | 39.2 | 34.7 | 34.6 (34.5) | 38.0 (38.0) |
| **(I,T)→T** | InfoSeek | 24.0 | 19.7 | 10.4 (10.5) | 18.0 (18.0) |
| **(I,T)→I** | FashionIQ | 26.2 | 28.5 | 13.1 (13.1) | 19.2 (19.3) |
| | CIRR | 43.0 | 51.4 | 20.1 (20.1) | 38.3 (38.1)|
| **(I,T)→(I,T)** | OVEN | 60.2 | 57.8 | 36.5 (36.6) | 48.6 (48.3)|
| | InfoSeek | 44.6 | 27.7 | 14.2 (14.3) | 28.6 (28.7) |

### ⚡ Efficiency

When the candidate pool grows, embedding‐based retrieval (e.g., CLIP + nearest neighbors) slows down dramatically. In contrast, GENIUS's discrete ID generation is **nearly constant time**. Empirically, GENIUS is roughly **4× faster** than competing generative methods like GRACE.

<p align="center"><img src="misc/efficiency.png" alt="cvpr25_genius" width="50%"></p>



## Citation

If you find this work useful, please cite:

```bibtex
@article{kim2024genius,
  title={GENIUS: A Generative Framework for Universal Multimodal Search},
  author={Kim, Sungyeon and Zhu, Xinliang and Lin, Xiaofan and Bastan, Muhammet and Gray, Douglas and Kwak, Suha},
  journal={arXiv preprint arXiv:2503.19868},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments & References

### Codebases
Our implementation is built upon and modified from these great repositories:
- [UniIR](https://github.com/TIGER-AI-Lab/UniIR) - Base framework for multimodal retrieval
- [GENRE](https://github.com/facebookresearch/GENRE) - Trie structure implementation
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) - Vector quantization implementation
- [CLIP4CIR](https://github.com/ABaldrati/CLIP4Cir) - Combining module that integrates image and text features

### Related Papers
- [UniIR: Training and Benchmarking Universal Multimodal Information Retrievers](https://arxiv.org/pdf/2311.17136)
- [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065)
- [GRACE: Generative Cross-Modal Retrieval](https://arxiv.org/pdf/2402.10805)
- [IRGen: Generative Modeling for Image Retrieval](https://arxiv.org/pdf/2303.10126)