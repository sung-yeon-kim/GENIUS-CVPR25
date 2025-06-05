# #!/bin/bash

# set -e  # Exit immediately if a command exits with a non-zero status

# Path to MBEIR data and GENIUS directory where we store the checkpoints, embeddings, etc.
genir_dir="/GENIUS" # <--- Change this to the GENIUS directory
SRC="$genir_dir/src"  # Absolute path to codebse /GENIUS/src # <--- Change this to the path of your GENIUS/src
MBEIR_DATA_DIR="$genir_dir/M-BEIR" # <--- Change this to the MBEIR data directory you download from HF page

# Path to common dir
COMMON_DIR="$SRC/common"

# Path to config dir
MODEL="residual_quantization"  # <--- Change this to the model you want to run
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="eval"  # <--- Change this to the mode you want to run
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs_scripts/$SIZE/$MODE/$EXP_NAME"

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3  # <--- Change this to the CUDA devices you want to use
NPROC=4 # <--- Change this to the number of GPUs you want to use
# export CUDA_VISIBLE_DEVICES=0  # <--- Change this to the CUDA devices you want to use
# NPROC=1 # <--- Change this to the number of GPUs you want to use
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo  "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# CD to script directory
cd $COMMON_DIR

# Run Embedding command
CONFIG_PATH="$CONFIG_DIR/embed.yaml"
SCRIPT_NAME="mbeir_embedder.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

python -m torch.distributed.run --nproc_per_node=$NPROC $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$genir_dir" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

# # Run Index command
CONFIG_PATH="$CONFIG_DIR/index.yaml"
SCRIPT_NAME="mbeir_embeding_retriever.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

python $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$genir_dir" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --enable_create_index

# Run retrieval command
CONFIG_PATH="$CONFIG_DIR/retrieval.yaml"
SCRIPT_NAME="mbeir_embeding_retriever.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

python $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$genir_dir" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --enable_retrieval