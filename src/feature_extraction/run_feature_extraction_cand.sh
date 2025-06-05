# Feature Extraction CLIP model on MBEIR dataset

# Path to the codebase and config file
genir_dir="/GENIUS" # <--- Change this to the GENIUS directory
SRC="$genir_dir/src"  # Absolute path to codebse /GENIUS/src # <--- Change this to the path of your GENIUS/src

# Path to common dir
COMMON_DIR="$SRC/common"

# Path to MBEIR data and GENIUS directory where we store the checkpoints, embeddings, etc.
MBEIR_DATA_DIR="$genir_dir/M-BEIR" # <--- Change this to the MBEIR data directory you download from HF page

# Path to config dir
MODEL="rq"  # <--- Change this to the model you want to run
MODEL_DIR="$SRC/feature_extraction"
CONFIG_DIR="$MODEL_DIR"

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3 # <--- Change this to the CUDA devices you want to us
NPROC=4
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Update config
CONFIG_PATH="$CONFIG_DIR/config_cand.yaml"
cd $COMMON_DIR
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

# Change to model directory
cd $MODEL_DIR
SCRIPT_NAME="clip_feature_extraction_cand.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

# Run training command
python3 -m torch.distributed.run --nproc_per_node=$NPROC $MODEL_DIR/$SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$genir_dir" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"