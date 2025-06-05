#!/bin/bash

# Script to train Residual Quantization model on MBEIR dataset
# This script handles distributed training setup and configuration updates

# Exit on error
set -e

# Default configuration
DEFAULT_GENIR_DIR="/GenIR"
DEFAULT_CUDA_DEVICES="0,1,2,3"
DEFAULT_NPROC=4
DEFAULT_MASTER_PORT=3131

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --genir_dir)
            GENIR_DIR="$2"
            shift 2
            ;;
        --cuda_devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default values if not provided
GENIR_DIR=${GENIR_DIR:-$DEFAULT_GENIR_DIR}
CUDA_DEVICES=${CUDA_DEVICES:-$DEFAULT_CUDA_DEVICES}
NPROC=${NPROC:-$DEFAULT_NPROC}
MASTER_PORT=${MASTER_PORT:-$DEFAULT_MASTER_PORT}

# Validate required directories
if [ ! -d "$GENIR_DIR" ]; then
    echo "Error: GENIR_DIR ($GENIR_DIR) does not exist"
    exit 1
fi

# Set up paths
SRC="$GENIR_DIR/src"
COMMON_DIR="$SRC/common"
MBEIR_DATA_DIR="$GENIR_DIR/M-BEIR"

# Model configuration
MODEL="residual_quantization"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="train"
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs_scripts/$SIZE/$MODE/$EXP_NAME"

# Validate model directory
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory ($MODEL_DIR) does not exist"
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export PYTHONPATH=$SRC

echo "Configuration:"
echo "GENIR_DIR: $GENIR_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
echo "NPROC: $NPROC"
echo "PYTHONPATH: $PYTHONPATH"

# Update config
CONFIG_PATH="$CONFIG_DIR/inbatch.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file ($CONFIG_PATH) does not exist"
    exit 1
fi

echo "Updating config..."
cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

# Change to model directory and run training
cd "$MODEL_DIR"
SCRIPT_NAME="train.py"

echo "Starting training..."
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

# Run training command
python -m torch.distributed.run \
    --master_port "$MASTER_PORT" \
    --nproc_per_node="$NPROC" \
    "$MODEL_DIR/$SCRIPT_NAME" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo "Training completed successfully!"
    