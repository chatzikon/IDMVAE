#!/bin/bash
#
# Generate 4x32x32 latent tensors for the CelebAMask-HQ dataset by calling
# dataset_CelebAMask_HQ.py.
#
# Usage (from idmvae/src):
#   bash commands/functions_post_eval/pregen_4x32x32_dataset_CelebAMask_HQ.sh
#
# Environment overrides:
#   GPU_ID         Override default GPU id (default: value of gpuid variable)
#   BATCH_SIZE     Override batch size passed to the python script
#   RESIZE_MODE    Override resize mode (128to256 or 256_direct)
#   SPLIT          Override split (train, val, test, all)

set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_DIR="/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask-HQ_from_SBM"
EXPERIMENT="CelebAMask_IDMVAE_release"
TITLE="IDMVAE"
RUN_ID="04-20_1_gpu1_Denoiser_IDMVAE_val_ckpt1_K1_B128_w128_z128_Laplace_b5.0_lw0.1_10.0_40.0_s2"

RUN_ID_key=$(echo "$RUN_ID" | cut -d'_' -f1-2)
CKPT_EPOCH=7


# GPU configuration
gpuid=0
GPU_ID="${GPU_ID:-$gpuid}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Optional tuning knobs
BATCH_SIZE="${BATCH_SIZE:-32}"
SD_VAE_VARIANT="${SD_VAE_VARIANT:-mse}"
RESIZE_MODE="${RESIZE_MODE:-128to256}" # Options: 128to256, 256_direct
SPLIT="${SPLIT:-all}" # Options: train, val, test, all

# ------------------------------------------------------------------------------
# Derived paths
# ------------------------------------------------------------------------------
debug=""  # _debug/
marker="_release"  #_debug
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASET_SCRIPT="${REPO_ROOT}/src/dataset_CelebAMask_HQ.py"
RUN_DIR="${REPO_ROOT}/outputs/${EXPERIMENT}/checkpoints/${RUN_ID}"
MODEL_ARGS_PATH="${RUN_DIR}/args.json"
CHECKPOINT_PATH="${RUN_DIR}/model_${CKPT_EPOCH}.rar"
OUTPUT_DIR="${DATA_DIR}/pregen_4x32x32/${debug}${TITLE}_${RUN_ID_key}_ep${CKPT_EPOCH}${marker}"
OUTPUT_DIR="${OUTPUT_DIR//-/_}"  # replace '-' with '_'
LOG_DIR="${OUTPUT_DIR}/pregen_logs"
mkdir -p "${LOG_DIR}"

timestamp="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/pregen_ep${CKPT_EPOCH}_${timestamp}.out"

echo "============================================================"
echo "Generating pre-encoded dataset for CelebAMask-HQ"
echo "Experiment      : ${EXPERIMENT}"
echo "Title           : ${TITLE}"
echo "Run ID          : ${RUN_ID}"
echo "Checkpoint epoch: ${CKPT_EPOCH}"
echo "Data directory  : ${DATA_DIR}"
echo "Checkpoint dir  : ${RUN_DIR}"
echo "Args file       : ${MODEL_ARGS_PATH}"
echo "Checkpoint      : ${CHECKPOINT_PATH}"
echo "Using GPU       : ${CUDA_VISIBLE_DEVICES}"
echo "Batch size      : ${BATCH_SIZE}"
echo "Resize Mode     : ${RESIZE_MODE}"
echo "Split           : ${SPLIT}"
echo "Log file        : ${LOG_FILE}"
echo "============================================================"

CMD=(
    python "${DATASET_SCRIPT}"
    --data-dir "${DATA_DIR}"
    --model-args "${MODEL_ARGS_PATH}"
    --checkpoint "${CHECKPOINT_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE}"
    --sd-vae "${SD_VAE_VARIANT}"
    --resize-mode "${RESIZE_MODE}"
    --split "${SPLIT}"
    --num-workers 8
)

(
    cd "${REPO_ROOT}/src"
    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
) 2>&1 | tee "${LOG_FILE}"

echo "Done. Outputs written to ${OUTPUT_DIR}"
echo "Full log stored at ${LOG_FILE}"
