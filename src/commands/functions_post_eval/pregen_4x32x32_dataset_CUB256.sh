#!/bin/bash
#
# Generate 4x32x32 latent tensors for the CUB 256px dataset by calling
# dataset_CUBcluster8.py (IDMVAE path only). Edit the configuration block below to match
# the checkpoint you want to use.
#
#
# Usage (from idmvae/src):
#   bash commands/functions_post_eval/pregen_4x32x32_dataset_CUB256.sh
#
# Environment overrides:
#   GPU_ID         Override default GPU id (default: value of gpuid variable)
#   BATCH_SIZE     Override batch size passed to the python script
#
# Running (from idmvae/src):
# % bash commands/functions_post_eval/pregen_4x32x32_dataset_CUB256.sh

set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_DIR="/data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox"  # cats22_256px_70_15_15_nonbbox_seed1087
EXPERIMENT="CUBcluster8_256_IDMVAE_release"
TITLE="IDMVAE"
RUN_ID="04-20_0_gpu1_Adam_ltCL_TDdevl_lw0.1_K1_B256_Normal_Laplace_b1.0_10.0_40.0_256_256_s2"
RUN_ID_key=$(echo "$RUN_ID" | cut -d'_' -f1-2)
CKPT_EPOCH=7
# MAX_SAMPLES=10  # Optional: limit number of samples for quicker testing (comment out or set to empty for all samples)

# RUN_ID NOTE:
# Baseline: 11-15_56_gpu3_-1-1_MMVAE+_ResN_Adam_ltCL_LCmu_TDdevl_lw0.0_vcca_K1_B256_Normal_Laplace_b1.0_0.0_0.0_0.0_256_256_s2
# IDMVAEAugMI: 11-19_62_gpu2_-1-1_AugMI_ResN_Adam_ltCL_LCmu_TDdevl_lw0.0_vcca_K1_B256_Normal_Laplace_b1.0_0.0_10.0_0.0_256_256_s2
# IDMVAECrossMI: 11-15_55_gpu2_-1-1_CrossMI_ResN_Adam_ltCL_LCmu_TDdevl_lw0.0_vcca_K1_B256_Normal_Laplace_b1.0_0.0_0.0_40.0_256_256_s2
# IDMVAE:11-15_53_gpu0_-1-1_Norm_ResN_Adam_ltCL_LCmu_TDdevl_lw0.0_vcca_K1_B256_Normal_Laplace_b1.0_0.0_10.0_40.0_256_256_s2
# IDMVAE_Diffdot1: 11-17_60_gpu3_-1-1_Diff_ResN_Adam_ltCL_LCmu_TDdevl_lw0.1_vcca_K1_B256_Normal_Laplace_b1.0_0.0_10.0_40.0_256_256_s2
# DMVAE: 11-19_14_gpu3_DMVAE_CUB256_LCmu_Mpoe_W256_Z256_B256_CUBcluster8_2025_11_19_05_40_43_801775

# GPU configuration
gpuid=1
GPU_ID="${GPU_ID:-$gpuid}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Optional tuning knobs
BATCH_SIZE="${BATCH_SIZE:-1024}"  # doule check that should be same or need no as training batch size?
SD_VAE_VARIANT="${SD_VAE_VARIANT:-mse}"
GENERATE_1X="${GENERATE_1X:-0}"            # set to 1 to also generate single-sample latents (IDMVAE path)
SKIP_10X="${SKIP_10X:-0}"                  # set to 1 to skip IDMVAE 10x tensors
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-10}"  # number of samples to draw per image for the default 10x outputs

# ------------------------------------------------------------------------------
# Derived paths
# ------------------------------------------------------------------------------
marker="_release"  #_debug
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATASET_SCRIPT="${REPO_ROOT}/src/dataset_CUBcluster8.py"
RUN_DIR="${REPO_ROOT}/outputs/${EXPERIMENT}/checkpoints/${RUN_ID}"
MODEL_ARGS_PATH="${RUN_DIR}/args.json"
CHECKPOINT_PATH="${RUN_DIR}/model_${CKPT_EPOCH}.rar"
OUTPUT_DIR="${DATA_DIR}/pregen_4x32x32_${SAMPLES_PER_IMAGE}x/${TITLE}_${RUN_ID_key}_ep${CKPT_EPOCH}${marker}"
LOG_DIR="${OUTPUT_DIR}/pregen_logs"
mkdir -p "${LOG_DIR}"

# date=`echo $(date '+%m-%d')`
timestamp="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/pregen_ep${CKPT_EPOCH}_${timestamp}.out"

echo "============================================================"
echo "Generating pre-encoded dataset for CUBcluster8 256px"
echo "Experiment      : ${EXPERIMENT}"
echo "Title           : ${TITLE}"
echo "Run ID         : ${RUN_ID}"
echo "Checkpoint epoch: ${CKPT_EPOCH}"
echo "Data directory : ${DATA_DIR}"
echo "Checkpoint dir : ${RUN_DIR}"
echo "Args file      : ${MODEL_ARGS_PATH}"
echo "Checkpoint     : ${CHECKPOINT_PATH}"
echo "Using GPU      : ${CUDA_VISIBLE_DEVICES}"
echo "Batch size     : ${BATCH_SIZE}"
if [ "${SKIP_10X}" -eq 1 ]; then
    echo "10x latents    : SKIPPED (--skip-10x)"
else
    echo "10x latents    : ENABLED (default, samples/image=${SAMPLES_PER_IMAGE})"
fi
if [ "${GENERATE_1X}" -eq 1 ]; then
    echo "1x latents     : ENABLED (--generate-1x)"
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
    echo "Max samples    : ${MAX_SAMPLES}"
fi
echo "Log file       : ${LOG_FILE}"
echo "============================================================"

CMD=(
    python "${DATASET_SCRIPT}"
    --data-dir "${DATA_DIR}"
    --model-args "${MODEL_ARGS_PATH}"
    --checkpoint "${CHECKPOINT_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE}"
    --sd-vae "${SD_VAE_VARIANT}"
    --num-workers 64
)

if [ "${SKIP_10X}" -eq 1 ]; then
    CMD+=(--skip-10x)
fi
if [ "${GENERATE_1X}" -eq 1 ]; then
    CMD+=(--generate-1x)
fi
if [ -n "${SAMPLES_PER_IMAGE:-}" ]; then
    CMD+=(--samples-per-image "${SAMPLES_PER_IMAGE}")
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi

(
    cd "${REPO_ROOT}/src"
    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
) 2>&1 | tee "${LOG_FILE}"

echo "Done. Outputs written to ${OUTPUT_DIR}"
echo "Full log stored at ${LOG_FILE}"
