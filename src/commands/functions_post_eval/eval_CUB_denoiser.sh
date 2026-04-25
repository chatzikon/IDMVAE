#!/bin/bash
# Launches CUB denoiser evaluation (see eval_CUB_denoiser.py).
set -euo pipefail

# No args: uses defaults below (old denoiser ckpt + release pregen latents).
# With args: passes everything through to eval_CUB_denoiser.py.
#
# Env overrides: GPU_ID, DATA_PATH, HIGH_RES_DATA_PATH, CKPT, OUTPUT_PATH,
#   RESNET_MODEL_ARGS, RESNET_CHECKPOINT, BATCH_SIZE, NUM_WORKERS, NUM_SAMPLING_STEPS, SEED, VAE

DATA_PATH="${DATA_PATH:-/data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/IDMVAE_04-20_0_ep7_release}"
HIGH_RES_DATA_PATH="${HIGH_RES_DATA_PATH:-/data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox}"
CKPT="${CKPT:-/data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/IDMVAE_Diffdot1_Aug10_Cross40_11_17_60_ep50_002-DiT-XL-2/checkpoints/0070000.pt}"
GPU_ID="${GPU_ID:-0}"
VAE="${VAE:-mse}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
SEED="${SEED:-0}"

# Optional (only required for multimodal generation modes like text2img/img2text/img2img):
RESNET_MODEL_ARGS="${RESNET_MODEL_ARGS:-/data/not_backed_up/yijie/Project/Lab/ICLR2026/ccadiffusion/outputs/CUBcluster8_256_IDMVAE_release/checkpoints/04-20_0_gpu1_Adam_ltCL_TDdevl_lw0.1_K1_B256_Normal_Laplace_b1.0_10.0_40.0_256_256_s2/args.json}"
RESNET_CHECKPOINT="${RESNET_CHECKPOINT:-/data/not_backed_up/yijie/Project/Lab/ICLR2026/ccadiffusion/outputs/CUBcluster8_256_IDMVAE_release/checkpoints/04-20_0_gpu1_Adam_ltCL_TDdevl_lw0.1_K1_B256_Normal_Laplace_b1.0_10.0_40.0_256_256_s2/model_7.rar}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}/src"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [ "$#" -gt 0 ]; then
  exec python functions_post_eval/eval_CUB_denoiser.py "$@"
fi

if [ -z "${OUTPUT_PATH:-}" ]; then
  RUN_DIR="$(cd "$(dirname "${CKPT}")/.." && pwd)"
  OUTPUT_PATH="${RUN_DIR}/eval_$(basename "${CKPT}" .pt)"
fi

exec python functions_post_eval/eval_CUB_denoiser.py \
  --high-res-data-path "${HIGH_RES_DATA_PATH}" \
  --data-path "${DATA_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --ckpt "${CKPT}" \
  --vae "${VAE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
  --seed "${SEED}" \
  --resnet_model_args "${RESNET_MODEL_ARGS}" \
  --resnet_checkpoint "${RESNET_CHECKPOINT}"
