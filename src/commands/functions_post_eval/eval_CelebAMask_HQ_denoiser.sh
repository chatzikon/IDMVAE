#!/bin/bash
# Launches CelebAMask denoiser evaluation (see eval_CelebAMask_HQ_denoiser.py).
set -euo pipefail

# No args: uses defaults below (old denoiser ckpt + release pregen latents).
# With args: passes everything through to eval_CelebAMask_HQ_denoiser.py.
#
# Env overrides: GPU_ID, DATA_PATH, HIGH_RES_DATA_PATH, CKPT, OUTPUT_PATH, SPLIT,
#   RESNET_MODEL_ARGS, RESNET_CHECKPOINT, BATCH_SIZE, NUM_WORKERS, NUM_SAMPLING_STEPS, SEED, VAE

DATA_PATH="${DATA_PATH:-/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/IDMVAE_04_20_1_ep7_release}"
HIGH_RES_DATA_PATH="${HIGH_RES_DATA_PATH:-/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask-HQ_from_SBM}"
CKPT="${CKPT:-/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/Diff_12_01_11_ep100_003-DiT-XL-2/checkpoints/0070000.pt}"
GPU_ID="${GPU_ID:-0}"
VAE="${VAE:-mse}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
SEED="${SEED:-0}"

# Optional (only required for multimodal generation modes like attr2img/mask2img/img2img):
RESNET_MODEL_ARGS="${RESNET_MODEL_ARGS:-/data/not_backed_up/yijie/Project/Lab/ICLR2026/ccadiffusion/outputs/CelebAMask_IDMVAE_release/checkpoints/04-20_1_gpu1_Denoiser_IDMVAE_val_ckpt1_K1_B128_w128_z128_Laplace_b5.0_lw0.1_10.0_40.0_s2/args.json}"
RESNET_CHECKPOINT="${RESNET_CHECKPOINT:-/data/not_backed_up/yijie/Project/Lab/ICLR2026/ccadiffusion/outputs/CelebAMask_IDMVAE_release/checkpoints/04-20_1_gpu1_Denoiser_IDMVAE_val_ckpt1_K1_B128_w128_z128_Laplace_b5.0_lw0.1_10.0_40.0_s2/model_7.rar}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}/src"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [ "$#" -gt 0 ]; then
  exec python functions_post_eval/eval_CelebAMask_HQ_denoiser.py "$@"
fi

if [ -z "${OUTPUT_PATH:-}" ]; then
  RUN_DIR="$(cd "$(dirname "${CKPT}")/.." && pwd)"
  OUTPUT_PATH="${RUN_DIR}/eval_$(basename "${CKPT}" .pt)"
fi

exec python functions_post_eval/eval_CelebAMask_HQ_denoiser.py \
  --high-res-data-path "${HIGH_RES_DATA_PATH}" \
  --data-path "${DATA_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --split "${SPLIT}" \
  --ckpt "${CKPT}" \
  --vae "${VAE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
  --seed "${SEED}" \
  --resnet_model_args "${RESNET_MODEL_ARGS}" \
  --resnet_checkpoint "${RESNET_CHECKPOINT}"
