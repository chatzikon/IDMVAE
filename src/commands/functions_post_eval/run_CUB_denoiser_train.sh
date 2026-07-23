#!/bin/bash
# Launches DiT denoiser training on CUB 256 pregen latents (train_CUB_denoiser.py).
#
# No args: uses defaults below; set PRETRAIN_CKPT to DiT-XL-2-256x256.pt (or export it).
# With args: passes everything through to train_CUB_denoiser.py (GPU still from GPU_ID unless you export CUDA_VISIBLE_DEVICES first).
#
# Env overrides: GPU_ID, DATA_PATH, HIGH_RES_DATA_PATH, PRETRAIN_CKPT, MODEL, EPOCHS,
#   BATCH_SIZE, CKPT_EVERY, LOG_EVERY, NUM_WORKERS
set -euo pipefail

# --- Defaults (release pregen ep7) ---
DATA_PATH="${DATA_PATH:-/home/chatziko/PycharmProjects/PythonProject/IDMVAE/CUB_HQ/pregen_4x32x32_10x/IDMVAE_07-21_0_ep36_release}"
HIGH_RES_DATA_PATH="${HIGH_RES_DATA_PATH:-/home/chatziko/PycharmProjects/PythonProject/IDMVAE/CUB_HQ}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
DEFAULT_PRETRAIN_CKPT="/home/chatziko/PycharmProjects/PythonProject/IDMVAE/checkpoints/DiT-XL-2-256x256.pt"
GPU_ID="${GPU_ID:-0}"
MODEL="${MODEL:-DiT-XL/2}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CKPT_EVERY="${CKPT_EVERY:-100000}"
LOG_EVERY="${LOG_EVERY:-5000}"
NUM_WORKERS="${NUM_WORKERS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}/src"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [ "$#" -gt 0 ]; then
  exec python functions_post_eval/train_CUB_denoiser.py "$@"
fi

if [ -z "${PRETRAIN_CKPT}" ] && [ -f "${DEFAULT_PRETRAIN_CKPT}" ]; then
  PRETRAIN_CKPT="${DEFAULT_PRETRAIN_CKPT}"
fi

if [ -z "${PRETRAIN_CKPT}" ]; then
  echo "ERROR: Set PRETRAIN_CKPT to your DiT-XL-2-256x256.pt path, e.g." >&2
  echo "  PRETRAIN_CKPT=/path/to/DiT-XL-2-256x256.pt $0" >&2
  echo "Or pass full arguments: $0 --data-path ... --pretrain-ckpt ..." >&2
  exit 1
fi

exec python functions_post_eval/train_CUB_denoiser.py \
  --high-res-data-path "${HIGH_RES_DATA_PATH}" \
  --data-path "${DATA_PATH}" \
  --pretrain-ckpt "${PRETRAIN_CKPT}" \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --ckpt-every "${CKPT_EVERY}" \
  --log-every "${LOG_EVERY}" \
  --num-workers "${NUM_WORKERS}"
