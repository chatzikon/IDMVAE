#!/bin/bash

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/data/backed_up/shared/Data/iclr_release_test}"
DATA_ROOT="${DATA_ROOT:-/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask-HQ_from_SBM}"
SAVE_PATH="${SAVE_PATH:-${OUT_ROOT}/CelebAMask_HQ/CelebAMask_HQ_from_SBM_pt}"
SRC_PT_PATH="${SRC_PT_PATH:-/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM_pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}/src"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

run_tag="$(date '+%Y%m%d_%H%M%S')"
if [ -d "${SAVE_PATH}" ] && [ "$(ls -A "${SAVE_PATH}" 2>/dev/null | wc -l)" -gt 0 ]; then
  SAVE_PATH="${SAVE_PATH}_${run_tag}"
fi

gpuid="${GPU_ID:-1}"
script_path="${REPO_ROOT}/src/functions_helper/convert_celebamask_hq_to_pt.py"

# For release verification, avoid recomputing huge .pt tensors.
# Instead, link the already-converted paper .pt files into a fresh test directory,
# then run --verify on that directory.
if [ ! -d "${SAVE_PATH}" ]; then
  mkdir -p "${SAVE_PATH}"
fi

needed_files=(splits_idx.pt images.pt masks.pt attributes.pt attr_names.json)
linked_any=0
for f in "${needed_files[@]}"; do
  if [ -e "${SAVE_PATH}/${f}" ]; then
    linked_any=1
    continue
  fi
  if [ -e "${SRC_PT_PATH}/${f}" ]; then
    ln -s "${SRC_PT_PATH}/${f}" "${SAVE_PATH}/${f}"
    linked_any=1
  fi
done

if [ "${linked_any}" -eq 0 ]; then
  echo "ERROR: Could not find converted CelebAMask-HQ .pt files to verify." >&2
  echo "  Expected existing converted dataset at: ${SRC_PT_PATH}" >&2
  echo "  You can override with: SRC_PT_PATH=/path/to/CelebAMask_HQ_from_SBM_pt $0" >&2
  echo "  Or run conversion flags in the python script to generate new .pt files." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpuid} python "${script_path}" \
    --data_root "${DATA_ROOT}" \
    --save_path "${SAVE_PATH}" \
    --verify \
    # --convert_images \
    # --convert_masks \
    # --convert_attributes \
    # --save_splits \
    # --verify \
    # --image_size 256 \
    # --mask_size 128 \
    # --save_path /data/backed_up/shared/Data/iclr_release_test/CelebAMask_HQ/CelebAMask_HQ_from_SBM_pt
