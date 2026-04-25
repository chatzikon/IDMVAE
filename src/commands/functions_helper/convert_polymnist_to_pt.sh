#!/bin/bash

# Description: Script to convert and/or verify the PolyMNIST dataset from PNG to PT format.

# --- Configuration ---
OUT_ROOT="${OUT_ROOT:-/data/backed_up/shared/Data/iclr_release_test}"
SRC_ROOT="${SRC_ROOT:-/data/backed_up/shared/Data/PolyMNIST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}/src"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export WANDB_DIR="${WANDB_DIR:-${OUT_ROOT}/_cache/wandb_polymnist_convert}"
mkdir -p "${WANDB_DIR}"

root="${SRC_ROOT}"
sub_dir="quadrants"
rep="${REP:-4}"  # 1 or 4
para="${PARA:-64x64_scl1}"
date="${DATE_TAG:-Mar19_2025}"
exp="${EXP_TAG:-t0}"
data_DIR="${root}/${sub_dir}_${rep}x_${para}/${date}_${exp}"

# Path to the original image-based dataset
data_path=${data_DIR}/PolyMNIST

# The output directory for the .pt files
out_base="${OUT_ROOT}/PolyMNIST/${sub_dir}_${rep}x_${para}/${date}_${exp}"
output_path="${out_base}/PolyMNIST_pt"

run_tag="$(date '+%Y%m%d_%H%M%S')"
if [ -d "${output_path}" ] && [ "$(ls -A "${output_path}" 2>/dev/null | wc -l)" -gt 0 ]; then
    output_path="${output_path}_${run_tag}"
fi

# --- Logging ---
log_dir="${output_path}/logs"
mkdir -p ${log_dir}
logfile="${log_dir}/log_convert_polymnist_to_pt_${run_tag}.txt"

# --- Wandb & Run Configuration ---
number=1
gpuid=0
note="_${rep}x_new_label_order"
RUN_NAME="${run_tag}_${number}_gpu${gpuid}${note}"

script_path="${REPO_ROOT}/src/functions_helper/convert_polymnist_to_pt.py"

echo "Running PolyMNIST conversion/verification..."
echo "Using data directory: ${data_path}"
echo "Using output directory: ${output_path}"
echo "Logs will be saved to: ${logfile}"
echo "Wandb run name: ${RUN_NAME}"

CUDA_VISIBLE_DEVICES=${gpuid} python "${script_path}" \
    --datadir ${data_path} \
    --outputdir ${output_path} \
    --wandb_run_name "${RUN_NAME}" \
    --verify \
    --convert \
    2>&1 | tee -a "${logfile}"

echo -e "\n--------------------------------------------------"
echo "Script finished. Logs are available in $logfile"
