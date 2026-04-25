#!/bin/bash

# Shared classifier directory layout.
# label type: 'shared' or 'private'/10 epoch is enough
OUT_ROOT="${OUT_ROOT:-/data/backed_up/shared/Data/iclr_release_test}"
SRC_ROOT="${SRC_ROOT:-/data/backed_up/shared/Data/PolyMNIST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}/src"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

root_out="${OUT_ROOT}/PolyMNIST"
root_src="${SRC_ROOT}"
sub_dir="quadrants"
rep="${REP:-4}"
para="${PARA:-64x64_scl1}"
date="${DATE_TAG:-Mar19_2025}"
exp="${EXP_TAG:-t0}"
data_DIR_SRC="${root_src}/${sub_dir}_${rep}x_${para}/${date}_${exp}"
data_DIR_OUT="${root_out}/${sub_dir}_${rep}x_${para}/${date}_${exp}"

condition_type='shared'
num_epoch=30
bs=512
tr_exp="pdb"
run_tag="$(date '+%Y%m%d_%H%M%S')"
save_path="${data_DIR_OUT}/clfs_${condition_type}/${tr_exp}_bs${bs}_ep${num_epoch}_${run_tag}"
gpuid=3

script_path="${REPO_ROOT}/src/functions_helper/pretrain_classifier_PolyMNIST.py"

CUDA_VISIBLE_DEVICES=${gpuid} python "${script_path}" \
    --num_modalities 5 \
    --num_epochs $num_epoch \
    --datadir $data_DIR_SRC \
    --save_dir $save_path \
    --batch_size $bs \
    --condition_type $condition_type
