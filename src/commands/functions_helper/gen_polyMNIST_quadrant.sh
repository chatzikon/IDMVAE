# Generates PolyMNIST quadrant dataset.
#
# To avoid overwriting paper artifacts, outputs default to:
#   /data/backed_up/shared/Data/iclr_release_test
# and source assets (e.g. backgrounds) default to the original PolyMNIST root.
#
# Env overrides:
#   OUT_ROOT, SRC_ROOT, REP, PARA, DATE_TAG, EXP_TAG

OUT_ROOT="${OUT_ROOT:-/data/backed_up/shared/Data/iclr_release_test}"
SRC_ROOT="${SRC_ROOT:-/data/backed_up/shared/Data/PolyMNIST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

root="${OUT_ROOT}/PolyMNIST"
sub_dir="quadrants"
rep="${REP:-4}"
para="${PARA:-64x64_scl1}"
date="${DATE_TAG:-Mar19_2025}"
exp="${EXP_TAG:-t0}"
data_DIR="${root}/${sub_dir}_${rep}x_${para}/${date}_${exp}"
savepath="${data_DIR}/PolyMNIST"

WORK_DIR="${OUT_ROOT}/_cache/polymnist_gen"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

python "${REPO_ROOT}/src/dataset_PolyMNIST_quadrant.py" \
    --seed 123 \
    --num-modalities 5 \
    --backgroundimagepath ${SRC_ROOT}/Background/Feb24_2025/ \
    --savepath-train ${savepath}/train \
    --savepath-val ${savepath}/val \
    --savepath-test ${savepath}/test \
    --repetitions ${rep} \
    --train-split-size 55000 \
    --wandb-project PolyMNIST_Dataset_test \
    --wandb-runname ${sub_dir}_${rep}x_${para}_${date}_${exp}
