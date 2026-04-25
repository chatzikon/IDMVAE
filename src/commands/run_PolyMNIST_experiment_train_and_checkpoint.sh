#!/bin/bash
# PolyMNIST-Quadrant IDMVAE. Defaults match paper Appendix B.1 (epochs 100, batch 128, β=2.5, z=32 w=128,
# λ1=80, λ_contrast=20 with CL, diffusion weight 1.0). Adjust POLYMNIST_ROOT if needed; then set MODE for test/resume.
# cd "$(dirname "$0")/.." || exit 1

# --- mode: train | resume | test | print_params ---
MODE="train"

test_time_dataset_state="eval" # eval or test

EXPERIMENT="PolyMNIST_IDMVAE_release"
OUTPUTDIR="../outputs"
POLYMNIST_ROOT="${POLYMNIST_ROOT:-/data/backed_up/shared/Data/PolyMNIST}"
POLYMNIST_TSNE_ROOT="${POLYMNIST_TSNE_ROOT:-../outputs/t_SNE}"

root="${POLYMNIST_ROOT}"
data_DIR="${root}/quadrants_4x_64x64_scl1/Mar19_2025_t0"
DATADIR="${data_DIR}/PolyMNIST_pt"
clf_digit_path="${data_DIR}/clfs_digit/t1_bs512_ep30/DigitClassifier/trained_clfs_polyMNIST"
clf_quadrant_path="${data_DIR}/clfs_quadrant/t1_bs512_ep10/DigitClassifier/trained_clfs_polyMNIST"

K=1
BATCH=128 # 128
EPOCHS=100
SEED=2
beta=2.5
priorposterior="Normal"
likelihood="Laplace"
dim_shared=32
dim_private=128

cross_mi_scale=80
gen_aug_scale=20
gen_aug_scheme="posterior"
gen_aug_type="CL"

LV_N_NEIGHBORS=50
LV_MIN_DIST=0.1

# Diffusion-prior setting
diff_lw=1.0 # Diffusion loss weight. Set to 0.0 to disable.
diff_sg="OFF" # Stop-gradient on diffusion input: "ON" or "OFF"

debug_mini_dataset=false # false

# Test-only checkpoint configuration
CPt_OUTPUTDIR="PolyMNIST_IDMVAE_release_test"
TEST_CHECKPOINT_EPOCH=1
CPt_RUNID=04-17_1_gpu1_debug_pmNormal_ltCL_TDdevl_lw0.0_K1_B4_Normal_Laplace_b2.5_20.0_80.0_128_32_s2
CPt_RUN_key=$(echo "$CPt_RUNID" | cut -d'_' -f1-2)
CPt_NOTE="ID(${CPt_RUN_key})-EP${TEST_CHECKPOINT_EPOCH}"
TEST_CHECKPOINT_PATH="../outputs/${CPt_OUTPUTDIR}/checkpoints/${CPt_RUNID}/model_${TEST_CHECKPOINT_EPOCH}.rar"

if [ "${debug_mini_dataset}" = true ]; then
    DEBUG_FLAG="--debug_mini_data"
    db="_debug"
    marker="_test"
else
    DEBUG_FLAG=""
    db=""
    marker=""
fi

if [ "$diff_sg" = "ON" ]; then
    diff_stop_grad="--diffusion_stop_grad_on_input"
elif [ "$diff_sg" = "OFF" ]; then
    diff_stop_grad=""
else
    echo "Error: diff_sg must be 'ON' or 'OFF'" >&2
    exit 1
fi

RUN_NOTE_PREFIX=""
MODE_FLAG_ARGS=()
if [ "$MODE" = "train" ]; then
    echo "Mode: train"
elif [ "$MODE" = "resume" ]; then
    echo "Mode: resume"
    MODE_FLAG_ARGS=(--resume)
elif [ "$MODE" = "test" ]; then
    echo "Mode: test-only"
    if [ -n "${CPt_NOTE}" ]; then
        RUN_NOTE_PREFIX="CPt_${CPt_NOTE}_"
    fi
    if [ -z "${TEST_CHECKPOINT_PATH}" ]; then
        echo "Error: TEST_CHECKPOINT_PATH is empty. Set CPt_OUTPUTDIR, CPt_RUNID, TEST_CHECKPOINT_EPOCH." >&2
        exit 1
    fi
    echo "Checkpoint: ${TEST_CHECKPOINT_PATH}"
    MODE_FLAG_ARGS=(--test-only --checkpoint-path "${TEST_CHECKPOINT_PATH}")
elif [ "$MODE" = "print_params" ]; then
    echo "Mode: print_params"
    MODE_FLAG_ARGS=(--print-params-only)
    if [ -n "${TEST_CHECKPOINT_PATH}" ] && [ -f "${TEST_CHECKPOINT_PATH}" ]; then
        MODE_FLAG_ARGS+=(--checkpoint-path "${TEST_CHECKPOINT_PATH}")
    fi
else
    echo "Error: MODE must be train, resume, test, or print_params." >&2
    exit 1
fi

date=$(date '+%m-%d')
number=0
gpuid=0
note="${db}_pm${priorposterior}_lt${gen_aug_type}_TD${test_time_dataset_state}_lw${diff_lw}"
RUN_NOTE="${RUN_NOTE_PREFIX}${date}_${number}_gpu${gpuid}${note}"

EXPERIMENT_NAME="${EXPERIMENT}${marker}"
tSNE_save_dir="${POLYMNIST_TSNE_ROOT}/${EXPERIMENT_NAME}/${RUN_NOTE}"

CMD_ARGS=(
    "python" "train_IDMVAE_polyMNIST.py"
    "--experiment" "${EXPERIMENT_NAME}"
    --K "${K}"
    --batch-size "${BATCH}"
    --epochs "${EPOCHS}"
    --latent-dim-z "${dim_shared}"
    --latent-dim-w "${dim_private}"
    --seed "${SEED}"
    --beta "${beta}"
    --datadir "${DATADIR}"
    --datadir_fid "${data_DIR}"
    --outputdir "${OUTPUTDIR}"
    --pretrained_clfs_digit_dir_path "${clf_digit_path}"
    --pretrained_clfs_quadrant_dir_path "${clf_quadrant_path}"
    --priorposterior "${priorposterior}"
    --likelihood "${likelihood}"
    --diffusion_loss_weight "${diff_lw}"
    --cross_mi_loss_scale "${cross_mi_scale}"
    --gen_aug_loss_scale "${gen_aug_scale}"
    --gen_aug_sampling_scheme "${gen_aug_scheme}"
    --gen_aug_loss_type "${gen_aug_type}"
    --note "${RUN_NOTE}"
    --tSNE_save_dir "${tSNE_save_dir}"
    --lv_umap_n_neighbors "${LV_N_NEIGHBORS}"
    --lv_umap_min_dist "${LV_MIN_DIST}"
    --test_time_dataset_state "${test_time_dataset_state}"
    --enable_test_epoch
    --enable_unconditional_generation
    --enable_latent_classification
    --use_mean_for_latent_clf
    --enable_generation_coherence
    --enable_fid
    --enable_tSNE_UMAP
    # --use_mean_in_latent_visualization

)

if [ -n "${diff_stop_grad}" ]; then
    CMD_ARGS+=("${diff_stop_grad}")
fi
if [ -n "${DEBUG_FLAG}" ]; then
    CMD_ARGS+=("${DEBUG_FLAG}")
fi
CMD_ARGS+=("${MODE_FLAG_ARGS[@]}")

echo "Executing:"
printf "CUDA_VISIBLE_DEVICES=%s " "${gpuid}"
printf "%q " "${CMD_ARGS[@]}"
echo
CUDA_VISIBLE_DEVICES=${gpuid} "${CMD_ARGS[@]}"
