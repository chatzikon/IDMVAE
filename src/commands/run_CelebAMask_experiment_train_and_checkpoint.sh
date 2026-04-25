#!/bin/bash

#================================================================================
# CelebAMask HQ → train_IDMVAE_CelebAMask.py (run from ccadiffusion/src).
#
# MODE=test: needs a checkpoint. Either (1) set CPt_RUNID + TEST_CHECKPOINT_EPOCHS
# and ensure ../outputs/${CPt_OUTPUTDIR}/checkpoints/${CPt_RUNID}/model_<epoch>.rar
# exists, or (2) set TEST_CHECKPOINT_PATH to a .rar file and clear TEST_CHECKPOINT_EPOCHS.
#
# Dataset default: CELEBAMASK_ROOT (train_img/ val_img/ under datadir). Denoiser is
# optional; if ${CELEBAMASK_ROOT}/pregen_4x32x32/_denoiser/... is missing, set DENOISER=""
# to skip loading.
#
# MODE=train: checkpoints go to ../outputs/${EXPERIMENT}/checkpoints/<runId>/model_<epoch>.rar
# (relative to ccadiffusion/src). Use that path with MODE=test after training.
#================================================================================
# SCRIPT CONFIGURATION
#================================================================================

# Set the execution mode for the script.
# Options:
#   "train" : Start a new training run from scratch.
#   "resume": Resume a training run from the latest saved checkpoint.
#   "test"  : Run evaluation only on a specific checkpoint.
#   "develop": Fresh run while keeping the same run ID/output directory for quick debugging iteration.
#   "print_params": Instantiate the model (optionally load a checkpoint) and print parameter counts, then exit.
MODE="resume" # Options: "train", "resume", "test", "develop", "print_params"


# Wandb project name
MARKER="" # __test
DATASET="CelebAMask"
CELEBAMASK_ROOT="${CELEBAMASK_ROOT:-/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask-HQ_from_SBM}" # (not .pt file)
DENOISER_ROOT="${DENOISER_ROOT:-${CELEBAMASK_ROOT}/pregen_4x32x32/_denoiser}"
DATADIR="${CELEBAMASK_ROOT}/"
EXPERIMENT="CelebAMask_IDMVAE_release${MARKER}"
CPt_OUTPUTDIR="CelebAMask_IDMVAE_release"

# --- Model Hyperparameters ---
BATCH=128
K=1
EPOCHS="${EPOCHS:-100}"
SEED=2
LATENT_DIM_W=128
LATENT_DIM_Z=128
LR=0.0002
BETA=5.0

# --- Optimizer ---
Opt="Adam"

TITLE="IDMVAE_" # AugMI_ Diff_
# --- Diffusion Prior Specific ---
diff_lw=0.1
# --- VCCA Specific Hyperparameters ---
AUG_L_SC=10.0
CROSS_L_SC=40.0

aug_mi_ss="posterior"
AUG_MI_LT="CL"

# --- Training, testing, and evaluating Control ---
SKIP_TRAIN=false # true or false to skip training
TEST_FREQ=1        # 0 disables
EVAL_FREQ=1        # 0 disables
QUALITATIVE_FREQ=1 # 0 disables, dominated by EVAL_FREQ
F1_FREQ=1          # 0 disables, dominated by EVAL_FREQ
FID_FREQ=1         # 0 disables, dominated by EVAL_FREQ
CKPT_FREQ=1
DEBUG_LOADER=false # true or false to use validation set for training (debug)

VAL_TEST_DATASET="val"  # Options: "val", "test"
EVAL_SHUFFLE=true # true or false to shuffle evaluation dataset
MAX_FID_IMAGES= # full (empty, default) Maximum number of images for FID calculation. Set to empty string or 0 to use default (or full dataset if handled by script)
EVAL_FUSION_RANDOM=true # true or false to enable random selection for multi-modal fusion evaluation
EVAL_FUSION_AVERAGE=true # true or false to enable latent averaging for multi-modal fusion evaluation

date=`echo $(date '+%m-%d')`
number=1
gpuid=1
note="_Denoiser_${TITLE}${VAL_TEST_DATASET}_ckpt${CKPT_FREQ}"
# _F1+FID: enable F1 and FID evaluation
# _fid256: fid batch size 256
# _ShufFull: evaluation with shuffling and full dataset for FID
# _both: evaluate both fusion methods

# CPt_OUTPUTDIR="CelebAMask_IDMVAE"
# --- Test-Only Mode Configuration (MODE=test) ---
# With TEST_CHECKPOINT_EPOCHS set: resolves ../outputs/${CPt_OUTPUTDIR}/checkpoints/${CPt_RUNID}/model_<n>.rar
# With TEST_CHECKPOINT_EPOCHS empty: set TEST_CHECKPOINT_PATH to a .rar (absolute path ok).
TEST_CHECKPOINT_EPOCHS="1"
TEST_CHECKPOINT_PATH=""
CPt_RUNID="04-20_1_gpu1_Denoiser_IDMVAE_val_ckpt1_K1_B128_w128_z128_Laplace_b5.0_lw0.1_10.0_40.0_s2"


#================================================================================
# Experiment Specific Configuration
#================================================================================
# TITLE="MMVAE+_"
# TEST_CHECKPOINT_EPOCHS="100" # ""50 75 100 150 200 250 300"
# CPt_RUNID="11-29_0_gpu0_MMVAE+_F1+FID_ckpt25_vcca_K1_B128_w128_z128_Laplace_b5.0_lw0.0_0.0_0.0_0.0_s2"
# diff_lw=0.0
# AUG_L_SC=0.0
# CROSS_L_SC=0.0

# TITLE="AugMI_"
# TEST_CHECKPOINT_EPOCHS="25 50 75 100" # "100 150 200"
# CPt_RUNID="12-02_14_gpu2_ShufFull_both_AugMI_test_F1+FID_ckpt25_vcca_K1_B128_w128_z128_Laplace_b5.0_lw0.0_0.0_30.0_0.0_s2"
# diff_lw=0.0
# AUG_L_SC=30.0
# CROSS_L_SC=0.0

# TITLE="CrossMI_"
# TEST_CHECKPOINT_EPOCHS="50 100" # "100 150 200"
# CPt_RUNID="11-28_0_CrossMI60_CelebAMask_IDMVAE__test_vcca_K1_B128_Laplace_b5.0_s2"
# diff_lw=0.0
# AUG_L_SC=0.0
# CROSS_L_SC=60.0

# # TITLE="CrossMI_"
# # TEST_CHECKPOINT_EPOCHS="25 50 75 100" # "100 150 200"
# # CPt_RUNID="11-29_01_gpu3_FID_ckpt25_vcca_K1_B128_w128_z128_Laplace_b5.0_lw0.0_0.0_0.0_40.0_s2"
# # diff_lw=0.0 
# # AUG_L_SC=0.0
# # CROSS_L_SC=40.0

# TITLE="IDMVAE_"
# TEST_CHECKPOINT_EPOCHS="100" # "25 50 75 100 150 200"
# CPt_RUNID="11-30_5_gpu1_IDMVAE_val_F1+FID_ckpt25_vcca_K1_B128_w128_z128_Laplace_b5.0_lw0.0_0.0_30.0_60.0_s2"
# diff_lw=0.0
# AUG_L_SC=30.0
# CROSS_L_SC=60.0

# Optional: switch to Diff-prior experiment (uncomment and adjust CPt_RUNID for MODE=test).
# TITLE="Diff_"
# TEST_CHECKPOINT_EPOCHS="100"
# CPt_RUNID="12-01_11_gpu3_fid256_both_Diff_val_F1+FID_ckpt25_vcca_K1_B128_w128_z128_Laplace_b5.0_lw0.1_0.0_30.0_60.0_s2"
# diff_lw=0.1
# AUG_L_SC=30.0
# CROSS_L_SC=60.0

CPt_RUNID="${CPt_RUNID:-}"
CPt_RUN_key=$(echo "$CPt_RUNID" | cut -d'_' -f1-2)
CPt_NOTE="${TITLE}ID${CPt_RUN_key}"

# Test mode: pass explicit file path(s) via --checkpoint-path (see MODE=test below).

#================================================================================

# # --- General & Dataset Paths ---
# OUTPUTDIR="../outputs"

# --- Denoiser (for denoised generation plotting) ---
# Path to trained DiT denoiser; leave DENOISER empty to disable.
OUTPUTDIR="${OUTPUTDIR:-../outputs}"
denoiser_root="${DENOISER_ROOT}"
# # IDMVAE denoiser:
# DENOISER="IDMVAE_11_30_5_ep100_000-DiT-XL-2/checkpoints/0050000.pt"
# # MMVAEplus denoiser:
# DENOISER="MMVAEplus_11_29_0_ep100_001-DiT-XL-2/checkpoints/0050000.pt"
# Diff denoiser (enable when pregen_4x32x32/_denoiser exists):
# DENOISER="Diff_12_01_11_ep100_003-DiT-XL-2/checkpoints/0050000.pt"
DENOISER=""
denoiser_ckpt_path=""
if [ -n "${DENOISER}" ]; then
    denoiser_ckpt_path="${denoiser_root}/${DENOISER}"
fi
denoiser_steps=250  #250
save_eval_images_root="${OUTPUTDIR}/${EXPERIMENT}/eval_images/${CPt_NOTE}"

#================================================================================
# SCRIPT LOGIC
#================================================================================

# --- Set up Mode-Specific Variables ---
MODE_FLAG=""
RUN_NOTE_PREFIX=""
IS_DEVELOP_MODE=0

if [ "$MODE" = "train" ]; then
    echo "Mode: Starting a new training run..."
    MODE_FLAG_ARGS=()
elif [ "$MODE" = "resume" ]; then
    echo "Mode: Resuming training..."
    MODE_FLAG_ARGS=("--resume")
elif [ "$MODE" = "test" ]; then
    echo "Mode: Test-only..."
    RUN_NOTE_PREFIX="CPt_${CPt_NOTE}_"
    MODE_FLAG_ARGS=("--test-only")
    if [ -n "$TEST_CHECKPOINT_EPOCHS" ]; then
        echo "Using checkpoint epochs: ${TEST_CHECKPOINT_EPOCHS}"
        for epoch in $TEST_CHECKPOINT_EPOCHS; do
            ckpt_file="../outputs/${CPt_OUTPUTDIR}/checkpoints/${CPt_RUNID}/model_${epoch}.rar"
            if [ ! -f "$ckpt_file" ]; then
                echo "Error: Checkpoint file not found: $ckpt_file" >&2
                exit 1
            fi
            echo "Verified checkpoint exists: $ckpt_file"
            MODE_FLAG_ARGS+=("--checkpoint-path" "$ckpt_file")
        done
    elif [ -n "$TEST_CHECKPOINT_PATH" ]; then
        echo "Using checkpoint path: ${TEST_CHECKPOINT_PATH}"
        MODE_FLAG_ARGS+=("--checkpoint-path" "${TEST_CHECKPOINT_PATH}")
    else
        echo "Error: For test mode, set TEST_CHECKPOINT_EPOCHS (with CPt_RUNID) or TEST_CHECKPOINT_PATH." >&2
        exit 1
    fi
elif [ "$MODE" = "develop" ]; then
    echo "Mode: Develop (fresh run in existing directory)..."
    MODE_FLAG_ARGS=(
        "--develop"
    )
    IS_DEVELOP_MODE=1
elif [ "$MODE" = "print_params" ]; then
    echo "Mode: Print parameters only..."
    MODE_FLAG_ARGS=("--print-params-only")
    if [ -n "$TEST_CHECKPOINT_PATH" ]; then
        MODE_FLAG_ARGS+=("--checkpoint-path" "${TEST_CHECKPOINT_PATH}")
    fi
else
    echo "Error: Invalid MODE specified. Choose 'train', 'resume', 'test', 'develop', or 'print_params'."
    exit 1
fi

# date=`echo $(date '+%m-%d')`
# number=C0
# gpuid=1
# note="_${VAL_TEST_DATASET}_F1+FID_ckpt${CKPT_FREQ}"

RUN_NOTE="${RUN_NOTE_PREFIX}${date}_${number}_gpu${gpuid}${note}"

CMD_ARGS=(
    "./train_IDMVAE_CelebAMask.py"
    "--experiment" "${EXPERIMENT}"
    "--datadir" "${DATADIR}"
    "--K" "${K}"
    "--batch-size" "${BATCH}"
    "--epochs" "${EPOCHS}"
    "--latent-dim-w" "${LATENT_DIM_W}"
    "--latent-dim-z" "${LATENT_DIM_Z}"
    "--lr" "${LR}"
    "--seed" "${SEED}"
    "--beta" "${BETA}"
    "--diffusion_loss_weight" "${diff_lw}"
    "--gen_aug_loss_scale" "${AUG_L_SC}"
    "--cross_mi_loss_scale" "${CROSS_L_SC}"
    "--gen_aug_sampling_scheme" "${aug_mi_ss}"
    "--gen_aug_loss_type" "${AUG_MI_LT}"
    "--test_freq" "${TEST_FREQ}"
    "--eval_freq" "${EVAL_FREQ}"
    "--qualitative_freq" "${QUALITATIVE_FREQ}"
    "--f1_freq" "${F1_FREQ}"
    "--fid_freq" "${FID_FREQ}"
    "--ckpt_freq" "${CKPT_FREQ}"
    "--run_note" "${RUN_NOTE}"
    "--val-test-dataset" "${VAL_TEST_DATASET}"
    "--outputdir" "${OUTPUTDIR}"
    "--denoiser_num_sampling_steps" "${denoiser_steps}"
    "--save_eval_images_root" "${save_eval_images_root}"
)
if [ -n "${denoiser_ckpt_path}" ]; then
    CMD_ARGS+=("--denoiser_ckpt" "${denoiser_ckpt_path}")
fi

if [ "$SKIP_TRAIN" = true ]; then
    CMD_ARGS+=("--skip_train")
fi

if [ "$DEBUG_LOADER" = true ]; then
    CMD_ARGS+=("--debug_loader")
fi

if [ "$EVAL_SHUFFLE" = true ]; then
    CMD_ARGS+=("--eval-shuffle")
fi

if [ -n "$MAX_FID_IMAGES" ]; then
    CMD_ARGS+=("--max-fid-images" "${MAX_FID_IMAGES}")
fi

if [ "$EVAL_FUSION_RANDOM" = true ]; then
    CMD_ARGS+=("--eval-fusion-random")
fi

if [ "$EVAL_FUSION_AVERAGE" = true ]; then
    CMD_ARGS+=("--eval-fusion-average")
fi

CMD_ARGS+=("${MODE_FLAG_ARGS[@]}")

echo "Executing command:"
echo "CUDA_VISIBLE_DEVICES=${gpuid} python $(printf '%q ' "${CMD_ARGS[@]}")"

CUDA_VISIBLE_DEVICES=${gpuid} python "${CMD_ARGS[@]}"
