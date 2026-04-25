#!/bin/bash
# CUBcluster8 @ 256px (CUB-HQ style). Paper Appendix C.2 defaults: 50 epochs, batch 256, β=1.0,
# latent 256/256, λ1=40 (cross_mi), λ2=10 (gen_aug CL), diffusion weight 0.1, Adam lr 1e-3 (see train script).
# cd "$(dirname "$0")/.." || exit 1

# Unified CUB 256 launcher:
# - train / resume / develop / test / print_params
# - test: set TEST_CHECKPOINT_PATH, or CPt_RUNID + TEST_CHECKPOINT_EPOCH (path uses same EXPERIMENT as training)

MODE="train"  # train, resume, test, develop, print_params — set train when not evaluating a checkpoint
test_time_dataset_state="eval"  # eval, test

# Optional path overrides for portability:
#   export CUB_ROOT=/path/to/CUB
#   export CUB_TSNE_ROOT=/path/to/t_SNE
#   export DENOISER_ROOT=/path/to/denoisers
CUB_ROOT="${CUB_ROOT:-/data/backed_up/shared/Data/CUB}"
CUB_TSNE_ROOT="${CUB_TSNE_ROOT:-../outputs/t_SNE}"
DENOISER_ROOT="${DENOISER_ROOT:-/data/backed_up/shared/Data/CUB/weiran_dit_denoisers}"

DATASET="CUBcluster8_256"
marker=""
DATADIR="${CUB_ROOT}/CUBcluster8_256/cats22_256px_70_15_15_nonbbox"
EXPERIMENT="CUBcluster8_256_IDMVAE_release${marker}"
CPt_OUTPUTDIR="${EXPERIMENT}"

OUTPUTDIR="../outputs"

# Test-only: set MODE=test and either TEST_CHECKPOINT_PATH (absolute recommended) or CPt_RUNID + TEST_CHECKPOINT_EPOCH.
TEST_CHECKPOINT_EPOCH=1
TEST_CHECKPOINT_EPOCHS=()  # Example: (10 20 30 40)
CPt_RUNID="04-17_1_gpu1_Adam_ltCL_TDdevl_lw0.1_K1_B256_Normal_Laplace_b1.0_10.0_40.0_256_256_s2"
CPt_RUN_key=$(echo "$CPt_RUNID" | cut -d'_' -f1-2)
CPt_NOTE="ID${CPt_RUN_key}-EP${TEST_CHECKPOINT_EPOCH}"
CPt_BASE_PATH="../outputs/${CPt_OUTPUTDIR}/checkpoints/${CPt_RUNID}"
TEST_CHECKPOINT_PATH=""

if [ "$MODE" = "test" ]; then
  if [ -z "$TEST_CHECKPOINT_PATH" ] && [ "${#TEST_CHECKPOINT_EPOCHS[@]}" -eq 0 ] && [ -n "$CPt_RUNID" ]; then
    TEST_CHECKPOINT_PATH="${CPt_BASE_PATH}/model_${TEST_CHECKPOINT_EPOCH}.rar"
    echo "Derived TEST_CHECKPOINT_PATH for test mode: ${TEST_CHECKPOINT_PATH}"
  fi
fi

# Model hyperparameters (Appendix C.2: 50 epochs; λ1=40, λ2=10, diffusion 0.1)
BATCH=256
K=1
EPOCHS=50
SEED=2
NUM_WORKERS=4

beta=1.0
priorposterior="Normal"
MS_LAT_DIM=256
SHARED_LAT_DIM=256

diff_lw=0.1
diff_sg="OFF"

CROSS_L_SC=40.0
GEN_AUG_L_SC=10.0
gen_aug_scheme="posterior"
GEN_AUG_TYPE="CL"

LV_N_NEIGHBORS=20
LV_MIN_DIST=0.05
thres_deg=0.0

ENABLE_TEST_EPOCH=true
ENABLE_UNCONDITIONAL_GENERATION=true
ENABLE_LATENT_CLASSIFICATION=true
USE_MEAN_FOR_LATENT_CLF=true
ENABLE_FID=false
ENABLE_TSNE_UMAP=true
USE_MEAN_IN_LATENT_VISUALIZATION=true

DENOISER=""  # Example: "IDMVAE_Cross40_11_15_55_ep50_003-DiT-XL-2/checkpoints/0050000.pt"
denoiser_steps=250
if [ -n "$DENOISER" ]; then
  denoiser_ckpt_path="${DENOISER_ROOT}/${DENOISER}"
else
  denoiser_ckpt_path=""
fi
save_eval_images_root="${OUTPUTDIR}/${EXPERIMENT}/eval_images/${CPt_NOTE}"

if [ "$diff_sg" = "ON" ]; then
  diff_stop_grad="--diffusion_stop_grad_on_input"
elif [ "$diff_sg" = "OFF" ]; then
  diff_stop_grad=""
else
  echo "Error: diff_sg must be ON or OFF" >&2
  exit 1
fi

MODE_FLAG_ARGS=()
RUN_NOTE_PREFIX=""
IS_DEVELOP_MODE=0

case "$MODE" in
  "train")
    echo "Mode: train"
    ;;
  "resume")
    echo "Mode: resume"
    MODE_FLAG_ARGS=("--resume")
    ;;
  "test")
    echo "Mode: test"
    RUN_NOTE_PREFIX="CPt_${CPt_NOTE}_"
    if [ "${#TEST_CHECKPOINT_EPOCHS[@]}" -gt 0 ]; then
      MODE_FLAG_ARGS=()
    elif [ -n "$TEST_CHECKPOINT_PATH" ]; then
      if [ ! -f "$TEST_CHECKPOINT_PATH" ]; then
        echo "Error: checkpoint not found: $TEST_CHECKPOINT_PATH (cwd=$(pwd))" >&2
        exit 1
      fi
      MODE_FLAG_ARGS=("--test-only" "--checkpoint-path" "${TEST_CHECKPOINT_PATH}")
    else
      echo "Error: MODE=test requires TEST_CHECKPOINT_PATH, or non-empty TEST_CHECKPOINT_EPOCHS, or CPt_RUNID to derive a path." >&2
      exit 1
    fi
    ;;
  "develop")
    echo "Mode: develop"
    MODE_FLAG_ARGS=("--develop")
    IS_DEVELOP_MODE=1
    ;;
  "print_params")
    echo "Mode: print_params"
    MODE_FLAG_ARGS=("--print-params-only")
    if [ -n "$TEST_CHECKPOINT_PATH" ] && [ -f "$TEST_CHECKPOINT_PATH" ]; then
      MODE_FLAG_ARGS+=("--checkpoint-path" "${TEST_CHECKPOINT_PATH}")
    fi
    ;;
  *)
    echo "Error: Invalid MODE=${MODE}" >&2
    exit 1
    ;;
esac

date=$(date '+%m-%d')
number=1
gpuid=3
note="_lt${GEN_AUG_TYPE}_TD${test_time_dataset_state}_lw${diff_lw}"
RUN_NOTE="${RUN_NOTE_PREFIX}${date}_${number}_gpu${gpuid}${note}"
tSNE_save_dir="${CUB_TSNE_ROOT}/${EXPERIMENT}/${date}_${number}_${note}"

CMD_ARGS_BASE=(
  "python" "train_IDMVAE_CUB.py"
  "--experiment" "${EXPERIMENT}"
  "--K" "${K}"
  "--batch-size" "${BATCH}"
  "--epochs" "${EPOCHS}"
  "--latent-dim-z" "${SHARED_LAT_DIM}"
  "--latent-dim-w" "${MS_LAT_DIM}"
  "--seed" "${SEED}"
  "--beta" "${beta}"
  "--datadir" "${DATADIR}"
  "--outputdir" "${OUTPUTDIR}"
  "--inception_path" "${DATADIR}/pt_inception-2015-12-05-6726825d.pth"
  "--dataset" "${DATASET}"
  "--priorposterior" "${priorposterior}"
  "--diffusion_loss_weight" "${diff_lw}"
  "--cross_mi_loss_scale" "${CROSS_L_SC}"
  "--gen_aug_loss_scale" "${GEN_AUG_L_SC}"
  "--gen_aug_sampling_scheme" "${gen_aug_scheme}"
  "--gen_aug_loss_type" "${GEN_AUG_TYPE}"
  "--note" "${RUN_NOTE}"
  "--tSNE_save_dir" "${tSNE_save_dir}"
  "--lv_umap_n_neighbors" "${LV_N_NEIGHBORS}"
  "--lv_umap_min_dist" "${LV_MIN_DIST}"
  "--test_time_dataset_state" "${test_time_dataset_state}"
  "--degree_away_center_threshold" "${thres_deg}"
  "--num_workers" "${NUM_WORKERS}"
  "--use_pretrain_feats"
)

if [ -n "${diff_stop_grad}" ]; then
  CMD_ARGS_BASE+=("${diff_stop_grad}")
fi
if [ -n "${denoiser_ckpt_path}" ]; then
  CMD_ARGS_BASE+=(
    "--denoiser_ckpt" "${denoiser_ckpt_path}"
    "--denoiser_num_sampling_steps" "${denoiser_steps}"
    "--save_eval_images_root" "${save_eval_images_root}"
  )
fi

if [ "${ENABLE_TEST_EPOCH}" = true ]; then
  CMD_ARGS_BASE+=("--enable_test_epoch")
fi
if [ "${ENABLE_UNCONDITIONAL_GENERATION}" = true ]; then
  CMD_ARGS_BASE+=("--enable_unconditional_generation")
fi
if [ "${ENABLE_LATENT_CLASSIFICATION}" = true ]; then
  CMD_ARGS_BASE+=("--enable_latent_classification")
fi
if [ "${USE_MEAN_FOR_LATENT_CLF}" = true ]; then
  CMD_ARGS_BASE+=("--use_mean_for_latent_clf")
fi
if [ "${ENABLE_FID}" = true ]; then
  CMD_ARGS_BASE+=("--enable_fid")
fi
if [ "${ENABLE_TSNE_UMAP}" = true ]; then
  CMD_ARGS_BASE+=("--enable_tSNE_UMAP")
fi
if [ "${USE_MEAN_IN_LATENT_VISUALIZATION}" = true ]; then
  CMD_ARGS_BASE+=("--use_mean_in_latent_visualization")
fi

execute_with_args() {
  local -a cmd=("$@")
  echo "Executing command:"
  printf "CUDA_VISIBLE_DEVICES=%s " "${gpuid}"
  printf "%q " "${cmd[@]}"
  echo
  CUDA_VISIBLE_DEVICES=${gpuid} "${cmd[@]}"
}

if [ "$MODE" = "test" ] && [ "${#TEST_CHECKPOINT_EPOCHS[@]}" -gt 0 ]; then
  for TEST_CHECKPOINT_EPOCH in "${TEST_CHECKPOINT_EPOCHS[@]}"; do
    local_checkpoint_path="${CPt_BASE_PATH}/model_${TEST_CHECKPOINT_EPOCH}.rar"
    if [ ! -f "${local_checkpoint_path}" ]; then
      echo "Skipping missing checkpoint: ${local_checkpoint_path}"
      continue
    fi
    execute_with_args "${CMD_ARGS_BASE[@]}" "--test-only" "--checkpoint-path" "${local_checkpoint_path}"
  done
else
  execute_with_args "${CMD_ARGS_BASE[@]}" "${MODE_FLAG_ARGS[@]}"
fi
