#!/bin/bash
# Train IDMVAE on TCGA (5 splits). Override DATADIR / CHECKPOINT_DIR / hyperparameters as needed.
# Run from ccadiffusion/src:  bash commands/run_TCGA_experiment.sh
# cd "$(dirname "$0")/.." || exit 1

DATE=$(date +%m%d%Y)

EXPERIMENT=TCGA_IDMVAE_release
PRIORPOSTERIOR=Normal
# D.2 TCGA: diffusion weight search {0.1,1,10}; best 0.1. Paper λ₁→cross-MI, λ₂→gen-aug (same order as objectives.py).
DLW=0.1
LIKELIHOOD=Laplace
BATCH_SIZE=128
EPOCHS=50
LATENT_DIM_W=32
LATENT_DIM_Z=16
BETA=2.5
CROSS_MI_SCALE=10.0
AUG_MI_SCALE=0.001
AUG_MI_SAMPLING_SCHEME="posterior"
AUG_MI_LT="CL"
DATADIR="${DATADIR:-/data/backed_up/shared/Data/TCGA/FINAL/complete_splits}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/backed_up/shared/Data/TCGA/ckpts/}"

for SPLIT in 0 1 2 3 4; do
     RUN_ID=${DATE}_idmvae_pp_${PRIORPOSTERIOR}_l_${LIKELIHOOD}_b_${BETA}_augmi_${AUG_MI_SAMPLING_SCHEME}_${AUG_MI_SCALE}_crossmi_${CROSS_MI_SCALE}_dlw_${DLW}_w_${LATENT_DIM_W}_z_${LATENT_DIM_Z}_split${SPLIT}
     python train_IDMVAE_TCGA.py \
          --experiment $EXPERIMENT \
          --priorposterior $PRIORPOSTERIOR \
          --diffusion_loss_weight $DLW \
          --likelihood $LIKELIHOOD \
          --beta $BETA \
          --batch_size $BATCH_SIZE \
          --epochs $EPOCHS \
          --latent_dim_w $LATENT_DIM_W \
          --latent_dim_z $LATENT_DIM_Z \
          --gen_aug_loss_scale $AUG_MI_SCALE \
          --cross_mi_loss_scale $CROSS_MI_SCALE \
          --gen_aug_sampling_scheme $AUG_MI_SAMPLING_SCHEME \
          --gen_aug_loss_type $AUG_MI_LT \
          --datadir $DATADIR \
          --split $SPLIT \
          --runId $RUN_ID \
          --checkpoint_dir $CHECKPOINT_DIR
done
