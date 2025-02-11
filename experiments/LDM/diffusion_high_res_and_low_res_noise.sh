#!/bin/sh

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  /nfs/home/apatel/resolution_invariant_AE/train_ddpm.py \
  --output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT_LDM/ \
  --model_name=ddpm_hr_50_sample_rest_lr_noise \
  --AE_checkpoint=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT/AE_KL_scaled_varied_noise_8_channels/checkpoint.pth \
  --training_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered \
  --validation_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_validation  \
  --is_grayscale=1 \
  --n_epochs=6000 \
  --batch_size=1 \
  --eval_freq=10 \
  --checkpoint_every=500 \
  --cache_data=1  \
  --prediction_type=epsilon \
  --model_type=small \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=3 \
  --noise_latents=1 \
  --multi_resolution=1 \
  --multi_resolution_perc=0.2



