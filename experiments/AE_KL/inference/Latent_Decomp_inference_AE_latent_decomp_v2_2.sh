#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/resolution_invariant_AE/run_ae_kl_invariant.py \
    --training_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered \
		--validation_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT/ \
		--model_checkpoint=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT/AE_KL_specific_noise_foreced_decomp_8_channels_varied_noise/checkpoint.pth \
    --model_name=AE_KL_specific_noise_foreced_decomp_8_channels_varied_noise \
    --is_grayscale=1 \
    --resolution_invariant=1 \
    --batch_size=1 \
    --geometric_augmentations=1 \
    --inference_type=decomp \
    --trainer_type=specific_noise_forced_decomp_v2 \
    --latent_channels=8

