#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/resolution_invariant_AE/run_ddpm.py \
    --training_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_short_tester \
		--validation_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_short_tester \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT_LDM/ \
		--AE_checkpoint=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT/AE_KL_scaled_varied_noise_8_channels/checkpoint.pth \
    --model_name=ddpm_hr_50_samples \


