#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/resolution_invariant_AE/run_nnunet_superres.py \
    --training_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_short_tester \
		--validation_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_short_tester \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT/ \
		--checkpoint_path=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT/Unet_4x_v2/checkpoint_200.pth \
    --model_name=Unet_4x_v2 \
    --resolution_multiplier=4


