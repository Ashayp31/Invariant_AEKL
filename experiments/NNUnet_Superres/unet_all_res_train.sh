#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/resolution_invariant_AE/train_nnunet_superres.py \
    --training_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered \
		--validation_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_short_tester \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT/ \
    --model_name=Unet_all_resolutions \
    --resolution_multiplier=4