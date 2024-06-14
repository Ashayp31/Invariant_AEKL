#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/resolution_invariant_AE/train_ae_kl.py \
    --training_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_short_tester \
		--validation_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_validation \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT/ \
    --model_name=AE_KL_test_v3 \
    --is_grayscale=1 \
    --batch_size=3 \
    --geometric_augmentations=0


