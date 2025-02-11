#!/bin/sh

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    /nfs/home/apatel/resolution_invariant_AE/train_ae_kl.py \
    --training_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered \
		--validation_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_short_tester \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/CT/ \
    --model_name=AE_KL_scaled_varied_noise_16_channels_4_layers_v2 \
    --is_grayscale=1 \
    --batch_size=1 \
    --geometric_augmentations=1 \
    --resolution_invariant=1 \
    --trainer_type=specific_noise_forced_decomp \
    --latent_channels=16


