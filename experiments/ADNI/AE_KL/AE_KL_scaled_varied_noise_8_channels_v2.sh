#!/bin/sh

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    /nfs/home/apatel/resolution_invariant_AE/train_ae_kl.py \
    --training_ids=/nfs/home/apatel/Data/ADNI1/bias_corrected_raw_resampled \
		--validation_ids=/nfs/home/apatel/Data/ADNI1/bias_corrected_raw_resampled_val \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/ADNI/ \
    --model_name=AE_KL_scaled_varied_noise_8_channels_ADNI_v2_000001kl \
    --is_grayscale=1 \
    --batch_size=1 \
    --geometric_augmentations=1 \
    --resolution_invariant=1 \
    --trainer_type=specific_noise_forced_decomp \
    --latent_channels=8 \
    --normalise_intensity=1 \
    --kl_weight=0.000001


