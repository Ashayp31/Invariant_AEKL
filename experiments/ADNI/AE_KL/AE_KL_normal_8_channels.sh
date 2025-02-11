#!/bin/sh

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    /nfs/home/apatel/resolution_invariant_AE/train_ae_kl.py \
    --training_ids=/nfs/home/apatel/Data/ADNI1/bias_corrected_raw_resampled \
		--validation_ids=/nfs/home/apatel/Data/ADNI1/bias_corrected_raw_resampled_val \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/ADNI/ \
    --model_name=AE_KL_normal_8_channels_ADNI_v3 \
    --is_grayscale=1 \
    --batch_size=1 \
    --geometric_augmentations=1 \
    --resolution_invariant=0 \
    --trainer_type=normal \
    --latent_channels=8 \
    --normalise_intensity=1


