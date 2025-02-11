#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/resolution_invariant_AE/test_classifier.py \
    --training_ids=/nfs/home/apatel/Data/ADNI1/bias_corrected_raw_testing \
		--validation_ids=/nfs/home/apatel/Data/ADNI1/bias_corrected_raw_testing \
		--labels_path=/nfs/home/apatel/Data/ADNI1/labels.csv \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/ADNI/ \
    --model_name=AE_invariant_classification_mixed_res \
    --AE_checkpoint=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/ADNI/AE_KL_scaled_varied_noise_8_channels_ADNI_v2/checkpoint_600.pth \
    --model_checkpoint=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/ADNI/AE_invariant_classification_mixed_res/checkpoint_60.pth \
    --batch_size=1 \
    --latent_channels=8 \
    --normalise_intensity=1 \
    --fixed_res_resample=2.4 \
    --multi_resolution=1 \
