#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/resolution_invariant_AE/train_classifier.py \
    --training_ids=/nfs/home/apatel/Data/ADNI1/bias_corrected_raw_resampled \
		--validation_ids=/nfs/home/apatel/Data/ADNI1/bias_corrected_raw_resampled_val \
		--labels_path=/nfs/home/apatel/Data/ADNI1/labels.csv \
		--output_dir=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/ADNI/ \
    --model_name=AE_normal_classification_mixed_res \
    --AE_checkpoint=/nfs/home/apatel/CT_PET_FDG/Resolution_Invariance/ADNI/AE_KL_normal_8_channels_ADNI_v3/checkpoint_600.pth \
    --batch_size=1 \
    --latent_channels=8 \
    --normalise_intensity=1 \
    --multi_resolution=1 \
    --multi_resolution_perc=0.2 \
    --ae_type=normal \
    --use_file_resolution=1


