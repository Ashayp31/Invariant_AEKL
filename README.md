# Resolution Invariant AutoEncoder
Codebase for a resolution Invariant Autoencoder


### 1. Train Resolution Invariant AE
To train the resolution invariant AE
```
# For example:
$ python /project/train_ae_kl.py \
    --training_ids=/path/to/training/data/ \
    --validation_ids=/path/to/validation/data/ \
    --output_dir=/path/to/project/directory/ \
    --model_name=example_run \
    --is_grayscale=1 \
    --batch_size=1 \
    --geometric_augmentations=1 \
    --resolution_invariant=1 \
    --trainer_type=specific_noise_forced_decomp \
    --latent_channels=8
```

Default settings and as used in experiments are given as in train_ae_kl.py file


### 2. Run Super-Resolution
```
# For example:
$ python /project/run_ae_kl_invariant.py \
    --training_ids=/path/to/training/data/ \
    --validation_ids=/path/to/validation/data/ \
    --output_dir=/path/to/project/directory/ \
    --model_name=example_run \
		--model_checkpoint=path/to/model/checkpoint.pth \
    --is_grayscale=1 \
    --resolution_invariant=1 \
    --batch_size=1 \
    --geometric_augmentations=1 \
    --inference_type=superres \
    --latent_channels=8
```


### 2. Run Uncertainty of Super-Resolution
```
# For example:
$ python /project/run_ae_kl_invariant.py \
    --training_ids=/path/to/training/data/ \
    --validation_ids=/path/to/validation/data/ \
    --output_dir=/path/to/project/directory/ \
    --model_name=example_run \
    --model_checkpoint=path/to/model/checkpoint.pth \
    --is_grayscale=1 \
    --resolution_invariant=1 \
    --batch_size=1 \
    --geometric_augmentations=1 \
    --inference_type=uncertainty \
    --latent_channels=8
```
