# Resolution Invariant AutoEncoder
Codebase for a resolution Invariant Autoencoder

### 1. Create Docker image
```
# Build the docker image
$ docker build -t vqvae-image . \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}
```
Note: --build-arg defined to avoid docker container to have root privileges and save files with diffferent permission. 

### 2. Running image using -it
```
$ nvidia-docker run -it --user=$(id -u):$(id -g) --shm-size=32g -v ${PROJECT_ROOT}:/project -v ${DATA_DIR}:/data ae-image
```

### 3. Train Resolution Invariant AE
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


### 4. Run Super-Resolution
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
