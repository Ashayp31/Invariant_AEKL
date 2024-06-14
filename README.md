# vqvae
Codebase for VQ-VAE based methods

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
$ nvidia-docker run -it --user=$(id -u):$(id -g) --shm-size=32g -v ${PROJECT_ROOT}:/project -v ${DATA_DIR}:/data vqvae-image

# Example
# $ nvidia-docker run -it --user=$(id -u):$(id -g) --shm-size=32g -v /home/u00u7bsr6oxNorIJxM357/brain_synthesis_transformers:/project -v /shared/data/c1project0/Biobank/:/data vqvae-image
```
Note: Define shared memory size according to your resources.

### 3. Create .tsv files
Before training, create the .tsv files containing the list of paths of the training and validation set using create_train_val_split.py.
```
# For example:
$ python /project/create_train_val_split.py \
    --data_dir /data \
    --output_dir /project/outputs \
    --n_val 2 \
    --n_train 8
```

### 4. Train transformer
To train the vq-vae
```
# For example:
$ python /project/train_vqvae_3d.py \
    --training_subjects="/path/to/training/data/" \
    --validation_subjects="/path/to/validation/data/" \
    --project_directory="/path/to/project/directory/" \
    --experiment_name="example_run" \
    --amp="True"
```

### 5. Train transformer
To train the transformer, specify the model_uri of the previously trained VQ-VAE.
```
# For example:
$ python /project/train_transformer_3d.py \
    --seed=2 \
    --run_dir="performer" \
    --train_list="/project/outputs/train_paths.tsv" \
    --val_list="/project/outputs/val_paths.tsv" \
    --input_height=24 \
    --input_width=32 \
    --input_depth=24 \
    --n_embd=128 \
    --n_layers=10 \
    --n_head=8 \
    --emb_dropout=0 \
    --ff_dropout=0 \
    --attn_dropout=0 \
    --vqvae_file="/project/outputs/checkpoint_epoch=500.pt" \
    --vqvae_from_checkpoint="True" \
    --vqvae_use_subpixel_conv=False \
    --vqvae_no_levels=3 \
    --vqvae_no_res_layers=1 \
    --vqvae_no_channels=192 \
    --vqvae_codebook_type="ema" \
    --vqvae_num_embeddings=32 \
    --vqvae_embedding_dim=64 \
    --vqvae_dropout=0.1 \
    --vqvae_act="RELU" \
    --batch_size=1 \
    --lr=0.001 \
    --lr_decay=0.9999 \
    --n_epochs=150 \
    --eval_freq=5 \
    --num_workers=4
```

### 6. Sample brain 
```
# For example:
$ python /project/sample_vqvae_3d.py \
    --seed 2 \
    --transformer_file="/project/outputs/performer/checkpoint_epoch=25.pt" \
    --transformer_from_checkpoint="True" \
    --transformer_input_height=24 \
    --transformer_input_width=32 \
    --transformer_input_depth=24 \
    --transformer_n_embd=128 \
    --transformer_n_layers=10 \
    --transformer_n_head=8 \
    --transformer_emb_dropout=0 \
    --transformer_ff_dropout=0 \
    --transformer_attn_dropout=0 \
    --vqvae_file="/project/outputs/checkpoint_epoch=500.pt" \
    --vqvae_from_checkpoint="True" \
    --vqvae_use_subpixel_conv=False \
    --vqvae_no_levels=3 \
    --vqvae_no_res_layers=1 \
    --vqvae_no_channels=192 \
    --vqvae_codebook_type="ema" \
    --vqvae_num_embeddings=32 \
    --vqvae_embedding_dim=64 \
    --vqvae_dropout=0.1 \
    --vqvae_act="RELU"
```
