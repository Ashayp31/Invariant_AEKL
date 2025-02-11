import argparse
import ast
from typing import Tuple, Union, Sequence

from src.trainers import AEKLSuperresInferer, AEKLUncertaintyInferer



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--model_checkpoint", type=str, help="Location of model checkpoint")
    parser.add_argument(
        "--spatial_dimension", default=3, type=int, help="Dimension of images: 2d or 3d."
    )

    # model params
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=1, type=int)

    parser.add_argument("--num_channels", default=(32, 64, 128, 256), type=ast.literal_eval)
    parser.add_argument(
        "--num_res_blocks", default=2, type=ast.literal_eval
    )
    parser.add_argument("--latent_channels", default=8, type=int)

    parser.add_argument("--norm_num_groups", default=8, type=float)
    parser.add_argument("--attention_levels", default=(False, False, False, False), type=Sequence[bool])
    # training param
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of epochs to train.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--cache_data",
        type=int,
        default=0,
        help="Whether or not to cache data in dataloaders.",
    )
    parser.add_argument("--is_grayscale", type=int, default=0, help="Is data grayscale.")
    parser.add_argument(
        "--quick_test",
        default=0,
        type=int,
        help="If True, runs through a single batch of the train and eval loop.",
    )
    parser.add_argument(
        "--inference_type",
        default="superres",
        type=str,
        help="If decomp runs through varying resolutions and saves original resolution recons and low res to high res recons"
             "as well as saving latents, normal mode runs just one high res and one low res recon",
    )
    parser.add_argument(
        "--trainer_type",
        default="normal",
        type=str,
        help="Train with resolution consistency, normal or latent decomposition"
    )
    args = parser.parse_args()
    return args


# to run using DDP, run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  train_ddpm.py --args
if __name__ == "__main__":
    args = parse_args()
    if args.inference_type == "superres":
        inferer = AEKLSuperresInferer(args)
    else:
        inferer = AEKLUncertaintyInferer(args)

    inferer.infer()
