import argparse
import ast
from typing import Tuple, Union, Sequence

from src.trainers import AEKLTrainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument(
        "--spatial_dimension", default=3, type=int, help="Dimension of images: 2d or 3d."
    )
    parser.add_argument("--image_size", default=None, help="Resize images.")
    parser.add_argument(
        "--image_roi",
        default=None,
        help="Specify central ROI crop of inputs, as a tuple, with -1 to not crop a dimension.",
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--resolution_invariant",
        default=0,
        help="Use resolution invariant network, 1 (True) or 0 (False).",
        type=ast.literal_eval,
    )

    # model params
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=1, type=int)

    parser.add_argument("--num_channels", default=(32, 64, 128, 256), type=ast.literal_eval)
    parser.add_argument(
        "--num_res_blocks", default=2, type=ast.literal_eval
    )
    parser.add_argument("--latent_channels", default=8, type=int)

    parser.add_argument("--norm_num_groups", default=16, type=float)
    parser.add_argument("--attention_levels", default=(False, False, False, False), type=Sequence[bool])
    # parser.add_argument("--ae_ddp_sync", default=True, type=bool)

    # training param
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of epochs to train.")
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5,
        help="Number of epochs to between evaluations.",
    )
    parser.add_argument(
        "--augmentation",
        type=int,
        default=1,
        help="Use of augmentation, 1 (True) or 0 (False).",
    )
    parser.add_argument(
        "--geometric_augmentations",
        type=int,
        default=0,
        help="Whether to apply cropping and resizing augmentations"
    )
    parser.add_argument(
        "--adversarial_warmup",
        type=int,
        default=20,
        help="Warmup the learning rate of the adversarial component.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--cache_data",
        type=int,
        default=0,
        help="Whether or not to cache data in dataloaders.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save a checkpoint every checkpoint_every epochs.",
    )
    parser.add_argument("--is_grayscale", type=int, default=0, help="Is data grayscale.")
    parser.add_argument(
        "--quick_test",
        default=0,
        type=int,
        help="If True, runs through a single batch of the train and eval loop.",
    )
    args = parser.parse_args()
    return args


# to run using DDP, run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  train_ddpm.py --args
if __name__ == "__main__":
    args = parse_args()
    trainer = AEKLTrainer(args)
    trainer.train(args)
