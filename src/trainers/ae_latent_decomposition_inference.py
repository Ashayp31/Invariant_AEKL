import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from src.networks import Resolution_Invariant_AE

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.invariant_validation_dataloader import get_invariant_validation_loader
import nibabel as nib
import gc

class AEKLDecompInferer:
    def __init__(self, args):

        # initialise DDP if run was launched with torchrun
        if "LOCAL_RANK" in os.environ:
            print("Setting up DDP.")
            self.ddp = True
            # disable logging for processes except 0 on every node
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f

            # initialize the distributed training process, every GPU runs in a process
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.ddp = False
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        print(f"Arguments: {str(args)}")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        self.output_dir = args.output_dir + "/" + args.model_name + "/" + "output"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)


        self.resolution_to_noise_dict = {0: 0,
                            0.1:     0.054566325692216,
                            0.15:    0.058487351122371,
                            0.2:     0.068105017191952,
                            0.3:     0.078441331427086,
                            0.4:     0.086617040844872,
                            0.6:     0.096673193631745,
                            0.8:     0.101024626451242,
                            1:   0.111119358708608,
                            1.2:     0.13625346873782,
                            1.4:     0.144827609443072,
                            1.6:     0.153792271196886,
                            1.8:     0.164495481627112,
                            2:   0.17673307393304,
                            2.4:     0.20202855756054,
                            2.8:     0.227056394667627,
                            3.2:     0.254479690604352,
                            3.6:     0.268402808332848,
                            4:   0.288671072644964,
                            5:   0.328458560932646,
                            6:   0.394685677641798}


        self.resolution_invariant = args.resolution_invariant == 1


        self.network_type = args.trainer_type
        self.num_layers = len(args.num_channels) - 1

        # set up model
        self.spatial_dimension = args.spatial_dimension
        ae_args = {
            "spatial_dims": args.spatial_dimension,
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "num_channels": args.num_channels,
            "latent_channels": args.latent_channels,
            "num_res_blocks": args.num_res_blocks,
            "norm_num_groups": args.norm_num_groups,
            "attention_levels": args.attention_levels,

        }


        self.model = Resolution_Invariant_AE(spatial_dims=args.spatial_dimension,
                                         in_channels=args.in_channels,
                                         out_channels=args.out_channels,
                                         num_channels=args.num_channels,
                                         latent_channels=args.latent_channels,
                                         num_res_blocks=args.num_res_blocks,
                                         norm_num_groups=args.norm_num_groups,
                                         attention_levels=args.attention_levels)
        self.model.to(self.device)

        model_checkpoint_path = args.model_checkpoint
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"Cannot find AE-KL checkpoint {model_checkpoint_path}")

        model_checkpoint = torch.load(model_checkpoint_path)

        self.model.to(self.device)
        self.model.load_state_dict(model_checkpoint["model_state_dict"])
        self.model.eval()

        self.val_loader = get_invariant_validation_loader(
            batch_size=args.batch_size,
            training_ids=args.training_ids,
            validation_ids=args.validation_ids,
            num_workers=args.num_workers,
            cache_data=bool(args.cache_data),
        )

    def calc_noise_to_add(self, low_resolution):
        resolution_scale = low_resolution
        past_value = 0
        for key, value in self.resolution_to_noise_dict.items():
            if resolution_scale < key:
                return past_value
            else:
                past_value = value
        return 0.4

    @torch.no_grad()
    def infer(self):
        progress_bar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            ncols=70,
            position=0,
            leave=True,
            desc="Validation",
        )

        resolution_changes =  [1,1.2,1.6,2,2.4,2.8,3.2,3.6,4]

        sample_num = 0
        for step, batch in progress_bar:
            for resolution_change in resolution_changes:
                if resolution_change == 1:
                    images = batch["image"].to(self.device)
                    resolution = batch["resolution"][0].detach().numpy()
                    resolution_high_res = batch["resolution"][0].detach().numpy()
                    resolution_tensor = batch["resolution_array"].to(self.device)
                    resolution_tensor_high_res = batch["resolution_array"].to(self.device)

                    res_change = ((self.num_layers ** 2) / resolution[0]) ** (1 / self.num_layers)
                    image_dimensions = images.shape[2:]

                    if self.num_layers == 3:
                        intermediate_spatial_dimensions_high_res = [
                            [int(image_dimensions[i]) for i in range(self.spatial_dimension)],
                            [int(image_dimensions[i] / res_change) for i in range(self.spatial_dimension)],
                            [int(image_dimensions[i] / (res_change ** 2)) for i in range(self.spatial_dimension)],
                            [int(image_dimensions[i] / (res_change ** 3)) for i in range(self.spatial_dimension)]]
                    elif self.num_layers == 4:
                        intermediate_spatial_dimensions_high_res = [
                            [int(image_dimensions[i]) for i in range(self.spatial_dimension)],
                            [int(image_dimensions[i] / res_change) for i in range(self.spatial_dimension)],
                            [int(image_dimensions[i] / (res_change ** 2)) for i in range(self.spatial_dimension)],
                            [int(image_dimensions[i] / (res_change ** 3)) for i in range(self.spatial_dimension)],
                            [int(image_dimensions[i] / (res_change ** 4)) for i in range(self.spatial_dimension)]]


                else:
                    images = batch["image_res_" + str(resolution_change)].to(self.device)
                    resolution = batch["resolution_res_" + str(resolution_change)][0].detach().numpy()
                    resolution_tensor = batch["resolution_array_res_" + str(resolution_change)].to(self.device)

                res_change = ((self.num_layers ** 2) / resolution[0]) ** (1 / self.num_layers)
                image_dimensions = images.shape[2:]

                if self.num_layers == 3:
                    intermediate_spatial_dimensions_low_res = [
                        [int(image_dimensions[i]) for i in range(self.spatial_dimension)],
                        [int(image_dimensions[i] / res_change) for i in range(self.spatial_dimension)],
                        [int(image_dimensions[i] / (res_change ** 2)) for i in range(self.spatial_dimension)],
                        [int(image_dimensions[i] / (res_change ** 3)) for i in range(self.spatial_dimension)]]
                elif self.num_layers == 4:
                    intermediate_spatial_dimensions_low_res = [
                        [int(image_dimensions[i]) for i in range(self.spatial_dimension)],
                        [int(image_dimensions[i] / res_change) for i in range(self.spatial_dimension)],
                        [int(image_dimensions[i] / (res_change ** 2)) for i in range(self.spatial_dimension)],
                        [int(image_dimensions[i] / (res_change ** 3)) for i in range(self.spatial_dimension)],
                        [int(image_dimensions[i] / (res_change ** 4)) for i in range(self.spatial_dimension)]]


                resolution_scale = resolution - 1
                noise_to_add = self.calc_noise_to_add(resolution_scale)


                # Normal Reconstruction
                reconstruction_lr, _, _, _ = self.model(images, intermediate_spatial_dimensions_low_res,
                                            intermediate_spatial_dimensions_low_res, noise_to_add, add_noise=False)

                reconstruction_lr_added_noise, _, _, _ = self.model(images, intermediate_spatial_dimensions_low_res,
                                                                    intermediate_spatial_dimensions_low_res, noise_to_add,
                                                                    add_noise=True)

                reconstruction_lr_to_hr_no_added_noise, z_no_added_noise, _, _ = self.model(images, intermediate_spatial_dimensions_low_res,
                                                              intermediate_spatial_dimensions_high_res, noise_to_add,
                                                              add_noise=False)

                reconstruction_lr_to_hr_1, z, z_mu, z_sigma = self.model(images, intermediate_spatial_dimensions_low_res,
                                                              intermediate_spatial_dimensions_high_res, noise_to_add,
                                                              add_noise=True)

                recon_lr = reconstruction_lr[0, 0]
                recon_lr_added_noise = reconstruction_lr_added_noise[0, 0]
                reconstruction_lr_to_hr_1 = reconstruction_lr_to_hr_1[0, 0]
                reconstruction_lr_to_hr_no_added_noise = reconstruction_lr_to_hr_no_added_noise[0, 0]
                orig_image = images[0,0]


                qform_lr = [[resolution[0], 0, 0, 0],
                             [0, resolution[0], 0, 0],
                             [0, 0, resolution[0], 0],
                             [0, 0, 0, 1]]

                qform_hr = [[resolution_high_res[0], 0, 0, 0],
                             [0, resolution_high_res[0], 0, 0],
                             [0, 0, resolution_high_res[0], 0],
                             [0, 0, 0, 1]]

                orig_image = orig_image.cpu().numpy()
                orig_image_nii = nib.Nifti1Image(orig_image, affine=qform_lr)

                recon_lr = recon_lr.cpu().numpy()
                recon_nii = nib.Nifti1Image(recon_lr, affine=qform_lr)

                recon_lr_added_noise = recon_lr_added_noise.cpu().numpy()
                recon_added_noise_nii = nib.Nifti1Image(recon_lr_added_noise, affine=qform_lr)

                reconstruction_lr_to_hr_1 = reconstruction_lr_to_hr_1.cpu().numpy()
                reconstruction_lr_to_hr_1_nii = nib.Nifti1Image(reconstruction_lr_to_hr_1, affine=qform_hr)

                reconstruction_lr_to_hr_no_added_noise = reconstruction_lr_to_hr_no_added_noise.cpu().numpy()
                reconstruction_lr_to_hr_no_added_noise_nii = nib.Nifti1Image(reconstruction_lr_to_hr_no_added_noise, affine=qform_hr)


                nib.save(orig_image_nii, self.output_dir + "/sample_" +str(sample_num) + "_orig_image_" + str(resolution_change).replace(".", "_") + ".nii.gz")
                nib.save(recon_nii, self.output_dir + "/sample_" +str(sample_num) + "_recon_lr_res_" + str(resolution_change).replace(".", "_") + ".nii.gz")
                nib.save(recon_added_noise_nii, self.output_dir + "/sample_" +str(sample_num) + "_recon_lr_added_noise_res_" + str(resolution_change).replace(".", "_") + ".nii.gz")
                nib.save(reconstruction_lr_to_hr_1_nii, self.output_dir + "/sample_" +str(sample_num) + "_recon_lr_to_hr_v1_res_" + str(resolution_change).replace(".", "_") + ".nii.gz")
                nib.save(reconstruction_lr_to_hr_no_added_noise_nii, self.output_dir + "/sample_" +str(sample_num) + "_recon_lr_to_hr_no_noise_res_" + str(resolution_change).replace(".", "_") + ".nii.gz")

        return
