from typing import List, Any, Tuple

import torch
from monai.inferers import Inferer, SlidingWindowInferer

from src.networks.autoencoders.AE_KL import AutoencoderKL



class VAE_Inferer(Inferer):
    def __init__(self) -> None:
        Inferer.__init__(self)

    def __call__(self, inputs: torch.Tensor, network: AutoencoderKL, *args: Any, **kwargs: Any):

        reconstruction = network.reconstruct(inputs)
        outputs = {"reconstruction": reconstruction}
        return outputs


class Sliding_Window_Inferer(Inferer):
    def __init__(self, window_size : Tuple[int, int, int], device:any) -> None:
        Inferer.__init__(self)
        self.window_size = window_size
        self.device = device

    def __call__(self, inputs: torch.Tensor, network: AutoencoderKL, *args: Any, **kwargs: Any):
        sliding_window_inf = SlidingWindowInferer(roi_size=self.window_size)

        reconstruction = sliding_window_inf(inputs=inputs, network=network)
        recon = reconstruction["reconstruction"][0]
        print(recon.shape)
        return recon


