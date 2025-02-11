from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep


def pixel_shuffle_3d(input_tensor, upscale_factor):
    """
    Rearranges elements in a 3D tensor of shape (N, C, D, H, W) to (N, C/(r^3), D*r, H*r, W*r)

    Parameters:
    - input_tensor: numpy array of shape (N, C, D, H, W)
    - upscale_factor: int, the scaling factor r

    Returns:
    - output_tensor: numpy array of shape (N, C//(r^3), D*r, H*r, W*r)
    """
    N, C, D, H, W = input_tensor.shape
    r = upscale_factor

    if C % (r ** 3) != 0:
        raise ValueError("Number of channels must be divisible by upscale_factor^3")

    # Reshape to (N, C//(r^3), r, r, r, D, H, W)
    output = input_tensor.reshape(N, C // (r ** 3), r, r, r, D, H, W)

    # Permute to (N, C//(r^3), D, r, H, r, W, r)
    output = output.transpose(0, 1, 5, 2, 6, 3, 7, 4)

    # Reshape to (N, C//(r^3), D*r, H*r, W*r)
    output = output.reshape(N, C // (r ** 3), D * r, H * r, W * r)

    return output


class ChannelDuplicatingPixelShuffleUpSampleLayer3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        """
        3D version of Channel-Duplicating Pixel Shuffle for upsampling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            factor (int): Upscaling factor (for depth, height, and width).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor

        # Ensure output channels are consistent after upsampling
        assert out_channels * factor ** 3 % in_channels == 0, "Invalid in_channels, out_channels, or factor"
        self.repeats = out_channels * factor ** 3 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D Channel-Duplicating Pixel Shuffle Upsampling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, D*factor, H*factor, W*factor)
        """
        # Duplicate channels to match required output channels
        x = x.repeat_interleave(self.repeats, dim=1)

        # Apply 3D pixel shuffle for spatial upsampling
        x = pixel_shuffle_3d(x, self.factor)

        return x


def pixel_unshuffle_3d(input_tensor, downscale_factor):
    """
    Rearranges elements in a 3D tensor of shape (N, C, D, H, W) to (N, C*(r^3), D//r, H//r, W//r)

    Parameters:
    - input_tensor: numpy array of shape (N, C, D, H, W)
    - downscale_factor: int, the scaling factor r

    Returns:
    - output_tensor: numpy array of shape (N, C*(r^3), D//r, H//r, W//r)
    """
    N, C, D, H, W = input_tensor.shape
    r = downscale_factor

    if D % r != 0 or H % r != 0 or W % r != 0:
        raise ValueError("Depth, Height, and Width must be divisible by downscale_factor")

    # Reshape to (N, C, D//r, r, H//r, r, W//r, r)
    output = input_tensor.reshape(N, C, D // r, r, H // r, r, W // r, r)

    # Permute to (N, C, r, r, r, D//r, H//r, W//r)
    output = output.transpose(0, 1, 3, 5, 7, 2, 4, 6)

    # Reshape to (N, C * (r^3), D//r, H//r, W//r)
    output = output.reshape(N, C * (r ** 3), D // r, H // r, W // r)

    return output



class PixelUnshuffleChannelAveragingDownSampleLayer3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        """
        3D version of PixelUnshuffle with Channel Averaging.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            factor (int): Downscaling factor (for depth, height, and width).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor

        # Ensure the number of input channels is divisible by the output channels after unshuffle
        assert in_channels * factor ** 3 % out_channels == 0, "Invalid in_channels, out_channels, or factor"
        self.group_size = in_channels * factor ** 3 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D Pixel Unshuffle with Channel Averaging.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, D//factor, H//factor, W//factor)
        """
        x = pixel_unshuffle_3d(x, self.factor)  # 3D pixel unshuffle
        B, C, D, H, W = x.shape

        # Reshape to split channels into groups
        x = x.view(B, self.out_channels, self.group_size, D, H, W)

        # Average over the group dimension
        x = x.mean(dim=2)

        return x