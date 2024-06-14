from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from numpy import fromfunction
from torch.fft import fftn
from torch.nn.modules.loss import _Loss
from torch.nn.functional import kl_div
from src.handlers.general import TBSummaryTypes
import warnings
from collections import OrderedDict



class PerceptualLoss(nn.Module):
    """
    Perceptual loss using features from pretrained deep neural networks trained. The function supports networks
    pretrained on: ImageNet that use the LPIPS approach from Zhang, et al. "The unreasonable effectiveness of deep
    features as a perceptual metric." https://arxiv.org/abs/1801.03924 ; RadImagenet from Mei, et al. "RadImageNet: An
    Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"
    https://pubs.rsna.org/doi/full/10.1148/ryai.210315 ; MedicalNet from Chen et al. "Med3D: Transfer Learning for
    3D Medical Image Analysis" https://arxiv.org/abs/1904.00625 ;
    and ResNet50 from Torchvision: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html .

    The fake 3D implementation is based on a 2.5D approach where we calculate the 2D perceptual on slices from the
    three axis.

    Args:
        spatial_dims: number of spatial dimensions.
        network_type: {``"alex"``, ``"vgg"``, ``"squeeze"``, ``"radimagenet_resnet50"``,
        ``"medicalnet_resnet10_23datasets"``, ``"medicalnet_resnet50_23datasets"``, ``"resnet50"``}
            Specifies the network architecture to use. Defaults to ``"alex"``.
        is_fake_3d: if True use 2.5D approach for a 3D perceptual loss.
        fake_3d_ratio: ratio of how many slices per axis are used in the 2.5D approach.
        cache_dir: path to cache directory to save the pretrained network weights.
        pretrained: whether to load pretrained weights. This argument only works when using networks from
            LIPIS or Torchvision. Defaults to ``"True"``.
        pretrained_path: if `pretrained` is `True`, users can specify a weights file to be loaded
            via using this argument. This argument only works when ``"network_type"`` is "resnet50".
            Defaults to `None`.
        pretrained_state_dict_key: if `pretrained_path` is not `None`, this argument is used to
            extract the expected state dict. This argument only works when ``"network_type"`` is "resnet50".
            Defaults to `None`.
    """

    def __init__(
        self,
        spatial_dims: int,
        network_type: str = "alex",
        is_fake_3d: bool = True,
        fake_3d_ratio: float = 0.5,
        cache_dir: str = None,
        pretrained: bool = True,
    ):
        super().__init__()

        if spatial_dims not in [2, 3]:
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        if (spatial_dims == 2 or is_fake_3d) and "medicalnet_" in network_type:
            raise ValueError(
                "MedicalNet networks are only compatible with ``spatial_dims=3``."
                "Argument is_fake_3d must be set to False."
            )

        if cache_dir:
            torch.hub.set_dir(cache_dir)

        self.spatial_dims = spatial_dims
        self.perceptual_function = LPIPS(pretrained=pretrained, net=network_type, verbose=False)
        self.is_fake_3d = is_fake_3d
        self.fake_3d_ratio = fake_3d_ratio
        self.summaries = Dict = {TBSummaryTypes.SCALAR: dict()}

    def _calculate_axis_loss(self, input: torch.Tensor, target: torch.Tensor, spatial_axis: int) -> torch.Tensor:
        """
        Calculate perceptual loss in one of the axis used in the 2.5D approach. After the slices of one spatial axis
        is transformed into different instances in the batch, we compute the loss using the 2D approach.

        Args:
            input: input 5D tensor. BNHWD
            target: target 5D tensor. BNHWD
            spatial_axis: spatial axis to obtain the 2D slices.
        """

        def batchify_axis(x: torch.Tensor, fake_3d_perm: tuple) -> torch.Tensor:
            """
            Transform slices from one spatial axis into different instances in the batch.
            """
            slices = x.float().permute((0,) + fake_3d_perm).contiguous()
            slices = slices.view(-1, x.shape[fake_3d_perm[1]], x.shape[fake_3d_perm[2]], x.shape[fake_3d_perm[3]])

            return slices

        preserved_axes = [2, 3, 4]
        preserved_axes.remove(spatial_axis)

        channel_axis = 1
        input_slices = batchify_axis(x=input, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        indices = torch.randperm(input_slices.shape[0])[: int(input_slices.shape[0] * self.fake_3d_ratio)].to(
            input_slices.device
        )
        input_slices = torch.index_select(input_slices, dim=0, index=indices)
        target_slices = batchify_axis(x=target, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        target_slices = torch.index_select(target_slices, dim=0, index=indices)

        axis_loss = torch.mean(self.perceptual_function(input_slices, target_slices))

        return axis_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D].
            target: the shape should be BNHW[D].
        """
        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        if self.spatial_dims == 3 and self.is_fake_3d:
            # Compute 2.5D approach
            loss_sagittal = self._calculate_axis_loss(input, target, spatial_axis=2)
            loss_coronal = self._calculate_axis_loss(input, target, spatial_axis=3)
            loss_axial = self._calculate_axis_loss(input, target, spatial_axis=4)
            loss = loss_sagittal + loss_axial + loss_coronal
        else:
            # 2D and real 3D cases
            loss = self.perceptual_function(input, target)

        self.summaries[TBSummaryTypes.SCALAR]["Perceptual_loss"] = loss


        return torch.mean(loss)

class LossVAE(_Loss):
    def __init__(
        self, weight=None, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super(LossVAE, self).__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.summaries = Dict = {TBSummaryTypes.SCALAR: dict()}
        self._weight = weight
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        reconstruction = outputs["reconstruction"][0]
        z_mu = outputs["z_mu"]
        z_sigma = outputs["z_sigma"]


        l1_loss = F.l1_loss(reconstruction, y, reduction=self.reduction)

        kl_loss = (4/212) * torch.mean(-0.5 * torch.sum(1 + z_sigma - z_mu ** 2 - z_sigma.exp(), dim = 1), dim = 0)

        self.summaries[TBSummaryTypes.SCALAR]["Loss"] = l1_loss + kl_loss
        self.summaries[TBSummaryTypes.SCALAR]["L1-Recon-Loss"] = l1_loss
        self.summaries[TBSummaryTypes.SCALAR]["KL-Loss"] = kl_loss


        return l1_loss + kl_loss


    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries


class LossAESpatial(_Loss):
    def __init__(
        self, weight=None, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super(LossAESpatial, self).__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.summaries = Dict = {TBSummaryTypes.SCALAR: dict()}
        self._weight = weight
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        reconstruction = outputs["reconstruction"][0]
        l1_loss = F.l1_loss(reconstruction, y, reduction=self.reduction)
        self.summaries[TBSummaryTypes.SCALAR]["Loss"] = l1_loss
        return l1_loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries


class LossAEDense(_Loss):
    def __init__(
        self, weight=None, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super(LossAEDense, self).__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.summaries = Dict = {TBSummaryTypes.SCALAR: dict()}
        self._weight = weight
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        reconstruction = outputs["reconstruction"][0].float()

        l1_loss = F.l1_loss(reconstruction, y, reduction=self.reduction)
        self.summaries[TBSummaryTypes.SCALAR]["Loss"] = l1_loss
        return l1_loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries



def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    outputs,
    Y,
    data_range=1,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """

    X = outputs["reconstruction"][0].float()
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    outputs, Y, data_range=1, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):

    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """

    X = outputs["reconstruction"][0].float()
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=1,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=1,
        spatial_dims=3,
        K=(0.01, 0.4),
        nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.summaries = Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(self, X, Y):

        ssim_loss = 1 - ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )
        self.summaries[TBSummaryTypes.SCALAR]["Loss"] = ssim_loss
        return ssim_loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=1,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=1,
        spatial_dims=3,
        weights=None,
        K=(0.01, 0.03),
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.summaries = Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(self, X, Y):
        msssim_loss = 1- ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )
        self.summaries[TBSummaryTypes.SCALAR]["Loss"] = msssim_loss
        return msssim_loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries
