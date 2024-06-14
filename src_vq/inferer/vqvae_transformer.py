from typing import List, Any

import copy
import torch
from monai.inferers import Inferer
import numpy as np
from torch.nn import functional as F

from monai.transforms import (
GaussianSmooth
)

from src.networks.vqvae.vqvae import VQVAE
from src.networks.transformers.transformer import TransformerBase


class VQVAETransformerInferer(Inferer):
    def __init__(self, transf_network: TransformerBase, device: Any, threshold: float, use_llmap: bool,
                 llmap: str) -> None:
        Inferer.__init__(self)

        self.transformer_network = transf_network
        self.device = device
        self.threshold = threshold
        self.use_llmap = use_llmap
        self.llmap = llmap

    def __call__(self, inputs: torch.Tensor, network: VQVAE, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        with the transformer model to replace low probability codes from the transformer.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """
        embedding_indices = network.index_quantize(images=inputs)[0]
        latent_shape = embedding_indices.shape
        index_sequence = self.transformer_network.ordering.index_sequence
        revert_ordering = np.argsort(index_sequence)

        embedding_indices = embedding_indices.reshape(embedding_indices.shape[0], -1)
        print(embedding_indices.shape)
        embedding_indices = embedding_indices[:, index_sequence]

        resampled_latent = self.get_anomalies(embedding_indices, latent_shape, revert_ordering, index_sequence)
        recon = network.decode_samples(embedding_indices=resampled_latent.to(self.device).long()).cpu().numpy()

        residuals = np.abs(inputs.cpu().numpy() - recon)

        outputs = {"healed_reconstruction": recon, "residual": residuals}

        return outputs

    @torch.no_grad()
    def get_anomalies(self, zs, latent_shape, rev_ordering, ind_sequencing):

        zs = F.pad(zs, (1, 0), "constant", 16)

        zs_in = zs[:, :-1]
        zs_out = zs[:, 1:]

        logits = self.transformer_network(zs_in)
        probs = F.softmax(logits, dim=-1).cpu()
        selected_probs = torch.gather(probs, 2, zs_out.cpu().unsqueeze(2).long())
        selected_probs = selected_probs.squeeze(2)

        print(selected_probs.shape)
        if self.use_llmap:
            average_map = np.load(self.llmap)
            average_map = torch.from_numpy(average_map)
            print(average_map.shape)
            average_map = average_map.unsqueeze(0)
            print(average_map.shape)
            average_map = average_map[:, ind_sequencing]
            print(average_map.shape)
            print(selected_probs.shape)
            selected_probs = torch.div(selected_probs, average_map)
        mask = (selected_probs.float() < self.threshold).long().squeeze(1)


        #resample_sum = torch.sum(mask)
        #f = open("/nfs/home/apatel/PET_FDG/results/vqgan_suv_15_jp_transfv2/baseline_vqvae/outputs/logging_mask.txt", "w+")
        #f.write("Number of tokens below threshold %d\r\n" % resample_sum)
        #f.close()

        sampled = zs_in.clone().to(self.device)
        number_resampled = 0
        for i in range(zs_in.shape[-1] - 1):
            if mask[:, i].max() == 0:
                continue
            else:
                number_resampled += 1
                logits = self.transformer_network(sampled[:, :i + 1])[:, i, :]
                probs_ = F.softmax(logits, dim=1)
                indexes = torch.multinomial(probs_[:, :-1], 1).squeeze(-1)
                sampled[:, i + 1] = mask[:, i] * indexes.cpu() + (1 - mask[:, i]) * sampled[:, i + 1].cpu()

        logits = self.transformer_network(sampled[:, 1:])[:, -1, :]
        probs_ = F.softmax(logits, dim=1)
        indexes = torch.multinomial(probs_[:, :-1], 1)
        sampled = torch.cat((sampled.cpu(), indexes.cpu()), dim=1)

        sampled = sampled[:, 1:][:, rev_ordering]
        sampled = sampled.reshape(latent_shape)
        sampled = sampled.unsqueeze(0)

        return sampled


class LikelihoodMapInferer(Inferer):
    def __init__(self, transf_network: TransformerBase, device: Any) -> None:
        Inferer.__init__(self)

        self.transformer_network = transf_network
        self.device = device

    def __call__(self, inputs: torch.Tensor, network: VQVAE, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        to ouput the probability maps of the tokens upsampled to the same shape as the original input image.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """

        embedding_indices = network.index_quantize(images=inputs)[0]
        latent_shape = embedding_indices.shape
        index_sequence = self.transformer_network.ordering.index_sequence
        revert_ordering = np.argsort(index_sequence)

        embedding_indices = embedding_indices.reshape(embedding_indices.shape[0], -1)
        embedding_indices = embedding_indices[:, index_sequence]

        likelihood_map = self.get_likelihood_maps(embedding_indices, latent_shape, revert_ordering)

        outputs = {"likelihood_map": likelihood_map}
        return outputs

    @torch.no_grad()
    def get_likelihood_maps(self, zs, latent_shape, rev_ordering):
        zs = F.pad(zs, (1, 0), "constant", 16)

        zs_in = zs[:, :-1]
        zs_out = zs[:, 1:]

        logits = self.transformer_network(zs_in)
        probs = F.softmax(logits, dim=-1).cpu()
        selected_probs = torch.gather(probs, 2, zs_out.cpu().unsqueeze(2).long())
        selected_probs = selected_probs.squeeze(2)

        reordered_mask = copy.deepcopy(selected_probs)
        reordered_mask = reordered_mask[:, rev_ordering]
        reordered_mask = reordered_mask.reshape(latent_shape)
        return reordered_mask

        # for visualising
        # upsampled_mask = copy.deepcopy(selected_probs)
        # upsampled_mask = upsampled_mask[:, rev_ordering]
        # upsampled_mask = upsampled_mask.reshape(latent_shape)
        # upsampled_mask = upsampled_mask.cpu().numpy()
        # upsampled_mask = upsampled_mask.repeat(8, axis=-1).repeat(8, axis=-2).repeat(8, axis=-3)
        # upsampled_mask = np.expand_dims(upsampled_mask, 1)
        #
        # return upsampled_mask


class VQVAETransformerUncertaintyInferer(VQVAETransformerInferer):
    def __init__(self, transf_network: TransformerBase, device: Any, num_passes: int, threshold: float, use_llmap: bool,
                 llmap: str, smoothing: float) -> None:
        VQVAETransformerInferer.__init__(self, transf_network=transf_network, device=device, threshold=threshold,
                                         use_llmap=use_llmap, llmap=llmap)

        self.transformer_network = transf_network
        self.device = device
        self.num_passes = num_passes
        self.threshold = threshold
        self.use_llmap = use_llmap
        self.llmap = llmap
        self.smoothing = smoothing
        self.eps = 1e-2

    def __call__(self, inputs: torch.Tensor, network: VQVAE, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        to ouput the probability maps of the tokens upsampled to the same shape as the original input image.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """
        embedding_indices = network.index_quantize(images=inputs)[0]
        latent_shape = embedding_indices.shape
        index_sequence = self.transformer_network.ordering.index_sequence
        revert_ordering = np.argsort(index_sequence)

        embedding_indices = embedding_indices.reshape(embedding_indices.shape[0], -1)
        embedding_indices = embedding_indices[:, index_sequence]

        resampled_latent = self.get_anomalies(embedding_indices, latent_shape, revert_ordering, index_sequence)
        network.train()
        for i in range(self.num_passes):
            if i == 0:
                recons = network.decode_samples(embedding_indices=resampled_latent.to(self.device).long()).cpu()
            else:
                recon = network.decode_samples(embedding_indices=resampled_latent.to(self.device).long()).cpu()
                print(recon.shape)
                print(recons.shape)
                recons = torch.cat((recons, recon), 1)

        mean_rec = torch.mean(recons, 1, True)
        stdev_rec = torch.std(recons, dim=1, keepdim=True)
        # k_quantile = torch.quantile(stdev_rec, 0.5)
        # stdev_rec[stdev_rec < k_quantile] = k_quantile
        # #stdev_rec[stdev_rec < 0.02] = 0.02
        # print(stdev_rec.shape)
        if self.smoothing is not None:
            smooth = GaussianSmooth(self.smoothing)
            stdev_rec = smooth(stdev_rec[0])
            stdev_rec = torch.from_numpy(stdev_rec).unsqueeze(0)
        zscore = torch.div(torch.sub(inputs.cpu(), mean_rec), torch.add(stdev_rec,self.eps))
        residual_mean = torch.sub(inputs.cpu(), mean_rec)

        outputs = {"zscore": zscore, "std_rec": stdev_rec, "mean": mean_rec, "residual_mean": residual_mean}

        return outputs


class VQVAETransformerUncertaintySamplingInferer(VQVAETransformerInferer):
    def __init__(self, transf_network: TransformerBase, device: Any, num_passes: int, threshold: float, use_llmap: bool,
                 llmap: str, smoothing: float) -> None:
        VQVAETransformerInferer.__init__(self, transf_network=transf_network, device=device, threshold=threshold,
                                         use_llmap=use_llmap, llmap=llmap)

        self.transformer_network = transf_network
        self.device = device
        self.num_passes = num_passes
        self.threshold = threshold
        self.use_llmap = use_llmap
        self.llmap = llmap
        self.smoothing = smoothing
        self.eps = 1e-2

    def __call__(self, inputs: torch.Tensor, network: VQVAE, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        to ouput the probability maps of the tokens upsampled to the same shape as the original input image.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """
        embedding_indices = network.index_quantize(images=inputs)[0]
        latent_shape = embedding_indices.shape
        index_sequence = self.transformer_network.ordering.index_sequence
        revert_ordering = np.argsort(index_sequence)

        embedding_indices = embedding_indices.reshape(embedding_indices.shape[0], -1)
        embedding_indices = embedding_indices[:, index_sequence]

        for i in range(self.num_passes):
            if i == 0:
                resampled_latent = self.get_anomalies(embedding_indices, latent_shape, revert_ordering, index_sequence)
                recons = network.decode_samples(embedding_indices=resampled_latent.to(self.device).long()).cpu()
            else:
                resampled_latent = self.get_anomalies(embedding_indices, latent_shape, revert_ordering, index_sequence)
                recon = network.decode_samples(embedding_indices=resampled_latent.to(self.device).long()).cpu()
                recons = torch.cat((recons, recon), 1)

        mean_rec = torch.mean(recons, 1, True)
        stdev_rec = torch.std(recons, dim=1, keepdim=True)
        if self.smoothing is not None:
            smooth = GaussianSmooth(self.smoothing)
            stdev_rec = smooth(stdev_rec[0])
            stdev_rec = torch.from_numpy(stdev_rec).unsqueeze(0)
        zscore = torch.div(torch.sub(inputs.cpu(), mean_rec), torch.add(stdev_rec,self.eps))
        residual_mean = torch.sub(inputs.cpu(), mean_rec)

        outputs = {"zscore": zscore, "std_rec": stdev_rec, "mean": mean_rec, "residual_mean": residual_mean}

        return outputs


class VQVAETransformerUncertaintyCombinedInferer(VQVAETransformerInferer):
    def __init__(self, transf_network: TransformerBase, device: Any, num_passes_sampling: int,
                 num_passes_dropout: int, threshold: float, use_llmap: bool,
                 llmap: str, smoothing: float) -> None:
        VQVAETransformerInferer.__init__(self, transf_network=transf_network, device=device, threshold=threshold,
                                         use_llmap=use_llmap, llmap=llmap)

        self.transformer_network = transf_network
        self.device = device
        self.num_passes_sampling = num_passes_sampling
        self.num_passes_dropout = num_passes_dropout
        self.threshold = threshold
        self.use_llmap = use_llmap
        self.llmap = llmap
        self.eps = 1e-2
        self.smoothing = smoothing

    def __call__(self, inputs: torch.Tensor, network: VQVAE, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        to ouput the probability maps of the tokens upsampled to the same shape as the original input image.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """
        embedding_indices = network.index_quantize(images=inputs)[0]
        latent_shape = embedding_indices.shape
        index_sequence = self.transformer_network.ordering.index_sequence
        revert_ordering = np.argsort(index_sequence)

        embedding_indices = embedding_indices.reshape(embedding_indices.shape[0], -1)
        embedding_indices = embedding_indices[:, index_sequence]

        network.train()
        for i in range(self.num_passes_sampling):
            if i == 0:
                resampled_latent = self.get_anomalies(embedding_indices, latent_shape, revert_ordering, index_sequence)
                for k in range(self.num_passes_dropout):
                    if k == 0:
                        recons = network.decode_samples(embedding_indices=resampled_latent.to(self.device).long()).cpu()
                    else:
                        recon = network.decode_samples(embedding_indices=resampled_latent.to(self.device).long()).cpu()
                        recons = torch.cat((recons, recon), 1)

            else:
                resampled_latent = self.get_anomalies(embedding_indices, latent_shape, revert_ordering, index_sequence)
                for k in range(self.num_passes_dropout):
                    recon = network.decode_samples(embedding_indices=resampled_latent.to(self.device).long()).cpu()
                    recons = torch.cat((recons, recon), 1)

        mean_rec = torch.mean(recons, 1, True)
        stdev_rec = torch.std(recons, dim=1, keepdim=True)

        if self.smoothing is not None:
            smooth = GaussianSmooth(self.smoothing)
            stdev_rec = smooth(stdev_rec[0])
            stdev_rec = torch.from_numpy(stdev_rec).unsqueeze(0)

        zscore = torch.div(torch.sub(inputs.cpu(), mean_rec), torch.add(stdev_rec,self.eps))
        residual_mean = torch.sub(inputs.cpu(), mean_rec)

        outputs = {"zscore": zscore, "std_rec": stdev_rec, "mean": mean_rec, "residual_mean": residual_mean}

        return outputs
