from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from monai.engines.trainer import Trainer
from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import exact_version, optional_import
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.engines.utils import AdversarialKeys

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.4.4", exact_version, "Metric")


class AdversarialTrainer(Trainer):
    """
    Adversarial network training, inherits from ``Trainer`` and ``Workflow``.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for engine to run.
        train_data_loader: Core ignite engines uses `DataLoader` for training loop batchdata.
        recon_loss_function: G loss fcuntion for reconstructions.
        g_network: ''generator'' (G) network architecture.
        g_optimizer: G optimizer function.
        g_loss_function: G loss function for adversarial training.
        d_network: discriminator (D) network architecture.
        d_optimizer: D optimizer function.
        d_loss_function: D loss function for adversarial training..
        epoch_length: number of iterations for one epoch, default to `len(train_data_loader)`.
        g_inferer: inference method to execute G model forward. Defaults to ``SimpleInferer()``.
            optimization step. It is expected to receive g_infer's output, device and non_blocking.
        d_inferer: inference method to execute D model forward. Defaults to ``SimpleInferer()``.
            receive the 'batchdata', g_inferer's output, device and non_blocking.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        post_transform: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision training or inference, default is False.
    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: DataLoader,
        recon_loss_function: Callable,
        perceptual_loss_function: Callable,
        g_network: torch.nn.Module,
        g_optimizer: Optimizer,
        g_loss_function: Callable,
        d_network: torch.nn.Module,
        d_optimizer: Optimizer,
        d_loss_function: Callable,
        epoch_length: Optional[int] = None,
        g_inferer: Optional[Inferer] = None,
        d_inferer: Optional[Inferer] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        train_handlers: Optional[Sequence] = None,
        amp: bool = False,
        kl_weight: float = 0.000001,
        perceptual_weight: float = 10,
        adversarial_weight: float = 1,
        reconstruction_weight: float = 5,
    ):
        # set up Ignite engine and environments
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            handlers=train_handlers,
            amp=amp,
        )
        self.g_network = g_network
        self.g_optimizer = g_optimizer
        self.g_loss_function = g_loss_function
        self.recon_loss_function = recon_loss_function
        self.perceptual_loss_function = perceptual_loss_function

        self.d_network = d_network
        self.d_optimizer = d_optimizer
        self.d_loss_function = d_loss_function

        self.g_inferer = SimpleInferer() if g_inferer is None else g_inferer
        self.d_inferer = SimpleInferer() if d_inferer is None else d_inferer

        self.g_scaler = torch.cuda.amp.GradScaler() if self.amp else None
        self.d_scaler = torch.cuda.amp.GradScaler() if self.amp else None

        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.reconstruction_weight = reconstruction_weight

    def _iteration(
        self, engine: Engine, batchdata: Union[Dict, Sequence]
    ) -> Dict[str, Union[torch.Tensor, int, float, bool]]:
        """
        Callback function for Adversarial Training processing logic of 1 iteration in Ignite Engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Returns:
            Dict: Results of the iterations
                - REALS: image Tensor data for model input, already moved to device.
                - FAKES: output Tensor data corresponding to the image, already moved to device.
                - GLOSS: the loss of the g_network
                - DLOSS: the loss of the d_network

        Raises:
            ValueError: When ``batchdata`` is None.

        """

        print("in iteration now")
        print(ok)
        if batchdata is None:
            raise ValueError("must provide batch data for current iteration.")

        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)

        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch

        batch_size = self.data_loader.batch_size  # type: ignore

        # Train generator
        self.g_network.train()
        self.g_optimizer.zero_grad(set_to_none=True)

        if self.amp and self.g_scaler is not None:
            with torch.cuda.amp.autocast():
                g_predictions, z_mu, z_sigma = self.g_inferer(inputs, self.g_network, *args, **kwargs)

                logits_fake = self.d_inferer(
                    g_predictions.float().contiguous(),
                    self.d_network,
                    *args,
                    **kwargs
                )

                recon_loss = self.recon_loss_function(g_predictions, targets).mean()
                perceptual_loss = self.perceptual_loss_function(g_predictions, targets).mean()

                g_loss = self.g_loss_function(logits_fake).mean()
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
                                          dim=[1, 2, 3, 4])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                g_loss = (self.reconstruction_weight * recon_loss) + (self.kl_weight * kl_loss) + (
                            self.adversarial_weight * g_loss) + (self.perceptual_weight * perceptual_loss)

            self.g_scaler.scale(g_loss).backward()
            self.g_scaler.step(self.g_optimizer)
            self.g_scaler.update()
        else:
            g_predictions = self.g_inferer(inputs, self.g_network, *args, **kwargs)
            logits_fake = self.d_inferer(
                g_predictions["reconstruction"][0].float().contiguous(),
                self.d_network,
                *args,
                **kwargs
            )
            recon_loss = self.recon_loss_function(g_predictions, targets).mean()

            g_loss = self.g_loss_function(logits_fake).mean()

            g_loss = recon_loss + g_loss
            g_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.g_network.parameters(), 5)
            self.g_optimizer.step()

        # Train Discriminator
        self.d_network.train()
        self.d_network.zero_grad(set_to_none=True)

        if self.amp and self.d_scaler is not None:
            with torch.cuda.amp.autocast():
                logits_fake = self.d_inferer(
                    g_predictions.float().contiguous().detach(),
                    self.d_network,
                    *args,
                    **kwargs
                )

                logits_real = self.d_inferer(
                    inputs.contiguous().detach(), self.d_network, *args, **kwargs
                )

                d_loss = self.d_loss_function(logits_fake,logits_real).mean()

            self.d_scaler.scale(d_loss).backward()
            self.d_scaler.step(self.d_optimizer)
            self.d_scaler.update()
        else:
            logits_fake = self.d_inferer(
                g_predictions["reconstruction"][0].float().contiguous().detach(),
                self.d_network,
                *args,
                **kwargs
            )

            logits_real = self.d_inferer(
                inputs.contiguous().detach(), self.d_network, *args, **kwargs
            )

            d_loss = self.d_loss_function(logits_fake, logits_real).mean()
            d_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.d_network.parameters(), 5)
            self.d_optimizer.step()

        return {
            Keys.IMAGE: inputs,
            Keys.LABEL: targets,
            Keys.PRED: g_predictions,
            Keys.LOSS: recon_loss.item(),
            AdversarialKeys.REALS: inputs,
            AdversarialKeys.FAKES: g_predictions,
            AdversarialKeys.GLOSS: g_loss.item(),
            AdversarialKeys.DLOSS: d_loss.item(),
        }
