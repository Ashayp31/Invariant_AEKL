from ignite.engine import Engine

from typing import Tuple

from ignite.engine import Events
from monai.engines.utils import CommonKeys

from src.handlers.general import TBSummaryTypes
#from src.networks.vqvae.vqvae import VQVAE
from src.engines.utils import AdversarialKeys

class VQVAELoggingHandler:
    """
    Handler that stores the VQ-VAE specific summaries into engine.state.output["summaries"].
    Args:
        network (VQVAEBase): VQVAE network
        is_eval (bool): Whether the network runs in an evaluation engine or not. If True we do not have a loss
            output, so we must skip that logic.
        log_2d (Tuple[str,...]): If it is not None the enumerated axis will be logged in 2D via mid slices.
        log_3d (Tuple[str,...]): If it is not None the enumerated axis will be logged in 3D via GIFs.
    """

    def __init__(
        self,
        network: any,
        log_2d: Tuple[str, ...] = ("coronal", "axial", "sagittal"),
        log_3d: Tuple[str, ...] = None,
        log_image: bool = False,
        is_eval: bool = False,
    ):
        self.is_eval = is_eval
        self.log_3d = log_3d
        self.log_2d = log_2d
        self.log_image = log_image

    def __call__(self, engine: Engine):
        output = engine.state.output

        if self.log_3d:
            if "coronal" in self.log_3d:
                output["summaries"][TBSummaryTypes.IMAGE3_CORONAL][
                    "3D-Originals-Coronal"
                ] = output[CommonKeys.IMAGE]

                output["summaries"][TBSummaryTypes.IMAGE3_CORONAL][
                    "3D-Predictions-Coronal"
                ] = output[CommonKeys.PRED]["reconstruction"][0]

            if "axial" in self.log_3d:
                output["summaries"][TBSummaryTypes.IMAGE3_AXIAL][
                    "3D-Originals-Axial"
                ] = output[CommonKeys.IMAGE]

                output["summaries"][TBSummaryTypes.IMAGE3_AXIAL][
                    "3D-Predictions-Axial"
                ] = output[CommonKeys.PRED]["reconstruction"][0]

            if "sagittal" in self.log_3d:
                output["summaries"][TBSummaryTypes.IMAGE3_SAGITTAL][
                    "3D-Originals-Sagittal"
                ] = output[CommonKeys.IMAGE]

                output["summaries"][TBSummaryTypes.IMAGE3_SAGITTAL][
                    "3D-Predictions-Sagittal"
                ] = output[CommonKeys.PRED]["reconstruction"][0]

        if self.log_2d:
            if "coronal" in self.log_2d:
                output["summaries"][TBSummaryTypes.IMAGE_CORONAL][
                    "2D-Originals-Coronal"
                ] = output[CommonKeys.IMAGE]

                output["summaries"][TBSummaryTypes.IMAGE_CORONAL][
                    "2D-Predictions-Coronal"
                ] = output[CommonKeys.PRED]["reconstruction"][0]

            if "axial" in self.log_2d:
                output["summaries"][TBSummaryTypes.IMAGE_AXIAL][
                    "2D-Originals-Axial"
                ] = output[CommonKeys.IMAGE]

                output["summaries"][TBSummaryTypes.IMAGE_AXIAL][
                    "2D-Predictions-Axial"
                ] = output[CommonKeys.PRED]["reconstruction"][0]

            if "sagittal" in self.log_2d:
                output["summaries"][TBSummaryTypes.IMAGE_SAGITTAL][
                    "2D-Originals-Sagittal"
                ] = output[CommonKeys.IMAGE]

                output["summaries"][TBSummaryTypes.IMAGE_SAGITTAL][
                    "2D-Predictions-Sagittal"
                ] = output[CommonKeys.PRED]["reconstruction"][0]

        if self.log_image:
            output["summaries"][TBSummaryTypes.IMAGE][
                "Originals-Image"
            ] = output[CommonKeys.IMAGE]

            output["summaries"][TBSummaryTypes.IMAGE][
                "Predictions-Image"
            ] = output[CommonKeys.PRED]["reconstruction"][0]


        if not self.is_eval:

            output["summaries"][TBSummaryTypes.SCALAR]["Loss"] = output[CommonKeys.LOSS]

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)



class AdversarialFinetuneHandler:
    # TODO: Make it DDP friendly
    """
        Handler that checks if the discriminator loss is within the range defined by adversarial_loss_range for a number of
        iterations equal to the adversarial_loss_patience. While it isn't the network will be put into evaluation mode.
        Afterwards it will turned to train model

        !!! This handler is not DDP friendly the logic of turing the model to train is rank specific and it not broadcasted
        across all ranks.

        Args:
            network (VQVAEBase): A VQ-VAE network
            adversarial_loss_range (Tuple[float,float]): The range within the discriminator loss needs to be within
            adversarial_loss_patience (int): The number of iterations the discriminator loss needs to be within range
    """

    def __init__(
        self,
        network: any,
        adversarial_loss_range: Tuple[float, float],
        adversarial_loss_patience: int,
    ):
        self.network = network
        self.adversarial_loss_range = adversarial_loss_range
        self.adversarial_loss_patience = adversarial_loss_patience

        self.counter = 0

    def __call__(self, engine: Engine):
        # This is done for the very first iteration where there is no output dictionary
        if engine.state.output:
            discriminator_loss = engine.state.output[AdversarialKeys.DLOSS]

            if self.counter < self.adversarial_loss_patience:
                if (
                    self.adversarial_loss_range[0]
                    <= discriminator_loss
                    <= self.adversarial_loss_range[1]
                ):
                    self.counter += 1
                else:
                    self.counter = 0
                # Just a sanity maintaining code to be sure the network is in train if and only if
                #   counter > adversarial_loss_patience
                self.network.eval()

            if self.counter >= self.adversarial_loss_patience:
                self.network.train()
        else:
            self.network.eval()

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_STARTED, self)

