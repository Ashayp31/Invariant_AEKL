from typing import Dict

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from src.handlers.general import TBSummaryTypes


class CELoss(_Loss):
    def __init__(
        self, weight=None, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super(CELoss, self).__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict(), TBSummaryTypes.HISTOGRAM: dict()}
        self._weight = weight

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.long()
        y_pred = y_pred.float()

        loss = F.cross_entropy(
            input=y_pred, target=y, reduction=self.reduction, weight=self._weight
        )
        loss_not_reduced = F.cross_entropy(
            input=y_pred, target=y, reduction="none", weight=self._weight
        )
        self.summaries[TBSummaryTypes.SCALAR]["Loss-CE-Prediction"] = loss
        self.summaries[TBSummaryTypes.HISTOGRAM]["Loss-CE-Hist"] = loss_not_reduced
        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries


class DiscCELoss(_Loss):

    def __init__(self, lamb: float = 0.1, size_average: bool = True, weight=None, reduction: str = "mean"):
        super(DiscCELoss, self).__init__(size_average, reduction)

        self.lamb = lamb
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict(), TBSummaryTypes.HISTOGRAM: dict()}
        self._weight = weight
        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        y = y.cpu().long()
        y_pred = y_pred.cpu().float()

        err = F.cross_entropy(
            input=y_pred, target=y, reduction="none", weight=self._weight
        )

        err_sorted, _ = torch.sort(err)
        total_var = err.sub(err.mean()).pow(2).sum()
        regul = 1e8
        obj = None
        err_sorted = err_sorted.view(-1)

        for i in range(err_sorted.size(0)-1):
            err_in = err_sorted[:i+1]
            err_out = err_sorted[i+1:]
            within_var = err_in.sub(err_in.mean()).pow(2).sum() + \
                             err_out.sub(err_out.mean()).pow(2).sum()
            h = within_var.div(total_var)
            if h < regul:
                regul = h
                if self.reduction == "mean":
                    obj = err_in.mean()
                else:
                    obj = err_in.sum()

        
        self.summaries[TBSummaryTypes.SCALAR]["Loss-CE-Prediction"] = err.mean() if self.reduction == "mean" \
            else err.sum()
        self.summaries[TBSummaryTypes.SCALAR]["Loss-CE-Discriminative"] = obj + self.lamb * regul

        self.summaries[TBSummaryTypes.HISTOGRAM]["Loss-CE-Hist"] = err_sorted

        return obj + self.lamb * regul

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries


class DiscCELoss2(_Loss):

    def __init__(self, lamb: float = 0.1, size_average: bool = True, weight=None, reduction: str = "mean"):
        super(DiscCELoss, self).__init__(size_average, reduction)

        self.lamb = lamb
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict(), TBSummaryTypes.HISTOGRAM: dict()}
        self._weight = weight
        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        y = y.cpu().long()
        y_pred = y_pred.cpu().float()

        err = F.cross_entropy(
            input=y_pred, target=y, reduction="none", weight=self._weight
        )

        # sort error in size
        err_sorted, _ = torch.sort(err)
        total_var = err.sub(err.mean()).pow(2).sum()
        regul = 1e8
        obj = None
        err_sorted = err_sorted.view(-1)

        for i in range(err_sorted.size(0)-1):
            err_in = err_sorted[:i+1]
            err_out = err_sorted[i+1:]
            within_var = err_in.sub(err_in.mean()).pow(2).sum() + \
                             err_out.sub(err_out.mean()).pow(2).sum()
            h = within_var.div(total_var)
            if h < regul:
                regul = h
                reduced_err_out = err_out
                if self.reduction == "mean":
                    obj = err_in.mean()
                else:
                    obj = err_in.sum()

        initial_loss = obj + self.lamb * regul

        # repeat for new err_in
        err_sorted_2, _ = torch.sort(reduced_err_out)
        total_var = reduced_err_out.sub(reduced_err_out.mean()).pow(2).sum()
        regul = 1e8
        obj = None
        err_sorted_2 = err_sorted_2.view(-1)

        for i in range(err_sorted_2.size(0)-1):
            err_in = err_sorted_2[:i+1]
            err_out = err_sorted_2[i+1:]
            within_var = err_in.sub(err_in.mean()).pow(2).sum() + \
                             err_out.sub(err_out.mean()).pow(2).sum()
            h = within_var.div(total_var)
            if h < regul:
                regul = h
                if self.reduction == "mean":
                    obj = err_in.mean()
                else:
                    obj = err_in.sum()

        total_error = initial_loss + (obj + self.lamb * regul)

        
        self.summaries[TBSummaryTypes.SCALAR]["Loss-CE-Prediction"] = err.mean() if self.reduction == "mean" \
            else err.sum()
        self.summaries[TBSummaryTypes.SCALAR]["Loss-CE-Discriminative"] = total_error

        self.summaries[TBSummaryTypes.HISTOGRAM]["Loss-CE-Hist"] = err_sorted

        return total_error

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

