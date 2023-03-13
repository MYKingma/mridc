# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from mridc.collections.motioncorrection.models.base import BaseMRIMoCoReconstructionModel
from mridc.collections.reconstruction.models.cirim import CIRIM
import torch
import numpy as np


__all__ = ["MoCoCIRIM"]


class MoCoCIRIM(BaseMRIMoCoReconstructionModel, CIRIM, ABC):
    pass

    def process_loss(self, target, pred, _loss_fn=None, mask=None):
        """
        Process the loss.

        Parameters
        ----------
        target: Target data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred: Final prediction(s).
            list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        _loss_fn: Loss function.
            torch.nn.Module, default torch.nn.L1Loss()

        Returns
        -------
        loss: torch.FloatTensor, shape [1]
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
        """
        target = torch.abs(target / torch.max(torch.abs(target)))

        if "ssim" in str(_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def loss_fn(x, y, m):
                """Calculate the ssim loss."""
                y = torch.abs(y / torch.max(torch.abs(y)))
                return _loss_fn(
                    x.unsqueeze(dim=self.coil_dim),
                    y.unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y, m):
                """Calculate other loss."""
                y = torch.abs(y / torch.max(torch.abs(y)))
                return _loss_fn(x, y)

        if self.accumulate_estimates:
            cascades_loss = []
            for cascade_pred in pred:
                time_steps_loss = [loss_fn(target, time_step_pred, mask) for time_step_pred in cascade_pred]
                _loss = [
                    x * torch.logspace(-1, 0, steps=self.time_steps).to(time_steps_loss[0]) for x in time_steps_loss
                ]
                cascades_loss.append(sum(sum(_loss) / self.time_steps))
            yield sum(list(cascades_loss)) / len(self.cirim)
        else:
            return loss_fn(target, pred, mask)
