# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import List, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import ifft2
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel
from mridc.collections.reconstruction.models.ssdu_base.resnet import SSDUResNet
from mridc.collections.reconstruction.models.ssdu_base.utils import conjugate_gradient
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["SSDU"]


class SSDU(BaseMRIReconstructionModel, ABC):
    """
    Implementation of the SSDU, as presented by Yaman, B., et al.

    References
    ----------
    ..

         Yaman, B, Hosseini, SAH, Moeller, S, Ellermann, J, Uğurbil, K, Akçakaya, M.
         Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference
         data. Magn Reson Med. 2020; 84: 3172– 3191. https://doi.org/10.1002/mrm.28378

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        dimensionality = cfg_dict["dimensionality"]
        if dimensionality == 3:
            raise NotImplementedError("3D is not implemented yet for SSDU.")

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        self.num_unroll_blocks = cfg_dict.get("num_unroll_blocks")
        self.CG_Iter = cfg_dict.get("conjugate_gradient_num_iterations")
        self.regularization_factor = cfg_dict.get("regularization_factor")

        self.model = SSDUResNet(
            kernels=cfg_dict.get("kernels"),
            dilations=cfg_dict.get("dilations"),
            bias=cfg_dict.get("bias"),
            in_channels=cfg_dict.get("in_channels"),
            num_filters=cfg_dict.get("num_filters"),
            out_channels=cfg_dict.get("out_channels"),
            conv_dim=cfg_dict.get("conv_dim"),
            num_pool_layers=cfg_dict.get("num_pool_layers"),
            regularization_factor=self.regularization_factor,
        )

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.train_loss_fn = SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()

        self.accumulate_estimates = False

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: Union[torch.Tensor, List],
        init_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        y: Subsampled k-space data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            torch.Tensor, shape [1, 1, n_x, n_y, 1]
        init_pred: Initial prediction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        target: Target data to compute the loss.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        pred: list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or  torch.Tensor, shape [batch_size, n_x, n_y, 2]
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        if isinstance(mask, list):
            mask = mask[0]

        eta = coil_combination(
            ifft2(y, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims),
            sensitivity_maps,
            method=self.coil_combination_method,
            dim=self.coil_dim,
        )
        _, eta = center_crop_to_smallest(target, eta)

        x = eta.permute(0, 3, 1, 2)

        mu = torch.nn.Parameter(torch.tensor(self.regularization_factor))
        for _ in range(self.num_unroll_blocks):
            x = torch.view_as_real(self.model(x))
            if self.CG_Iter > 0:
                rhs = eta + mu * x
                x = (
                    conjugate_gradient(
                        rhs,
                        sensitivity_maps,
                        mask,
                        mu,
                        self.CG_Iter,
                        self.fft_centered,
                        self.fft_normalization,
                        self.spatial_dims,
                        self.coil_dim,
                    )
                    .squeeze(self.coil_dim)
                    .permute(0, 3, 1, 2)
                )
        x = x.permute(0, 2, 3, 1)
        return x[..., 0] + 1j * x[..., 1]
