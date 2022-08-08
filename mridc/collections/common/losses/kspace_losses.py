# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Union

import torch

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import is_none
from mridc.core.classes.common import Typing
from mridc.core.classes.loss import Loss
from mridc.core.neural_types.elements import LossType
from mridc.core.neural_types.neural_type import NeuralType
from mridc.utils import logging

__all__ = ["SSDUKSPACELoss"]


class SSDUKSPACELoss(Loss, Typing):
    """
    This function computes the kspace loss between the predicted and the ground truth kspace.
    """

    @property
    def output_types(self):
        """Returns the output types of the loss function."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        loss_function: str = "l2_l1",
        reduction: str = "mean",
        complex: bool = False,
        regularization_factor: float = 0.5,
        normalize: Union[str, None] = None,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims=None,
    ):
        """
        Initializes the loss function.

        Parameters
        ----------
        loss_function : str
            The loss function to use.
        reduction : str, optional
            The reduction to apply to the loss.
        complex : bool, optional
            Whether compute the loss on complex inputs or not.
        regularization_factor : float, optional
            The regularization factor.
        normalize : str, optional
            Whether to normalize the input or not.
        fft_centered : bool, optional
            Whether to center the FFT or not.
        fft_normalization : str, optional
            Whether to normalize the FFT or not.
        spatial_dims : tuple, optional
            The spatial dimensions to use for the FFT.
        """
        super().__init__()
        if loss_function == "l2_l1":
            self.loss_function = lambda x, y: self._norm_loss(x, y)
        elif loss_function == "l2":
            self.loss_function = torch.nn.MSELoss()
        elif loss_function == "l1":
            self.loss_function = torch.nn.L1Loss()
        else:
            raise ValueError(f"{loss_function} is not implemented as k-space loss function.")

        if reduction not in ["mean", "sum"]:
            logging.warning(f'{reduction} reduction is not supported. Setting reduction to "mean"')
            reduction = "mean"
        self.reduction = reduction
        self.complex = complex
        self.regularization_factor = torch.tensor(regularization_factor, dtype=torch.float32)
        self.normalize = normalize

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        if spatial_dims is None:
            spatial_dims = [-2, -1]
        self.spatial_dims = spatial_dims

    @staticmethod
    def _helper_verify_real_tensor(x):
        """Helper function to verify that the input is a real tensor."""
        if x.shape[-1] != 2:
            x = torch.view_as_real(x)
        return x

    @staticmethod
    def _helper_verify_complex_tensor(x):
        """Helper function to verify that the input is a complex tensor."""
        if x.shape[-1] == 2:
            x = torch.view_as_complex(x)
        return x

    def _helper_normalize(self, x):
        """Helper function to normalize the input."""
        if self.normalize == "kspace":
            return x / torch.max(torch.abs(x))
        elif self.normalize == "imspace":
            x = self._helper_verify_real_tensor(x)
            x = ifft2(x, self.fft_centered, self.fft_normalization, self.spatial_dims)
            x = x / torch.max(torch.abs(x))
            x = fft2(x, self.fft_centered, self.fft_normalization, self.spatial_dims)
            return x
        elif not is_none(self.normalize):
            raise ValueError(f"{self.normalize} is not a valid normalization method")

    @staticmethod
    def _helper_flatten(x):
        """Helper function to flatten the input."""
        return torch.flatten(x, start_dim=0, end_dim=-1)

    def _norm_loss(self, reference_kspace, predicted_kspace):
        """Computes the loss."""
        l2 = torch.linalg.norm(reference_kspace - predicted_kspace) / torch.linalg.norm(reference_kspace)
        l2_regularization = self.regularization_factor * l2

        l1 = torch.linalg.norm(self._helper_flatten(reference_kspace - predicted_kspace), ord=1) / torch.linalg.norm(
            self._helper_flatten(reference_kspace), ord=1
        )
        l1_regularization = self.regularization_factor * l1

        return l2_regularization + l1_regularization

    def forward(
        self,
        reference_kspace,
        predicted_kspace,
    ):
        """
        Computes the kspace loss between the predicted and the ground truth kspace.

        Parameters
        ----------
        reference_kspace : torch.Tensor
            The ground truth kspace.
        predicted_kspace : torch.Tensor
            The predicted kspace.

        Returns
        -------
        torch.Tensor
            The kspace loss.
        """
        if not is_none(self.normalize):
            reference_kspace = self._helper_normalize(reference_kspace)
            predicted_kspace = self._helper_normalize(predicted_kspace)

        if self.complex:
            reference_kspace = self._helper_verify_complex_tensor(reference_kspace)
            predicted_kspace = self._helper_verify_complex_tensor(predicted_kspace)
        else:
            reference_kspace = self._helper_verify_real_tensor(reference_kspace)
            predicted_kspace = self._helper_verify_real_tensor(predicted_kspace)

        return self.loss_function(reference_kspace, predicted_kspace)
