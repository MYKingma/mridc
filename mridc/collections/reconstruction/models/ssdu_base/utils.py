# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Sequence

import torch

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import coil_combination, complex_mul


def conjugate_gradient(
    pred: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    mask: torch.Tensor,
    mu_param: torch.Tensor,
    CG_Iter: int = 10,
    fft_centered: bool = True,
    fft_normalization: str = "backward",
    spatial_dims: Sequence[int] = None,
    coil_dim: int = 1,
    coil_combination_method: str = "SENSE",
):
    """
    Solve the linear system Ax = b using conjugate gradient.

    Parameters
    ----------
    pred: torch.Tensor, shape [batch_size, n_x, n_y, 2]
        Initial estimate.
    sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        Coil sensitivity maps.
    mask: torch.Tensor, shape [1, 1, n_x, n_y, 1]
        Sampling mask.
    mu_param: torch.Tensor, shape [1]
        Regularization parameter.
    CG_Iter: int, optional
        Number of iterations of the conjugate gradient algorithm.
    fft_centered: bool, optional
        If True, use centered FFT.
    fft_normalization: str, optional
        If "none", do not use FFT normalization.
    spatial_dims: tuple, optional
        Spatial dimensions to use for the FFT.
    coil_dim: int, optional
        Dimension of the coil dimension.
    coil_combination_method: str, optional
        Method to use for coil combination.

    Returns
    -------
    pred: torch.Tensor, shape [batch_size, n_x, n_y, 2]
        Final estimate.
    """
    spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]

    if pred.shape[-1] == 2:
        pred = torch.view_as_complex(pred)
    scalar = torch.sqrt(torch.tensor(mask.shape[2] * mask.shape[3]))

    def body(i, rsold, x, r, p, mu):
        """
        Conjugate gradient body.

        Parameters
        ----------
        i: int
            Current iteration.
        rsold: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Initial residual.
        x: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Current estimate.
        r: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Current residual.
        p: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Current search direction.
        mu: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Regularization parameter.
        """
        pred = p.clone()
        if pred.shape[-1] != 2:
            pred = torch.view_as_real(pred)
        pred = (
            coil_combination(
                ifft2(
                    (fft2(complex_mul(pred, sensitivity_maps), fft_centered, fft_normalization, spatial_dims) / scalar)
                    * mask,
                    fft_centered,
                    fft_normalization,
                    spatial_dims,
                ),
                sensitivity_maps,
                coil_combination_method,
                coil_dim,
            ).unsqueeze(coil_dim)
            * scalar
        )
        Ap = (pred[..., 0] + 1j * pred[..., 1]) + mu * p
        alpha = rsold / torch.sum(torch.conj(p) * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(torch.conj(r) * r)
        beta = rsnew / rsold
        p = r + beta * p
        return i + 1, rsnew, x, r, p, mu

    x = torch.zeros_like(pred)
    i, r, p = 0, pred, pred
    rsold = torch.sum(torch.conj(r) * r)
    while i < CG_Iter:
        i, rsold, x, r, p, mu_param = body(i, rsold, x, r, p, mu_param)
    cg_out = x
    return torch.view_as_real(cg_out)
