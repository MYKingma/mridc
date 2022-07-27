# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from: https://github.com/facebookresearch/fastMRI

import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.common.parts.fft import fft2
from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.cirim import CIRIM
from mridc.collections.reconstruction.parts import transforms
from tests.collections.reconstruction.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, self_supervised, reference_kspace_normalization, "
    "self_supervised_masking_type",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 1,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "dimensionality": 2,
                "self_supervised": "SSDU",
                "self_supervised_train_loss": "l2_l1",
                "self_supervised_val_loss": "l2_l1",
                "self_supervised_complex_kspace_loss": False,
                "self_supervised_loss_reg_factor": 0.5,
                "self_supervised_normalize_loss": "none",
            },
            [0.08],
            [4],
            2,
            "SSDU",
            True,
            "gaussian",
        ),
        (
            [1, 5, 15, 12, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 5,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "dimensionality": 2,
                "self_supervised": "SSDU",
                "self_supervised_train_loss": "l2_l1",
                "self_supervised_val_loss": "l2_l1",
                "self_supervised_complex_kspace_loss": True,
                "self_supervised_loss_reg_factor": 0.5,
                "self_supervised_normalize_loss": "none",
            },
            [0.08],
            [4],
            2,
            "SSDU",
            True,
            "gaussian",
        ),
        (
            [1, 8, 13, 18, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 16,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "dimensionality": 2,
                "self_supervised": "SSDU",
                "self_supervised_train_loss": "l2_l1",
                "self_supervised_val_loss": "l2_l1",
                "self_supervised_complex_kspace_loss": False,
                "self_supervised_loss_reg_factor": 0.5,
                "self_supervised_normalize_loss": "kspace",
            },
            [0.08],
            [4],
            2,
            "SSDU",
            True,
            "gaussian",
        ),
        (
            [10, 2, 3, 15, 12, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 3,
                "time_steps": 8,
                "num_cascades": 5,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "dimensionality": 3,
                "self_supervised": "SSDU",
                "self_supervised_train_loss": "l2_l1",
                "self_supervised_val_loss": "l2_l1",
                "self_supervised_complex_kspace_loss": True,
                "self_supervised_loss_reg_factor": 0.5,
                "self_supervised_normalize_loss": "kspace",
            },
            [0.08],
            [4],
            3,
            "SSDU",
            True,
            "gaussian",
        ),
        (
            [3, 2, 5, 15, 12, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 3,
                "time_steps": 8,
                "num_cascades": 5,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "dimensionality": 3,
                "self_supervised": "SSDU",
                "self_supervised_train_loss": "l2_l1",
                "self_supervised_val_loss": "l2_l1",
                "self_supervised_complex_kspace_loss": False,
                "self_supervised_loss_reg_factor": 0.5,
                "self_supervised_normalize_loss": "imspace",
            },
            [0.08],
            [4],
            3,
            "SSDU",
            True,
            "gaussian",
        ),
        (
            [6, 2, 15, 15, 12, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 3,
                "time_steps": 8,
                "num_cascades": 5,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "dimensionality": 3,
                "self_supervised": "SSDU",
                "self_supervised_train_loss": "l2_l1",
                "self_supervised_val_loss": "l2_l1",
                "self_supervised_complex_kspace_loss": True,
                "self_supervised_loss_reg_factor": 0.5,
                "self_supervised_normalize_loss": "imspace",
            },
            [0.08],
            [4],
            3,
            "SSDU",
            True,
            "gaussian",
        ),
    ],
)
def test_ssl_cirim(
    shape,
    cfg,
    center_fractions,
    accelerations,
    dimensionality,
    self_supervised,
    reference_kspace_normalization,
    self_supervised_masking_type,
):
    """
    Test self-supervised CIRIM with different parameters

    Args:
        shape: shape of the input
        cfg: configuration of the model
        center_fractions: center fractions
        accelerations: accelerations
        dimensionality: 2D or 3D inputs
        self_supervised: SSDU or None
        reference_kspace_normalization: True or False
        self_supervised_masking_type: gaussian or uniform

    Returns:
        None
    """
    mask_func = RandomMaskFunc(center_fractions, accelerations)
    x = create_input(shape)

    outputs, masks = [], []
    for i in range(x.shape[0]):
        output, mask, _ = transforms.apply_mask(x[i : i + 1], mask_func, seed=123)
        outputs.append(output)
        masks.append(mask)

    output = torch.cat(outputs)
    mask = torch.cat(masks)

    if dimensionality == 3 and shape[1] > 1:
        mask = torch.cat([mask, mask], 1)

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    fft_centered = cfg.fft_centered
    fft_normalization = cfg.fft_normalization
    spatial_dims = cfg.spatial_dims
    coil_dim = cfg.coil_dim

    cirim = CIRIM(cfg)

    with torch.no_grad():
        y = cirim.forward(
            output,
            output,
            mask,
            None,
            target=torch.abs(torch.view_as_complex(output)),
        )

        try:
            y = next(y)
        except StopIteration:
            pass

        if isinstance(y, list):
            y = y[-1]

        if isinstance(y, list):
            y = y[-1]

    if dimensionality == 3:
        x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]])
        output = output.reshape(
            [output.shape[0] * output.shape[1], output.shape[2], output.shape[3], output.shape[4], output.shape[5]]
        )
        mask = mask.reshape(
            [mask.shape[0] * mask.shape[1], mask.shape[2], mask.shape[3], mask.shape[4], mask.shape[5]]
        )
        y = (
            fft2(torch.view_as_real(y).unsqueeze(coil_dim) * output, fft_centered, fft_normalization, spatial_dims)
            * mask
        )
    else:
        y = fft2(
            torch.view_as_real(y).unsqueeze(coil_dim) * output, fft_centered, fft_normalization, spatial_dims
        ) * mask.squeeze(0)

    if y.shape != x.shape:
        raise AssertionError
    if y.shape[1] != x.shape[1]:
        raise AssertionError
