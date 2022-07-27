# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from: https://github.com/facebookresearch/fastMRI

import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.common.parts.fft import fft2
from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.ssdu import SSDU
from mridc.collections.reconstruction.parts import transforms
from tests.collections.reconstruction.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "num_unroll_blocks": 10,
                "conjugate_gradient_num_iterations": 10,
                "kernels": [3, 3, 3],
                "dilations": [1, 1, 1],
                "bias": [True, True, True],
                "in_channels": 2,
                "num_filters": 64,
                "out_channels": 2,
                "conv_dim": 2,
                "num_pool_layers": 15,
                "regularization_factor": 0.1,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "dimensionality": 2,
            },
            [0.08],
            [4],
            2,
        ),
        (
            [1, 5, 15, 12, 2],
            {
                "num_unroll_blocks": 0,
                "conjugate_gradient_num_iterations": 0,
                "kernels": [3, 3, 3],
                "dilations": [1, 1, 1],
                "bias": [True, True, True],
                "in_channels": 2,
                "num_filters": 32,
                "out_channels": 2,
                "conv_dim": 2,
                "num_pool_layers": 15,
                "regularization_factor": 0.1,
                "use_sens_net": False,
                "dimensionality": 2,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
            2,
        ),
        (
            [1, 6, 54, 32, 2],
            {
                "num_unroll_blocks": 5,
                "conjugate_gradient_num_iterations": 1,
                "kernels": [3, 3, 3],
                "dilations": [1, 1, 1],
                "bias": [True, True, True],
                "in_channels": 2,
                "num_filters": 128,
                "out_channels": 2,
                "conv_dim": 2,
                "num_pool_layers": 7,
                "regularization_factor": 0.1,
                "use_sens_net": False,
                "dimensionality": 2,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
            2,
        ),
    ],
)
def test_ssdu(shape, cfg, center_fractions, accelerations, dimensionality):
    """
    Test SSDU with different parameters

    Args:
        shape: shape of the input
        cfg: configuration of the model
        center_fractions: center fractions
        accelerations: accelerations

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

    model = SSDU(cfg)

    with torch.no_grad():
        y = model.forward(output, output, mask, output, target=torch.abs(torch.view_as_complex(output)))
        y = fft2(
            torch.view_as_real(y).unsqueeze(coil_dim) * output, fft_centered, fft_normalization, spatial_dims
        ) * mask.squeeze(0)

    if y.shape != x.shape:
        raise AssertionError
    if y.shape[1] != x.shape[1]:
        raise AssertionError
