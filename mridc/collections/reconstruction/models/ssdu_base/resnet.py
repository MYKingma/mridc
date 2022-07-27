# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch

from mridc.collections.reconstruction.models.rim.conv_layers import ConvNonlinear


class SSDUResNet(torch.nn.Module):
    """
    Implementation of the SSDU, as presented by Yaman, B., et al.

    References
    ----------
    ..

         Yaman, B, Hosseini, SAH, Moeller, S, Ellermann, J, Uğurbil, K, Akçakaya, M.
         Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference
         data. Magn Reson Med. 2020; 84: 3172– 3191. https://doi.org/10.1002/mrm.28378

    """

    def __init__(
        self,
        kernels=None,
        dilations=None,
        bias=None,
        in_channels: int = 2,
        num_filters: int = 64,
        out_channels: int = 2,
        conv_dim: int = 2,
        num_pool_layers: int = 15,
        regularization_factor: float = 0.1,
    ):
        super().__init__()

        self.input_layer = ConvNonlinear(
            in_channels,
            num_filters,
            conv_dim=conv_dim,
            kernel_size=kernels[0],
            dilation=dilations[0],
            bias=bias[0],
            nonlinear=None,
        )
        self.layers = torch.nn.ModuleList()
        for _ in range(num_pool_layers):
            self.layers.append(
                ConvNonlinear(
                    num_filters,
                    num_filters,
                    conv_dim=conv_dim,
                    kernel_size=kernels[1],
                    dilation=dilations[1],
                    bias=bias[1],
                    nonlinear="relu",
                )
            )
        self.scaling_layers = torch.nn.ModuleList()
        for _ in range(num_pool_layers):
            self.scaling_layers.append(
                ConvNonlinear(
                    num_filters,
                    num_filters,
                    conv_dim=conv_dim,
                    kernel_size=kernels[1],
                    dilation=dilations[1],
                    bias=bias[1],
                    nonlinear=None,
                )
            )
        self.output_layer = ConvNonlinear(
            num_filters,
            num_filters,
            conv_dim=conv_dim,
            kernel_size=kernels[1],
            dilation=dilations[1],
            bias=bias[1],
            nonlinear=None,
        )
        self.final_layer = ConvNonlinear(
            num_filters,
            out_channels,
            conv_dim=conv_dim,
            kernel_size=kernels[2],
            dilation=dilations[2],
            bias=bias[2],
            nonlinear=None,
        )
        self.num_pool_layers = num_pool_layers
        self.regularization_factor = torch.nn.Parameter(torch.tensor(regularization_factor))

    def forward(self, input_data: torch.Tensor):
        """Forward pass of the resnet."""
        intermediate_outputs = {"layer0": self.input_layer(input_data)}
        for i in range(self.num_pool_layers):
            output = self.layers[i](intermediate_outputs["layer" + str(i)])
            output = self.scaling_layers[i](output) * self.regularization_factor
            intermediate_outputs["layer" + str(i + 1)] = output + intermediate_outputs["layer" + str(i)]
        output = self.output_layer(intermediate_outputs["layer" + str(i + 1)]) + intermediate_outputs["layer0"]
        output = self.final_layer(output).permute(0, 2, 3, 1)
        return output[..., 0] + 1j * output[..., 1]
