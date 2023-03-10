# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from math import sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.data.subsample as subsample
from mridc.collections.motioncorrection.parts.motionsimulation import MotionSimulation
from mridc.collections.reconstruction.parts.transforms import GeometricDecompositionCoilCompression, NoisePreWhitening

__all__ = ["MRIMoCoDataTransforms"]


class MRIMoCoDataTransforms:
    """MRI preprocessing data transforms."""

    def __init__(
        self,
        apply_prewhitening: bool = False,
        prewhitening_scale_factor: float = 1.0,
        prewhitening_patch_start: int = 10,
        prewhitening_patch_length: int = 30,
        apply_gcc: bool = False,
        gcc_virtual_coils: int = 10,
        gcc_calib_lines: int = 24,
        gcc_align_data: bool = True,
        coil_combination_method: str = "SENSE",
        dimensionality: int = 2,
        mask_func: Optional[List[subsample.MaskFunc]] = None,
        shift_mask: bool = False,
        mask_center_scale: Optional[float] = 0.02,
        half_scan_percentage: float = 0.0,
        remask: bool = False,
        crop_size: Optional[Tuple[int, int]] = None,
        kspace_crop: bool = False,
        crop_before_masking: bool = True,
        kspace_zero_filling_size: Optional[Tuple] = None,
        normalize_inputs: bool = False,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        max_norm: bool = True,
        spatial_dims: Sequence[int] = None,
        coil_dim: int = 0,
        use_seed: bool = True,
        
        # TODO: Add args to class comment
        random_motion: bool = False,
        random_motion_type: str = "gaussian",
        random_motion_angle: int = 10,
        random_motion_translation: int = 10,
        random_motion_center_percentage: float = 0.02,
        random_motion_motion_percentage: list = [20, 20],
        random_motion_num_segments: int = 8,
        random_motion_random_num_segments: bool = True,
        random_motion_non_uniform: bool = False,
    ):
        """
        Initialize the data transform.

        Parameters
        ----------
        apply_prewhitening : bool
            Whether to apply prewhitening.
        prewhitening_scale_factor : float
            The scale factor for the prewhitening.
        prewhitening_patch_start : int
            The start index for the prewhitening patch.
        prewhitening_patch_length : int
            The length of the prewhitening patch.
        apply_gcc : bool
            Whether to apply GCC.
        gcc_virtual_coils : int
            The number of virtual coils.
        gcc_calib_lines : int
            The number of calibration lines.
        gcc_align_data : bool
            Whether to align the data.
        coil_combination_method : str
            The coil combination method.
        dimensionality : int
            The dimensionality of the data.
        mask_func : Optional[List[subsample.MaskFunc]]
            The mask functions.
        shift_mask : bool
            Whether to shift the mask.
        mask_center_scale : Optional[float]
            The scale for the mask center.
        half_scan_percentage : float
            The percentage of the scan to use.
        remask : bool
            Whether to remask the data.
        crop_size : Optional[Tuple[int, int]]
            The crop size.
        kspace_crop : bool
            Whether to crop the kspace.
        crop_before_masking : bool
            Whether to crop before masking.
        kspace_zero_filling_size : Optional[Tuple]
            The zero filling size.
        normalize_inputs : bool
            Whether to normalize the inputs.
        fft_centered : bool
            Whether to center the FFT.
        fft_normalization : str
            The FFT normalization.
        max_norm : bool
            Whether to apply max norm.
        spatial_dims : Sequence[int]
            The spatial dimensions.
        coil_dim : int
            The coil dimension.
        use_seed : bool
            Whether to use a seed.
        """
        self.coil_combination_method = coil_combination_method
        self.dimensionality = dimensionality
        self.mask_func = mask_func
        self.shift_mask = shift_mask
        self.mask_center_scale = mask_center_scale
        self.half_scan_percentage = half_scan_percentage
        self.remask = remask
        self.crop_size = crop_size
        self.kspace_crop = kspace_crop
        self.crop_before_masking = crop_before_masking
        self.kspace_zero_filling_size = kspace_zero_filling_size
        self.normalize_inputs = normalize_inputs
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.max_norm = max_norm
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim - 1 if self.dimensionality == 2 else coil_dim

        self.apply_prewhitening = apply_prewhitening
        self.prewhitening = (
            NoisePreWhitening(
                patch_size=[
                    prewhitening_patch_start,
                    prewhitening_patch_length + prewhitening_patch_start,
                    prewhitening_patch_start,
                    prewhitening_patch_length + prewhitening_patch_start,
                ],
                scale_factor=prewhitening_scale_factor,
            )
            if apply_prewhitening
            else None
        )

        self.gcc = (
            GeometricDecompositionCoilCompression(
                virtual_coils=gcc_virtual_coils,
                calib_lines=gcc_calib_lines,
                align_data=gcc_align_data,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            if apply_gcc
            else None
        )

        # Motion compensation
        self.random_motion = random_motion
        self.random_motion_type = random_motion_type
        self.random_motion_angle = random_motion_angle
        self.random_motion_translation = random_motion_translation
        self.random_motion_center_percentage = random_motion_center_percentage
        self.random_motion_motion_percentage = random_motion_motion_percentage
        self.random_motion_num_segments = random_motion_num_segments
        self.random_motion_random_num_segments = random_motion_random_num_segments
        self.random_motion_non_uniform = random_motion_non_uniform

        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        eta: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_idx: int,
    ) -> Tuple[
        torch.Tensor,
        Union[Union[List, torch.Tensor], torch.Tensor],
        Union[Optional[torch.Tensor], Any],
        Union[List, Any],
        Union[Optional[torch.Tensor], Any],
        Union[torch.Tensor, Any],
        str,
        int,
        Union[List, Any],
    ]:
        """
        Apply the data transform.

        Parameters
        ----------
        kspace: The kspace.
        sensitivity_map: The sensitivity map.
        mask: The mask.
        eta: The initial estimation.
        target: The target.
        attrs: The attributes.
        fname: The file name.
        slice_idx: The slice number.

        Returns
        -------
        The transformed data.
        """
        kspace = utils.to_tensor(kspace)

        # This condition is necessary in case of auto estimation of sense maps.
        if sensitivity_map is not None and sensitivity_map.size != 0:
            sensitivity_map = utils.to_tensor(sensitivity_map)

        if self.apply_prewhitening:
            kspace = self.prewhitening(kspace)  # type: ignore

        if self.gcc is not None:
            kspace = self.gcc(kspace)
            if isinstance(sensitivity_map, torch.Tensor):
                sensitivity_map = fft.ifft2(
                    self.gcc(
                        fft.fft2(
                            sensitivity_map,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        )
                    ),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )

        # Apply zero-filling on kspace
        if self.kspace_zero_filling_size is not None and self.kspace_zero_filling_size not in ("", "None"):
            padding_top = np.floor_divide(abs(int(self.kspace_zero_filling_size[0]) - kspace.shape[1]), 2)
            padding_bottom = padding_top
            padding_left = np.floor_divide(abs(int(self.kspace_zero_filling_size[1]) - kspace.shape[2]), 2)
            padding_right = padding_left

            kspace = torch.view_as_complex(kspace)
            kspace = torch.nn.functional.pad(
                kspace, pad=(padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0
            )
            kspace = torch.view_as_real(kspace)

            sensitivity_map = fft.fft2(
                sensitivity_map,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            sensitivity_map = torch.view_as_complex(sensitivity_map)
            sensitivity_map = torch.nn.functional.pad(
                sensitivity_map,
                pad=(padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
            sensitivity_map = torch.view_as_real(sensitivity_map)
            sensitivity_map = fft.ifft2(
                sensitivity_map,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        # Initial estimation
        eta = utils.to_tensor(eta) if eta is not None and eta.size != 0 else torch.tensor([])

        # If the target is not given, we need to compute it.
        if self.coil_combination_method.upper() == "RSS":
            target = utils.rss(
                fft.ifft2(
                    kspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                dim=self.coil_dim,
            )
        elif self.coil_combination_method.upper() == "SENSE":
            if sensitivity_map is not None and sensitivity_map.size != 0:
                target = utils.sense(
                    fft.ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_map,
                    dim=self.coil_dim,
                )
        elif target is not None and target.size != 0:
            target = utils.to_tensor(target)
        elif "target" in attrs or "target_rss" in attrs:
            target = torch.tensor(attrs["target"])
        else:
            raise ValueError("No target found")

        target = torch.view_as_complex(target)
        target = torch.abs(target / torch.max(torch.abs(target)))

        seed = tuple(map(ord, fname)) if self.use_seed else None
        acq_start = attrs["padding_left"] if "padding_left" in attrs else 0
        acq_end = attrs["padding_right"] if "padding_left" in attrs else 0

        # This should be outside the condition because it needs to be returned in the end, even if cropping is off.
        # crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])
        crop_size = target.shape
        if self.crop_size is not None and self.crop_size not in ("", "None"):
            # Check for smallest size against the target shape.
            h = min(int(self.crop_size[0]), target.shape[0])
            w = min(int(self.crop_size[1]), target.shape[1])

            # Check for smallest size against the stored recon shape in metadata.
            if crop_size[0] != 0:
                h = h if h <= crop_size[0] else crop_size[0]
            if crop_size[1] != 0:
                w = w if w <= crop_size[1] else crop_size[1]

            self.crop_size = (int(h), int(w))

            target = utils.center_crop(target, self.crop_size)
            if sensitivity_map is not None and sensitivity_map.size != 0:
                sensitivity_map = (
                    fft.ifft2(
                        utils.complex_center_crop(
                            fft.fft2(
                                sensitivity_map,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            ),
                            self.crop_size,
                        ),
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                    if self.kspace_crop
                    else utils.complex_center_crop(sensitivity_map, self.crop_size)
                )

            if eta is not None and eta.ndim > 2:
                eta = (
                    fft.ifft2(
                        utils.complex_center_crop(
                            fft.fft2(
                                eta,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            ),
                            self.crop_size,
                        ),
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                    if self.kspace_crop
                    else utils.complex_center_crop(eta, self.crop_size)
                )

        # Cropping before masking will maintain the shape of original kspace intact for masking.
        if self.crop_size is not None and self.crop_size not in ("", "None") and self.crop_before_masking:
            kspace = (
                utils.complex_center_crop(kspace, self.crop_size)
                if self.kspace_crop
                else fft.fft2(
                    utils.complex_center_crop(
                        fft.ifft2(
                            kspace,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        ),
                        self.crop_size,
                    ),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
            )

        # Motion simulation
        if self.random_motion:
            motion_layer = MotionSimulation(
                type=self.random_motion_type,
                angle=self.random_motion_angle,
                translation=self.random_motion_translation,
                center_percentage=self.random_motion_center_percentage,
                motion_percentage=self.random_motion_motion_percentage,
                spatial_dims=self.spatial_dims,
                num_segments=self.random_motion_num_segments,
                random_num_segments=self.random_motion_random_num_segments,
                non_uniform=self.random_motion_non_uniform,
            )
            kspace = motion_layer.forward(kspace)


        if not utils.is_none(mask):
            for _mask in mask:
                if list(_mask.shape) == [kspace.shape[-3], kspace.shape[-2]]:
                    mask = torch.from_numpy(_mask).unsqueeze(0).unsqueeze(-1)
                    break

            padding = (acq_start, acq_end)
            if (not utils.is_none(padding[0]) and not utils.is_none(padding[1])) and padding[0] != 0:
                mask[:, :, : padding[0]] = 0
                # padding value inclusive on right of zeros
                mask[:, :, padding[1] :] = 0

            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)

            if self.shift_mask:
                mask = torch.fft.fftshift(mask, dim=(self.spatial_dims[0] - 1, self.spatial_dims[1] - 1))

            if self.crop_size is not None and self.crop_size not in ("", "None") and self.crop_before_masking:
                mask = utils.complex_center_crop(mask, self.crop_size)

            masked_kspace = kspace * mask + 0.0  # the + 0.0 removes the sign of the zeros

            acc = 1
        elif utils.is_none(self.mask_func):
            masked_kspace = kspace.clone()
            acc = torch.tensor([1])

            if mask is None:
                mask = torch.ones(masked_kspace.shape[-3], masked_kspace.shape[-2]).type(torch.float32)
            else:
                mask = torch.from_numpy(mask)

                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)

                if mask.shape[0] == masked_kspace.shape[2]:  # type: ignore
                    mask = mask.permute(1, 0)
                elif mask.shape[0] != masked_kspace.shape[1]:  # type: ignore
                    mask = torch.ones(
                        [masked_kspace.shape[-3], masked_kspace.shape[-2]], dtype=torch.float32  # type: ignore
                    )

            if mask.shape[-2] == 1:  # 1D mask
                mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)
            else:  # 2D mask
                # Crop loaded mask.
                if self.crop_size is not None and self.crop_size not in ("", "None"):
                    mask = utils.center_crop(mask, self.crop_size)
                mask = mask.unsqueeze(0).unsqueeze(-1)

            if self.shift_mask:
                mask = torch.fft.fftshift(mask, dim=(1, 2))

            masked_kspace = masked_kspace * mask
            mask = mask.byte()
        elif isinstance(self.mask_func, list):
            masked_kspaces = []
            masks = []
            accs = []
            for m in self.mask_func:
                if self.dimensionality == 2:
                    _masked_kspace, _mask, _acc = utils.apply_mask(
                        kspace,
                        m,
                        seed,
                        (acq_start, acq_end),
                        shift=self.shift_mask,
                        half_scan_percentage=self.half_scan_percentage,
                        center_scale=self.mask_center_scale,
                    )
                elif self.dimensionality == 3:
                    _masked_kspace = []
                    _mask = None
                    for i in range(kspace.shape[0]):
                        _i_masked_kspace, _i_mask, _i_acc = utils.apply_mask(
                            kspace[i],
                            m,
                            seed,
                            (acq_start, acq_end),
                            shift=self.shift_mask,
                            half_scan_percentage=self.half_scan_percentage,
                            center_scale=self.mask_center_scale,
                            existing_mask=_mask,
                        )
                        if self.remask:
                            _mask = _i_mask
                        if i == 0:
                            _acc = _i_acc
                        _masked_kspace.append(_i_masked_kspace)
                    _masked_kspace = torch.stack(_masked_kspace, dim=0)
                    _mask = _i_mask.unsqueeze(0)
                else:
                    raise ValueError(f"Unsupported data dimensionality {self.dimensionality}D.")
                masked_kspaces.append(_masked_kspace)
                masks.append(_mask.byte())
                accs.append(_acc)
            masked_kspace = masked_kspaces
            mask = masks
            acc = accs  # type: ignore
        else:
            masked_kspace, mask, acc = utils.apply_mask(
                kspace,
                self.mask_func[0],  # type: ignore
                seed,
                (acq_start, acq_end),
                shift=self.shift_mask,
                half_scan_percentage=self.half_scan_percentage,
                center_scale=self.mask_center_scale,
            )
            mask = mask.byte()

        # Cropping after masking.
        if self.crop_size is not None and self.crop_size not in ("", "None") and not self.crop_before_masking:
            kspace = (
                utils.complex_center_crop(kspace, self.crop_size)
                if self.kspace_crop
                else fft.fft2(
                    utils.complex_center_crop(
                        fft.ifft2(
                            kspace,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        ),
                        self.crop_size,
                    ),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
            )

            masked_kspace = (
                utils.complex_center_crop(masked_kspace, self.crop_size)
                if self.kspace_crop
                else fft.fft2(
                    utils.complex_center_crop(
                        fft.ifft2(
                            masked_kspace,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        ),
                        self.crop_size,
                    ),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
            )

            mask = utils.center_crop(mask.squeeze(-1), self.crop_size).unsqueeze(-1)

        # Normalize by the max value.
        if self.normalize_inputs:
            if isinstance(self.mask_func, list):
                if self.fft_normalization in ("backward", "ortho", "forward"):
                    imspace = fft.ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                    if self.max_norm:
                        imspace = imspace / torch.max(torch.abs(imspace))
                    kspace = fft.fft2(
                        imspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                elif self.fft_normalization in ("none", None) and self.max_norm:
                    imspace = torch.fft.ifftn(torch.view_as_complex(kspace), dim=list(self.spatial_dims), norm=None)
                    imspace = imspace / torch.max(torch.abs(imspace))
                    kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))

                masked_kspaces = []
                for y in masked_kspace:
                    if self.fft_normalization in ("backward", "ortho", "forward"):
                        imspace = fft.ifft2(
                            y,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        )
                        if self.max_norm:
                            imspace = imspace / torch.max(torch.abs(imspace))
                        y = fft.fft2(
                            imspace,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        )
                    elif self.fft_normalization in ("none", None) and self.max_norm:
                        imspace = torch.fft.ifftn(torch.view_as_complex(y), dim=list(self.spatial_dims), norm=None)
                        imspace = imspace / torch.max(torch.abs(imspace))
                        y = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))
                    masked_kspaces.append(y)
                masked_kspace = masked_kspaces
            elif self.fft_normalization in ("backward", "ortho", "forward"):
                imspace = fft.ifft2(
                    kspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                if self.max_norm:
                    imspace = imspace / torch.max(torch.abs(imspace))
                kspace = fft.fft2(
                    imspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                imspace = fft.ifft2(
                    masked_kspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                if self.max_norm:
                    imspace = imspace / torch.max(torch.abs(imspace))
                masked_kspace = fft.fft2(
                    imspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
            elif self.fft_normalization in ("none", None) and self.max_norm:
                imspace = torch.fft.ifftn(torch.view_as_complex(masked_kspace), dim=list(self.spatial_dims), norm=None)
                imspace = imspace / torch.max(torch.abs(imspace))
                masked_kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))

                imspace = torch.fft.ifftn(torch.view_as_complex(kspace), dim=list(self.spatial_dims), norm=None)
                imspace = imspace / torch.max(torch.abs(imspace))
                kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))

            if self.max_norm:
                if sensitivity_map.size != 0:
                    sensitivity_map = sensitivity_map / torch.max(torch.abs(sensitivity_map))

                if eta.size != 0 and eta.ndim > 2:
                    eta = eta / torch.max(torch.abs(eta))

                target = target / torch.max(torch.abs(target))

        return kspace, masked_kspace, sensitivity_map, mask, eta, target, fname, slice_idx, acc
