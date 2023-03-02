# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric

import mridc.collections.common.metrics.reconstruction_metrics as reconstruction_metrics
import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.data.mri_data as mri_data
import mridc.collections.reconstruction.data.subsample as subsample
import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block
import mridc.collections.motioncompensation.parts.transforms as transforms
import mridc.core.classes.modelPT as modelPT
import mridc.utils.model_utils as model_utils

wandb.require("service")

__all__ = ["BaseMRIMoCoReconstructionModel"]


class BaseMRIMoCoReconstructionModel(BaseMRIReconstructionModel):
    """Base class of all BaseMRIMoCoReconstructionModel models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """
        Setups the dataloader from the configuration (yaml) file.

        Parameters
        ----------
        cfg: Configuration file.
            dict

        Returns
        -------
        dataloader: DataLoader.
            torch.utils.data.DataLoader
        """
        mask_root = cfg.get("mask_path")
        mask_args = cfg.get("mask_args")
        shift_mask = mask_args.get("shift_mask")
        mask_type = mask_args.get("type")

        mask_func = None  # type: ignore
        mask_center_scale = 0.02

        if utils.is_none(mask_root) and not utils.is_none(mask_type):
            accelerations = mask_args.get("accelerations")
            center_fractions = mask_args.get("center_fractions")
            mask_center_scale = mask_args.get("scale")

            mask_func = (
                [
                    subsample.create_mask_for_mask_type(mask_type, [cf] * 2, [acc] * 2)
                    for acc, cf in zip(accelerations, center_fractions)
                ]
                if len(accelerations) >= 2
                else [subsample.create_mask_for_mask_type(mask_type, center_fractions, accelerations)]
            )

        dataset = mri_data.MRISliceDataset(
            root=cfg.get("data_path"),
            sense_root=cfg.get("sense_path"),
            mask_root=cfg.get("mask_path"),
            challenge=cfg.get("challenge"),
            transform=transforms.MRIDataTransforms(
                coil_combination_method=cfg.get("coil_combination_method"),
                dimensionality=cfg.get("dimensionality"),
                mask_func=mask_func,
                shift_mask=shift_mask,
                mask_center_scale=mask_center_scale,
                remask=cfg.get("remask"),
                normalize_inputs=cfg.get("normalize_inputs"),
                crop_size=cfg.get("crop_size"),
                crop_before_masking=cfg.get("crop_before_masking"),
                kspace_zero_filling_size=cfg.get("kspace_zero_filling_size"),
                fft_centered=cfg.get("fft_centered"),
                fft_normalization=cfg.get("fft_normalization"),
                max_norm=cfg.get("max_norm"),
                spatial_dims=cfg.get("spatial_dims"),
                coil_dim=cfg.get("coil_dim"),
                use_seed=cfg.get("use_seed"),
                random_motion=cfg.get("random_motion"),
                random_motion_type=cfg.get("random_motion_type"),
                random_motion_angle=cfg.get("random_motion_angle"),
                random_motion_translation=cfg.get("random_motion_translation"),
                random_motion_center_percentage=cfg.get("random_motion_center_percentage"),
                random_motion_motion_percentage=cfg.get("random_motion_motion_percentage"),
                random_motion_num_segments=cfg.get("random_motion_num_segments"),
                random_motion_random_num_segments=cfg.get("random_motion_random_num_segments"),
                random_motion_non_uniform=cfg.get("random_motion_non_uniform"),
            ),
            sample_rate=cfg.get("sample_rate"),
            consecutive_slices=cfg.get("consecutive_slices"),
        )
        if cfg.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.get("batch_size"),
            sampler=sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
