# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
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
from mridc.collections.reconstruction.models.base import BaseSensitivityModel, DistributedMetricSum
import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block
import mridc.collections.motioncorrection.parts.transforms as transforms
import mridc.core.classes.modelPT as modelPT
import mridc.utils.model_utils as model_utils

wandb.require("service")

__all__ = ["BaseMRIMoCoReconstructionModel"]


class BaseMRIMoCoReconstructionModel(modelPT.ModelPT, ABC):
    """Base class of all MRIMoCoReconstruction models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                cfg_dict.get("sens_chans"),
                cfg_dict.get("sens_pools"),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
                coil_dim=self.coil_dim,
                mask_type=cfg_dict.get("sens_mask_type"),
                normalize=cfg_dict.get("sens_normalize"),
                mask_center=cfg_dict.get("sens_mask_center"),
            )

        self.MSE = DistributedMetricSum()
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()

        # Set evaluation metrics dictionaries
        self.mse_vals: Dict = defaultdict(dict)
        self.nmse_vals: Dict = defaultdict(dict)
        self.ssim_vals: Dict = defaultdict(dict)
        self.psnr_vals: Dict = defaultdict(dict)

        self.log_images = cfg_dict.get("log_images", True)

    def process_loss(self, target, pred, _loss_fn, mask=None):
        """
        Processes the loss.

        Parameters
        ----------
        target: Target data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred: Final prediction(s).
            list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        _loss_fn: Loss function.
            torch.nn.Module, default torch.nn.L1Loss()
        mask: Mask for the loss.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        loss: torch.FloatTensor, shape [1]
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
        """
        target = torch.abs(target / torch.max(torch.abs(target)))
        pred = torch.abs(pred / torch.max(torch.abs(pred)))

        if "ssim" in str(_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def loss_fn(x, y):
                """Calculate the ssim loss."""
                y = torch.abs(y / torch.max(torch.abs(y)))
                return _loss_fn(
                    x.unsqueeze(dim=self.coil_dim),
                    y.unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                y = torch.abs(y / torch.max(torch.abs(y)))
                return _loss_fn(x, y)

        return loss_fn(target, pred)

    @staticmethod
    def process_inputs(y, mask, init_pred):
        """
        Processes the inputs to the method.

        Parameters
        ----------
        y: Subsampled k-space data.
            list of torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            list of torch.Tensor, shape [1, 1, n_x, n_y, 1]
        init_pred: Initial prediction.
            list of torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        y: Subsampled k-space data.
            randomly selected y
        mask: Sampling mask.
            randomly selected mask
        init_pred: Initial prediction.
            randomly selected init_pred
        r: Random index.
        """
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
            init_pred = init_pred[r]
        else:
            r = 0
        return y, mask, init_pred, r

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a training step.

        Parameters
        ----------
        batch: Batch of data.
            Dict[str, torch.Tensor], with keys,

            'y': subsampled kspace,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'sensitivity_maps': sensitivity_maps,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': sampling mask,
                torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_pred': initial prediction. For example zero-filled or PICS.
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'target': target data,
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'phase_shift': phase shift for simulated motion,
                torch.Tensor
            'fname': filename,
                str, shape [batch_size]
            'slice_idx': slice_idx,
                torch.Tensor, shape [batch_size]
            'acc': acceleration factor,
                torch.Tensor, shape [batch_size]
            'max_value': maximum value of the magnitude image space,
                torch.Tensor, shape [batch_size]
            'crop_size': crop size,
                torch.Tensor, shape [n_x, n_y]
        batch_idx: Batch index.
            int

        Returns
        -------
        Dict[str, torch.Tensor], with keys,
        'loss': loss,
            torch.Tensor, shape [1]
        'log': log,
            dict, shape [1]
        """
        kspace, y, sensitivity_maps, mask, init_pred, target, _, _, acc = batch

        y, mask, init_pred, r = self.process_inputs(y, mask, init_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        if self.accumulate_estimates:
            try:
                preds = next(preds)
            except StopIteration:
                pass

            train_loss = sum(self.process_loss(target, preds, _loss_fn=self.train_loss_fn, mask=None))
        else:
            train_loss = self.process_loss(target, preds, _loss_fn=self.train_loss_fn, mask=None)

        acc = r if r != 0 else acc
        tensorboard_logs = {
            f"train_loss_{acc}x": train_loss.item(),  # type: ignore
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """
        Performs a validation step.

        Parameters
        ----------
        batch: Batch of data. Dict[str, torch.Tensor], with keys,
            'y': subsampled kspace,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'sensitivity_maps': sensitivity_maps,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': sampling mask,
                torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_pred': initial prediction. For example zero-filled or PICS.
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'target': target data,
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'phase_shift': phase shift for simulated motion,
                torch.Tensor
            'fname': filename,
                str, shape [batch_size]
            'slice_idx': slice_idx,
                torch.Tensor, shape [batch_size]
            'acc': acceleration factor,
                torch.Tensor, shape [batch_size]
            'max_value': maximum value of the magnitude image space,
                torch.Tensor, shape [batch_size]
            'crop_size': crop size,
                torch.Tensor, shape [n_x, n_y]
        batch_idx: Batch index.
            int

        Returns
        -------
        Dict[str, torch.Tensor], with keys,
        'loss': loss,
            torch.Tensor, shape [1]
        'log': log,
            dict, shape [1]
        """
        kspace, y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _ = batch

        y, mask, init_pred, r = self.process_inputs(y, mask, init_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        if self.accumulate_estimates:
            try:
                preds = next(preds)
            except StopIteration:
                pass
            val_loss = sum(self.process_loss(target, preds, _loss_fn=self.val_loss_fn, mask=None))
        else:
            val_loss = self.process_loss(target, preds, _loss_fn=self.val_loss_fn, mask=None)

        # Cascades
        if isinstance(preds, list):
            preds = preds[-1]

        # Time-steps
        if isinstance(preds, list):
            preds = preds[-1]

        key = f"{fname[0]}_images_idx_{int(slice_num)}"  # type: ignore
        output = torch.abs(preds).detach().cpu()
        target = torch.abs(target).detach().cpu()
        output = output / output.max()  # type: ignore
        target = target / target.max()  # type: ignore

        if self.log_images:
            error = torch.abs(target - output)
            self.log_image(f"{key}/target", target)
            self.log_image(f"{key}/reconstruction", output)
            self.log_image(f"{key}/error", error)

        target = target.numpy()  # type: ignore
        output = output.numpy()  # type: ignore
        self.mse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.mse(target, output)).view(1)
        self.nmse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.nmse(target, output)).view(1)
        self.ssim_vals[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(target, output, maxval=output.max() - output.min())
        ).view(1)
        self.psnr_vals[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(target, output, maxval=output.max() - output.min())
        ).view(1)

        return {"val_loss": val_loss}

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """
        Performs a test step.

        Parameters
        ----------
        batch: Batch of data. Dict[str, torch.Tensor], with keys,
            'y': subsampled kspace,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'sensitivity_maps': sensitivity_maps,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': sampling mask,
                torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_pred': initial prediction. For example zero-filled or PICS.
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'target': target data,
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'phase_shift': phase shift for simulated motion,
                torch.Tensor
            'fname': filename,
                str, shape [batch_size]
            'slice_idx': slice_idx,
                torch.Tensor, shape [batch_size]
            'acc': acceleration factor,
                torch.Tensor, shape [batch_size]
            'max_value': maximum value of the magnitude image space,
                torch.Tensor, shape [batch_size]
            'crop_size': crop size,
                torch.Tensor, shape [n_x, n_y]
        batch_idx: Batch index.
            int

        Returns
        -------
        name: Name of the volume.
            str
        slice_num: Slice number.
            int
        pred: Predicted data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        """
        kspace, y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _ = batch

        y, mask, init_pred, r = self.process_inputs(y, mask, init_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        if self.accumulate_estimates:
            try:
                preds = next(preds)
            except StopIteration:
                pass

        # Cascades
        if isinstance(preds, list):
            preds = preds[-1]

        # Time-steps
        if isinstance(preds, list):
            preds = preds[-1]

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore

        output = torch.abs(preds).detach().cpu()
        output = output / output.max()  # type: ignore

        target = torch.abs(target).detach().cpu()
        target = target / target.max()  # type: ignore

        if self.log_images:
            error = torch.abs(target - output)
            self.log_image(f"{key}/target", target)
            self.log_image(f"{key}/reconstruction", output)
            self.log_image(f"{key}/error", error)

        target = target.numpy()  # type: ignore
        output = output.numpy()  # type: ignore
        self.mse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.mse(target, output)).view(1)
        self.nmse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.nmse(target, output)).view(1)
        self.ssim_vals[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(target, output, maxval=output.max() - output.min())
        ).view(1)
        self.psnr_vals[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(target, output, maxval=output.max() - output.min())
        ).view(1)

        return name, slice_num, preds.detach().cpu().numpy()

    def log_image(self, name, image):
        """
        Logs an image.

        Parameters
        ----------
        name: Name of the image.
            str
        image: Image to log.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        """
        if image.dim() > 3:
            image = image[0, 0, :, :].unsqueeze(0)
        elif image.shape[0] != 1:
            image = image[0].unsqueeze(0)

        if "wandb" in self.logger.__module__.lower():
            if image.is_cuda:
                image = image.detach().cpu()
            self.logger.experiment.log({name: wandb.Image(image.numpy())})
        else:
            self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation epoch to aggregate outputs.

        Parameters
        ----------
        outputs: List of outputs of the validation batches.
            list of dicts

        Returns
        -------
        metrics: Dictionary of metrics.
            dict
        """
        self.log("val_loss", torch.stack([x["val_loss"] for x in outputs]).mean())

        # Log metrics.
        # Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)

        for k in self.mse_vals.keys():
            mse_vals[k].update(self.mse_vals[k])
        for k in self.nmse_vals.keys():
            nmse_vals[k].update(self.nmse_vals[k])
        for k in self.ssim_vals.keys():
            ssim_vals[k].update(self.ssim_vals[k])
        for k in self.psnr_vals.keys():
            psnr_vals[k].update(self.psnr_vals[k])

        # apply means across image volumes
        metrics = {"MSE": 0, "NMSE": 0, "SSIM": 0, "PSNR": 0}
        local_examples = 0
        for fname in mse_vals:
            local_examples += 1
            metrics["MSE"] = metrics["MSE"] + torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            metrics["NMSE"] = metrics["NMSE"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["SSIM"] = metrics["SSIM"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["PSNR"] = metrics["PSNR"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"] = self.MSE(metrics["MSE"])
        metrics["NMSE"] = self.NMSE(metrics["NMSE"])
        metrics["SSIM"] = self.SSIM(metrics["SSIM"])
        metrics["PSNR"] = self.PSNR(metrics["PSNR"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics.items():
            self.log(f"{metric}", value / tot_examples)

    def test_epoch_end(self, outputs):
        """
        Called at the end of test epoch to aggregate outputs.

        Parameters
        ----------
        outputs: List of outputs of the test batches.
            list of dicts

        Returns
        -------
        Saves the reconstructed images to .h5 files.
        """
        # Log metrics.
        # Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)

        for k in self.mse_vals.keys():
            mse_vals[k].update(self.mse_vals[k])
        for k in self.nmse_vals.keys():
            nmse_vals[k].update(self.nmse_vals[k])
        for k in self.ssim_vals.keys():
            ssim_vals[k].update(self.ssim_vals[k])
        for k in self.psnr_vals.keys():
            psnr_vals[k].update(self.psnr_vals[k])

        # apply means across image volumes
        metrics = {"MSE": 0, "NMSE": 0, "SSIM": 0, "PSNR": 0}
        local_examples = 0
        for fname in mse_vals:
            local_examples += 1
            metrics["MSE"] = metrics["MSE"] + torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            metrics["NMSE"] = metrics["NMSE"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["SSIM"] = metrics["SSIM"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["PSNR"] = metrics["PSNR"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"] = self.MSE(metrics["MSE"])
        metrics["NMSE"] = self.NMSE(metrics["NMSE"])
        metrics["SSIM"] = self.SSIM(metrics["SSIM"])
        metrics["PSNR"] = self.PSNR(metrics["PSNR"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics.items():
            self.log(f"{metric}", value / tot_examples)

        reconstructions = defaultdict(list)
        for fname, slice_num, output in outputs:
            reconstructions[fname].append((slice_num, output))

        for fname in reconstructions:
            reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])  # type: ignore

        out_dir = Path(os.path.join(self.logger.log_dir, "reconstructions"))
        out_dir.mkdir(exist_ok=True, parents=True)
        for fname, recons in reconstructions.items():
            with h5py.File(out_dir / fname, "w") as hf:
                hf.create_dataset("reconstruction", data=recons)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        """
        Setups the training data.

        Parameters
        ----------
        train_data_config: Training data configuration.
            dict

        Returns
        -------
        train_data: Training data.
            torch.utils.data.DataLoader
        """
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        """
        Setups the validation data.

        Parameters
        ----------
        val_data_config: Validation data configuration.
            dict

        Returns
        -------
        val_data: Validation data.
            torch.utils.data.DataLoader
        """
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        """
        Setups the test data.

        Parameters
        ----------
        test_data_config: Test data configuration.
            dict

        Returns
        -------
        test_data: Test data.
            torch.utils.data.DataLoader
        """
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

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
            transform=transforms.MRIMoCoDataTransforms(
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
