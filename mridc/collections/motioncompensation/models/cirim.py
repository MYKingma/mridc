# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from abc import ABC
from typing import Generator, Union
from mridc.collections.motioncompensation.models.base import BaseMRIMoCoReconstructionModel
from mridc.collections.reconstruction.models.cirim import CIRIM

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

import mridc.collections.common.losses.ssim as losses
import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.rnn_utils as rnn_utils
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.models.base as base_models
import mridc.collections.reconstruction.models.rim.rim_block as rim_block
import mridc.core.classes.common as common_classes

__all__ = ["CIRIM"]


class MoCoCIRIM(BaseMRIMoCoReconstructionModel, CIRIM, ABC):
    pass
