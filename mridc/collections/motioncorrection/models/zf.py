# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from mridc.collections.motioncorrection.models.base import BaseMRIMoCoReconstructionModel
from mridc.collections.reconstruction.models.zf import ZF
import torch
import numpy as np


__all__ = ["MoCoZF"]


class MoCoZF(BaseMRIMoCoReconstructionModel, ZF, ABC):
    pass
