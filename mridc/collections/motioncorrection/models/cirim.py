# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from mridc.collections.motioncorrection.models.base import BaseMRIMoCoReconstructionModel
from mridc.collections.reconstruction.models.cirim import CIRIM


__all__ = ["MoCoCIRIM"]


class MoCoCIRIM(BaseMRIMoCoReconstructionModel, CIRIM, ABC):
    pass