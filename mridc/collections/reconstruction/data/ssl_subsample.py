# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from https://github.com/byaman14/SSDU/blob/main/masks/ssdu_masks.py

import numpy as np
from numba import float32, int32, jit, types
from numba.core.extending import overload


@jit(nopython=True)
def index_flatten2nd(ind, shape):
    """
        Parameters
        ----------
        ind : 1D vector containing chosen locations.
        shape : shape of the matrix/tensor for mapping ind.
    q
        Returns
        -------
        list of >=2D indices containing non-zero locations
    """
    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


@jit(nopython=True, fastmath=True)
@overload(np.linalg.norm)
@overload(np.argsort)
def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Finds the center index of the kspace.

    Parameters
    ----------
    kspace : 3D tensor containing kspace data.
    axes : tuple of ints containing the axes to compute the center.

    Returns
    -------
    center_ind : 1D vector containing the indices of the center.
    """
    center_locs = []
    for axis in axes:
        for i in range(kspace.shape[axis]):
            if axis == 0:
                x = np.linalg.norm(kspace[i, :, :])
            elif axis == 1:
                x = np.linalg.norm(kspace[:, i, :])
            elif axis == 2:
                x = np.linalg.norm(kspace[:, :, i])
            center_locs.append(x)
    return np.argsort(np.asarray(center_locs))[-1:]


class SSDUMasker:
    _type: str  # type: ignore
    _rho: float32  # type: ignore
    _small_acs_block: types.List(int32, int32)  # type: ignore

    def __init__(
        self,
        type: str = "gaussian",
        rho: float32 = 0.4,  # type: ignore
        small_acs_block: types.List(int32, int32) = (4, 4),  # type: ignore
    ):
        self._type = type
        self._rho = rho
        self._small_acs_block = small_acs_block

    @staticmethod
    @jit(nopython=True)
    def Gaussian_selection(input_data, input_mask, _rho, _small_acs_block, std_scale=4):
        nrow, ncol = input_data.shape[0], input_data.shape[1]
        center_kx = find_center_ind(input_data, axes=(1, 2))[0]
        center_ky = find_center_ind(input_data, axes=(0, 2))[0]

        temp_mask = np.copy(input_mask)
        temp_mask[
            center_kx - _small_acs_block[0] // 2 : center_kx + _small_acs_block[0] // 2,
            center_ky - _small_acs_block[1] // 2 : center_ky + _small_acs_block[1] // 2,
        ] = 0

        loss_mask = np.zeros_like(input_mask)

        idx = np.int(np.ceil(np.sum(input_mask[:]) * _rho))
        for count in range(idx):
            indx = np.int(np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / std_scale)))
            indy = np.int(np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / std_scale)))

            if 0 <= indx < nrow and 0 <= indy < ncol and temp_mask[indx, indy] == 1 and loss_mask[indx, indy] != 1:
                loss_mask[indx, indy] = 1

        trn_mask = input_mask - loss_mask

        return trn_mask, loss_mask, [center_kx, center_ky]

    @staticmethod
    @jit(nopython=True)
    def uniform_selection(input_data, input_mask, _rho, _small_acs_block):
        nrow, ncol = input_data.shape[0], input_data.shape[1]

        center_kx = find_center_ind(input_data, axes=(1, 2))[0]
        center_ky = find_center_ind(input_data, axes=(0, 2))[0]

        temp_mask = np.copy(input_mask)
        temp_mask[
            center_kx - _small_acs_block[0] // 2 : center_kx + _small_acs_block[0] // 2,
            center_ky - _small_acs_block[1] // 2 : center_ky + _small_acs_block[1] // 2,
        ] = 0

        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(
            np.arange(nrow * ncol), size=np.int(np.count_nonzero(pr) * _rho), replace=False, p=pr / np.sum(pr)
        )

        [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

        loss_mask = np.zeros_like(input_mask)
        loss_mask[ind_x, ind_y] = 1

        trn_mask = input_mask - loss_mask

        return trn_mask, loss_mask, [center_kx, center_ky]

    def __call__(self, input_data, input_mask):
        if self._type == "gaussian":
            mask, loss_mask, coords = self.Gaussian_selection(input_data, input_mask, self._rho, self._small_acs_block)
        elif self._type == "uniform":
            mask, loss_mask, coords = self.uniform_selection(input_data, input_mask, self._rho, self._small_acs_block)
        else:
            raise ValueError(
                "Unknown type of self-supervised masking. "
                f"For now, only gaussian and uniform are supported, found {self._type}."
            )
        return mask, loss_mask, coords
