from typing import Callable
import logging

import numpy as np

import gunpowder as gp
from .batch_filter import BatchFilter
from gunpowder.array import ArrayKey

logger = logging.getLogger(__name__)

class ReduceDim(BatchFilter):

    def __init__(self,
        in_array: ArrayKey,
        out_array: ArrayKey,
        proj_funct: Callable[[np.ndarray, int], np.ndarray],
        axis: int = 0,
        ax_offset: int = 0,
        ax_size: int = 1,
    ) -> None:

        self.in_array = in_array
        self.out_array = out_array

        self.axis = axis
        self.proj_funct = proj_funct
        self.ax_offset = ax_offset
        self.ax_size = ax_size

    def setup(self):

        spec = self.spec[self.in_array].copy()
        shape, offset, voxel_size = tuple(
            tuple(x for i, x in enumerate(vec) if i != self.axis)
                for vec in [
                    spec.roi.get_shape(),
                    spec.roi.get_offset(),
                    spec.voxel_size]
        )

        spec.roi = gp.Roi(offset, shape)
        spec.voxel_size = voxel_size

        self.provides(
            self.out_array,
            spec)


    def prepare(self, request):

        # to deliver reduced dims, we need the raw data with extented ROI
        deps = gp.BatchRequest()
        deps[self.in_array] = request[self.out_array].copy()

        requested_roi = request[self.out_array].roi
        offset, shape = requested_roi.get_offset(), requested_roi.get_shape()
        offset = offset[:self.axis] + (self.ax_offset,) + offset[self.axis:]
        shape = shape[:self.axis] + (self.ax_size,) + shape[self.axis:]

        deps[self.in_array].roi = gp.Roi(offset, shape)

        return deps

    def process(self, batch, request):

        # get the data from in_array and reduce the dims
        data = self.proj_funct(batch[self.in_array].data, self.axis)

        # create the array spec for the new array
        spec = batch[self.in_array].spec.copy()
        spec.roi = request[self.out_array].roi.copy()

        spec.voxel_size = tuple(
            v for i, v in enumerate(spec.voxel_size)
                if i != self.axis)

        # create a new batch to hold the new array
        batch = gp.Batch()

        # create a new array
        reduced = gp.Array(data, spec)

        # store it in the batch
        batch[self.out_array] = reduced

        # return the new batch
        return batch