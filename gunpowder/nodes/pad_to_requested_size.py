from typing import List
import logging

import numpy as np

from ..batch import Batch
from .batch_filter import BatchFilter
from ..array import ArrayKey
from ..batch_request import BatchRequest

logger = logging.getLogger(__name__)


class PadToRequestedSize(BatchFilter):
    '''Extent the downstream batches with a constant intensity value such 
    that it fits the requested shape.

    Args:

        array_keys (:class: `List[ArrayLey]`):

            The array keys to pad.
    '''

    def __init__(self, array_keys: List[ArrayKey]) -> None:
        self.array_keys = array_keys

    def setup(self) -> None:
        self.enable_autoskip()

        for key in self.array_keys:
            spec = self.spec[key].copy()
            spec.roi.set_shape(None)
            self.updates(key, spec)

    def prepare(self, request: BatchRequest) -> BatchRequest:
        deps = BatchRequest()

        upstream_spec = self.get_upstream_provider().spec

        for key in self.array_keys:
            request[key].roi = upstream_spec[key].roi
            deps[key] = request[key].copy()


        return deps

    def process(self, batch: Batch, request: BatchRequest) -> Batch:

        for key in self.array_keys:
            data = batch.arrays[key].data
            requested_shape = request[key].roi.get_shape()
            logger.debug(f'Requested shape: {requested_shape}')
            logger.debug(f'Got shape: {data.shape}')
            diff_shape = tuple(req_s - data_s
                for req_s, data_s in zip(requested_shape, data.shape))
            logger.debug(f'Difference: {diff_shape}')
            pad_before = tuple(np.ceil(d/2).astype(int) for d in diff_shape)
            pad_after = tuple(np.floor(d/2).astype(int) for d in diff_shape)
            pad = tuple((d_b, d_a) for d_b, d_a in zip(pad_before, pad_after))
            logger.debug(f'Pad with: {pad}')
            data = np.pad(data, pad)

            batch.arrays[key].data = data
            batch.arrays[key].spec.roi = request[key].roi

        return batch