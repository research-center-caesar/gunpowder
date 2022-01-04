from typing import List, Tuple
import logging

import numpy as np

from ..batch import Batch
from .batch_filter import BatchFilter
from ..array import ArrayKey
from ..batch_request import BatchRequest
from ..roi import Roi

logger = logging.getLogger(__name__)


class PadToRequestedSize(BatchFilter):
    '''Extent the downstream batches with a constant intensity value such 
    that it fits the requested shape.

    Args:

        array_keys (:class: `List[ArrayLey]`):

            The array keys to pad.
    '''

    def __init__(self, array_keys: List[ArrayKey], shape: Tuple[int]) -> None:
        self.array_keys = array_keys
        self.shape = shape

    def setup(self) -> None:
        self.enable_autoskip()


        for key in self.array_keys:
            spec = self.spec[key].copy()
            spec.roi.set_shape(self.shape)
            self.updates(key, spec)

        
    def prepare(self, request: BatchRequest) -> BatchRequest:
        deps = BatchRequest()

        spec = {k: None for k in self.array_keys}

        logger.debug(f'request: {request}')

        for key in self.array_keys:
            upstream_provider = self.get_upstream_provider()
            spec[key] = upstream_provider.spec[key]

            while spec[key].roi.unbounded():
                upstream_provider = upstream_provider.get_upstream_provider()
                spec[key] = upstream_provider.spec[key]


            request[key].roi = Roi(
                request[key].roi.get_offset(),
                tuple(min([r, s]) for r, s in zip(request[key].roi.get_shape(), spec[key].roi.get_shape()))
            )
            deps[key] = request[key].copy()

        logger.debug(f'change request to: {request}')


        return deps

    def process(self, batch: Batch, request: BatchRequest) -> Batch:

        for key in self.array_keys:
            data = batch.arrays[key].data
            logger.debug(f'Requested shape: {self.shape}')
            logger.debug(f'Got shape: {data.shape}')

            assert len(data.shape) >= len(self.shape)

            channel_dims = len(data.shape) - len(self.shape)

            diff_shape = tuple(req_s - data_s
                for req_s, data_s in zip(self.shape, data.shape[channel_dims:]))

            logger.debug(f'channel dimensions = {channel_dims}')

            logger.debug(f'Difference: {diff_shape}')
            pad_before = tuple(np.ceil(d/2).astype(int) for d in diff_shape)
            pad_after = tuple(np.floor(d/2).astype(int) for d in diff_shape)
            pad = tuple((d_b, d_a) for d_b, d_a in zip(pad_before, pad_after))
            pad = (((0,0),) * channel_dims) + pad
            logger.debug(f'Pad with: {pad}')
            data = np.pad(data, pad)

            batch.arrays[key].data = data
            batch.arrays[key].spec.roi = request[key].roi

        return batch