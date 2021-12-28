import copy
import logging
import math
from gunpowder.array_spec import ArraySpec
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate

import numpy as np
from skimage.measure import block_reduce

from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)

class RandomOrder(BatchProvider):
    '''Selects upstream provider in random order::

        (a, b, c) + RandomOrder()

    will return a provider from ``a``, ``b``, or ``c`` randomly. 
    When called again, it won't return the already returned one until
    all provider have been selected once.
    '''

    def __init__(self):
        self.not_returend = None

    
    def setup(self):
        self.enable_placeholders()
        assert(len(self.get_upstream_providers())) > 0, \
            "at leat one batch provider must be added to the RandomOrder node"

        self.not_returend = np.ones(len(self.get_upstream_providers()), dtype=bool)

        common_spec = None

        for provider in self.get_upstream_providers():
            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for key, spec in list(common_spec.items()):
                    if key not in provider.spec:
                        del common_spec[key]

        # find for each key the least common multiply of all provided voxel sizes
        for key, spec in common_spec.items():
            lcm_voxel_size = None
            for provider in self.get_upstream_providers():
                voxel_size = provider.spec[key].voxel_size
                if voxel_size is None:
                    continue
                if lcm_voxel_size is None:
                    lcm_voxel_size = voxel_size
                else:
                    lcm_voxel_size = Coordinate(
                        (a * b // math.gcd(a, b)
                        for a, b in zip(lcm_voxel_size, voxel_size)))
            if not common_spec[key].voxel_size == lcm_voxel_size:
                logger.warning(f'Change provided {key} voxel size from ' +
                     f'{common_spec[key].voxel_size} to '+
                     f'{lcm_voxel_size}')

                common_spec[key].voxel_size = lcm_voxel_size

        for key, spec in common_spec.items():
            self.provides(key, spec)

    def provide(self, request):
        # Random seed is set in provide rather than prepare since this node
        # is not a batch filter
        np.random.seed(request.random_seed)

        if not np.any(self.not_returend):
            logger.debug('Reset RandomOrder provider')
            self.not_returend[:] = True

        available_choices = np.nonzero(self.not_returend)[0]

        logger.debug(f'Available choices: {available_choices} from \
            {len(self.get_upstream_providers())} items.' )

        choice = np.random.choice(available_choices)

        logger.debug(f'Select item {choice}')
        self.not_returend[choice] = False

        ## Change request voxel size from least common multiple voxel size
        # to available voxel size and post process the requested batch 
        upstream_spec = self.get_upstream_providers()[choice].spec

        new_request = BatchRequest()
        factors = {}

        for key in request.array_specs.keys():
            factors[key] = self.spec[key].voxel_size / upstream_spec[key].voxel_size
            new_request.array_specs[key] = ArraySpec(
                roi = request.array_specs[key].roi,
                voxel_size = upstream_spec[key].voxel_size,
            )

        batch = self.get_upstream_providers()[choice].request_batch(new_request)

        for key in batch.arrays.keys():
            factors_ = factors[key]
            data = batch.arrays[key].data
            if data.ndim > factors_.dims():
                factors_ = (1,) * (data.ndim - factors_.dims()) + \
                    factors_
            data = block_reduce(data, factors_, np.mean, func_kwargs={'dtype': data.dtype})

            batch.arrays[key].data = data
            batch.arrays[key].spec.voxel_size = self.spec[key].voxel_size

        return batch
