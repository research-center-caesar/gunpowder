import copy
import logging

import numpy as np

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

        return self.get_upstream_providers()[choice].request_batch(request)