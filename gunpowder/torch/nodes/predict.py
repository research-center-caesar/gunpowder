from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.ext import torch
from gunpowder.nodes.generic_predict import GenericPredict

from functools import reduce
from operator import mul
import ctypes
import logging
import multiprocessing as mp
import numpy as np
from typing import Dict, Union

logger = logging.getLogger(__name__)


class Predict(GenericPredict):
    """Torch implementation of :class:`gunpowder.nodes.Predict`.

    Args:

        model (subclass of ``torch.nn.Module``):

            The model to use for prediction.

        inputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of input tensors (argument names of the
            ``forward`` method) in the model to array keys.

        outputs (``dict``, ``string`` or ``int`` -> :class:`ArrayKey`):

            Dictionary from the names of tensors in the network to array
            keys. If the key is a string, the tensor will be retrieved
            by checking the module for an attribute with they key as its name.
            If the key is an integer, it is interpreted as a tuple index of
            the outputs of the network.
            New arrays will be generated by this node for each entry (if
            requested downstream).

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            Used to set the specs of generated arrays (``outputs``). This is
            useful to set the ``voxel_size``, for example, if they differ from
            the voxel size of the input arrays. Only fields that are not
            ``None`` in the given :class:`ArraySpec` will be used.

        checkpoint: (``string``, optional):

            An optional path to the saved parameters for your torch module.
            These will be loaded and used for prediction if provided.
        
        use_gpus: (``list``, ``int``):

            Which gpu's to use for prediction.
            Not yet implemented.
    """

    def __init__(
        self,
        model,
        inputs: Dict[str, ArrayKey],
        outputs: Dict[Union[str, int], ArrayKey],
        array_specs: Dict[ArrayKey, ArraySpec] = {},
        checkpoint=None,
        gpus=[0],
    ):

        super(Predict, self).__init__(inputs, outputs, array_specs)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = model.to(self.device)
        self.checkpoint = checkpoint
        self.gpus = gpus

        if self.checkpoint is not None:
            self.model.load_state_dict(torch.load(self.checkpoint))
        self.intermediate_layers = {}
        self.register_hooks()

    def start(self):
        pass

    def predict(self, batch, request):
        inputs = self.get_inputs(batch)
        out = self.model.forward(**inputs)
        outputs = self.get_outputs(out)
        self.update_batch(batch, request, outputs)

    def get_inputs(self, batch):
        model_inputs = {
            key: torch.as_tensor(batch[value].data, device=self.device)
            for key, value in self.inputs.items()
        }
        return model_inputs

    def register_hooks(self):
        for key in self.outputs:
            if isinstance(key, str):
                layer = getattr(self.model, key)
                layer.register_forward_hook(self.create_hook(key))

    def create_hook(self, key):
        def save_layer(module, input, output):
            self.intermediate_layers[key] = output
        return save_layer

    def get_outputs(self, module_out):
        outputs = {}
        if isinstance(module_out, tuple):
            module_outs = module_out
        else:
            module_outs = (module_out,)
        for key, value in self.outputs.items():
            if isinstance(key, str):
                outputs[value] = \
                    self.intermediate_layers[key].cpu().detach().numpy()
            elif isinstance(key, int):
                outputs[value] = module_outs[key].cpu().detach().numpy()
        return outputs

    def update_batch(self, batch, request, outputs):
        for key, data in outputs.items():
            spec = self.spec[key].copy()
            spec.roi = request[key].roi
            batch[key] = Array(data, spec)

    def stop(self):
        pass
