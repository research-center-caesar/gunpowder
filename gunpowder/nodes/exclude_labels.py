import logging
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from .batch_filter import BatchFilter
from gunpowder.volume import Volume, VolumeType

logger = logging.getLogger(__name__)

class ExcludeLabels(BatchFilter):

    def __init__(self, labels, include_context, background_value=0):
        self.labels = set(labels)
        self.include_context = include_context
        self.background_value = background_value

    def process(self, batch):

        gt = batch.volumes[VolumeType.GT_LABELS]

        # 0 marks included regions (to be used directly with distance transform 
        # later)
        include_mask = np.ones(gt.data.shape)

        for label in np.unique(gt.data):
            if label in self.labels:
                gt.data[gt.data==label] = self.background_value
            else:
                include_mask[gt.data==label] = 0

        distance_to_include = distance_transform_edt(include_mask, sampling=batch.spec.resolution)
        logger.debug("ExcludeLabels: max distance to foreground is " + str(distance_to_include.max()))

        # 1 marks included regions, plus a context area around them
        include_mask = distance_to_include<self.include_context

        if VolumeType.GT_MASK in batch.volumes:
            batch.volumes[VolumeType.GT_MASK].data &= include_mask
        else:
            batch.volumes[VolumeType.GT_MASK] = Volume(include_mask, interpolate=False)
