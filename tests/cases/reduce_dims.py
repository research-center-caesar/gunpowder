import numpy as np
from pathlib import Path

import gunpowder as gp
from .wkw_source import _create_dataset
from gunpowder import build, ArrayKey, ArraySpec, WKWSource, BatchRequest, Roi

def test_WKWSource__2d(tmp_path):


    wkw_file = _create_dataset(
        tmp_path, 'raw', np.zeros((100, 100, 100), dtype=np.uint8))

    raw_org = ArrayKey('RAW')
    raw = ArrayKey('RAW_2D')

    source = WKWSource(
        wkw_file,
        {
            raw_org: 'raw',
        },
        mag_specs={
            raw_org: 1,
        }
    )

    pipeline = source + gp.RandomLocation() + gp.Squeeze([raw_org]) + gp.ReduceDim(raw_org, raw, np.squeeze, axis=2)
    with build(pipeline):
        batch = pipeline.request_batch(
            BatchRequest({
                raw: ArraySpec(roi=Roi((0, 0), (100, 100))),
            })
        )
