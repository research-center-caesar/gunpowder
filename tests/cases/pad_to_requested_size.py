import gunpowder as gp
from gunpowder.array_spec import ArraySpec
import numpy as np
import zarr

from gunpowder import PadToRequestedSize



def test_import():
    pass

def test_PadToRequestedSize(tmp_path):
    with zarr.open(tmp_path / 'sample.zarr', 'w') as f:
        f['raw'] = np.zeros((10, 8, 10), dtype=int)
        f['raw'].attrs['resolution'] = (1, 1, 1)

    raw = gp.ArrayKey('RAW')

    source = gp.ZarrSource(
        str(tmp_path / 'sample.zarr'),
        {raw: 'raw'},
        {raw: ArraySpec(
            interpolatable=True,
        )}
    )

    pipeline = source + PadToRequestedSize([raw])

    request = gp.BatchRequest()
    request[raw] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), (10, 10, 10)))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert batch.arrays[raw].data.shape == (10, 10, 10)



