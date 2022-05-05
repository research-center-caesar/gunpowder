import gunpowder as gp
from gunpowder.array_spec import ArraySpec
from gunpowder.nodes.random_location import RandomLocation
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

def test_PadToRequestedSize__real(tmp_path):

    voxel_size = gp.Coordinate((1024, 1024, 35))

    with zarr.open(tmp_path / 'sample.zarr', 'w') as f:
        f['raw'] = np.zeros((1, 512, 100, 10))
        f['raw'].attrs['resolution'] = tuple(voxel_size)

    raw = gp.ArrayKey('RAW')

    source = gp.ZarrSource(
        str(tmp_path / 'sample.zarr'),
        {raw: 'raw'},
        {raw: ArraySpec(
            interpolatable=True,
        )},
    )

    pipeline = source + RandomLocation() + PadToRequestedSize([raw])
    request = gp.BatchRequest()
    request[raw] = gp.ArraySpec(
        roi=gp.Roi((0, 0, 0), (256, 256, 1)) * voxel_size,
        voxel_size=voxel_size,)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert batch.arrays[raw].data.shape == (1, 256, 256, 1)



