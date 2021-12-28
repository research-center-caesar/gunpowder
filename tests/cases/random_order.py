import zarr
import numpy as np

import gunpowder as gp


def test_random_order(tmp_path):
    zarr_filename = tmp_path / 'sample_array.zarr'
    raw = gp.ArrayKey('RAW')

    f = zarr.open(zarr_filename, 'w')
    f['raw'] = np.zeros((100, 100, 100), dtype=np.uint8)
    f['raw'].attrs['resolution'] = (1, 1, 1)

    bboxes = [
        gp.Roi((50, 50, 50), (50, 50, 50)),
        gp.Roi((0, 0, 0), (50, 50, 50))
    ]

    sources = tuple(
        gp.ZarrSource(
            str(zarr_filename),
            {raw: 'raw'},
            {raw: gp.ArraySpec(
                roi = bbox,
                interpolatable=True)}
        ) + gp.RandomLocation() for bbox in bboxes
    )

    pipeline = sources + gp.RandomOrder()


    request = gp.BatchRequest()
    request[raw] = gp.Roi((0,0,0),(32, 32, 32))

    with gp.build(pipeline):
        for _ in range(3):
            batch = pipeline.request_batch(request) 

    return

def test_random_order__diff_voxel_size(tmp_path):
    raw = gp.ArrayKey('RAW')

    for i, voxel_size in enumerate([(1,1,1), (2,2,2)]):
        zarr_filename = tmp_path / f'sample_array_{i}.zarr'


        f = zarr.open(zarr_filename, 'w')
        f['raw'] = np.zeros((100, 100, 100), dtype=np.uint8)
        f['raw'].attrs['resolution'] = voxel_size

    sources = tuple(
        gp.ZarrSource(
            str(tmp_path / f'sample_array_{i}.zarr'),
            {raw: 'raw'},
            {raw: gp.ArraySpec(
                interpolatable=True)}
        ) + gp.RandomLocation() for i in range(2)
        )

    pipeline = sources + gp.RandomOrder()


    request = gp.BatchRequest()
    request[raw] = gp.Roi((0,0,0),(32, 32, 32))

    with gp.build(pipeline):
        for _ in range(3):
            batch = pipeline.request_batch(request) 

    return