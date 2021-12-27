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