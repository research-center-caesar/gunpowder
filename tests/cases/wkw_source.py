from logging import setLogRecordFactory
from pathlib import Path
from typing import Tuple

from gunpowder.nodes import squeeze
from gunpowder.nodes.zarr_source import ZarrSource
from gunpowder.batch import Batch
import numpy as np
from gunpowder import WKWSource, roi
from gunpowder.array import Array, ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.batch_request import BatchRequest
from gunpowder.build import build
from gunpowder import Roi
import gunpowder as gp
from webknossos import Dataset


def _create_dataset(
    tmp_path: Path, key: str, data: np.ndarray,
    magnification: Tuple[int] =(1, 1, 1), category: str = 'color',
    scale: Tuple[int] = (1,1,1), **kwargs
) -> str:
    data_file = tmp_path / 'test_wkw'

    Dataset.get_or_create(data_file, scale=scale)\
        .get_or_add_layer(key, category, dtype_per_layer=data.dtype, **kwargs)\
        .get_or_add_mag(magnification)\
        .write(data)

    return str(data_file)

def test_WKKSource__data_integrity(tmp_path):

    data = np.arange(27,dtype=np.uint8).reshape(3,3,3)

    wkw_file = _create_dataset(
        tmp_path, 'seg', data, scale=(1,2,3),
        category='segmentation', magnification=(1,1,1), largest_segment_id=int(data.max())
    )

    seg = ArrayKey('SEG')

    source = WKWSource(
        wkw_file,
        {seg: 'seg'},
        mag_specs = {seg: 1}
    )

    request = BatchRequest()
    request[seg] = Roi((0, 0, 0), (3,6,9))

    with build(source):
        batch = source.request_batch(request)

    assert np.all(batch[seg].data == data)

def test_WKWSource__data_integrity__offset(tmp_path):
    data = np.arange(5*5*5, dtype=np.uint8).reshape(5,5,5)

    wkw_file = _create_dataset(
        tmp_path, 'seg', data,
        category='segmentation', magnification=(1,1,1),
        largest_segment_id=int(data.max()), scale=(1,2,3))

    seg = ArrayKey('SEG')

    source = WKWSource(
        wkw_file, 
        {seg: 'seg'},
        mag_specs = {seg: 1} 
    )

    request = BatchRequest()
    request[seg] = Roi((3, 2, 0), (2, 4, 9))

    with build(source):
        batch = source.request_batch(request)

    assert np.all(batch[seg].data == data[3:, 1:3, :3])

def test_WKWSource__3d(tmp_path):

    wkw_file = _create_dataset(
        tmp_path, 'raw', np.zeros((100, 100, 100), dtype=np.float32))
    wkw_file = _create_dataset(
        tmp_path, 'raw_low', np.zeros((16, 16, 16), dtype=np.float32),
        magnification=(16, 16, 16))
    wkw_file = _create_dataset(
        tmp_path, 'seg', np.ones((100, 100, 100), dtype=np.uint64),
        category='segmentation', largest_segment_id=1)

    raw = ArrayKey('RAW')
    raw_low = ArrayKey('RAW_LOW')
    seg = ArrayKey('SEG')

    source = WKWSource(
        wkw_file,
        {
            raw: 'raw',
            raw_low: 'raw_low',
            seg: 'seg'
        },
        mag_specs={
            raw: 1,
            raw_low: [16, 16, 16],
            seg: [1, 1, 1],
        }
    )

    with build(source):
        batch = source.request_batch(
            BatchRequest({
                raw: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
                raw_low: ArraySpec(roi=Roi((0, 0, 0), (128, 128, 128))),
                seg: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
            })
        )

        assert batch.arrays[raw].spec.interpolatable
        assert batch.arrays[raw_low].spec.interpolatable
        assert not batch.arrays[seg].spec.interpolatable

