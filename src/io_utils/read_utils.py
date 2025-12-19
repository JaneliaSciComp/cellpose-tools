import logging
import os
import zarr

from tifffile import TiffFile
from zarr_tools.io.zarr_io import open_zarr_store


logger = logging.getLogger(__name__)


def open_array(array_container_path:str, array_subpath:str):
    path_comps = os.path.splitext(array_container_path)
    array_type = path_comps[1].strip('.')

    if array_type == 'tif' or array_type == 'tiff':
        return _open_tiff_array(array_container_path)
    elif array_type == 'n5' or array_type == 'zarr':
        return zarr.open_array(store=array_container_path, path=array_subpath)
    else:
        raise ValueError(f'Cannot handle {array_container_path}:{array_subpath}')


def read_array_attrs(container_path:str, subpath:str) -> dict:
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    array_type = path_comps[1].strip('.')

    if array_type == 'tif' or array_type == 'tiff':
        logger.info(f'Open tiff {container_path} ({real_container_path})')
        return _read_tiff_attrs(container_path)
    elif array_type == 'n5' or array_type == 'zarr':
        logger.info(f'Open {container_path} ({real_container_path}):{subpath}')
        _, zattrs = open_zarr_store(container_path, subpath)
        return zattrs
    else:
        logger.warning(f'Cannot handle {container_path} ({real_container_path}): {subpath}')
        return {}


def _open_tiff_array(input_path):
    tif = TiffFile(input_path)
    tif_store = tif.aszarr()
    return zarr.open(tif_store)


def _read_tiff_attrs(input_path):
    with TiffFile(input_path) as tif:
        tif_store = tif.aszarr()
        tif_array = zarr.open(tif_store)
        return {
            'array_axes': tif.series[0].axes,
            'array_storepath': input_path,
            'array_subpath': '',
            'array_shape': tif_array.shape,
            'array_ndim': tif_array.ndim,
            'array_dtype': tif_array.dtype.name,
            'array_chunksize': tif_array.chunks,
        }
