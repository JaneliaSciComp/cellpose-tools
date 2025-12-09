import dask.array as da
import functools
import logging
import nrrd
import numpy as np
import os
import tifffile


logger = logging.getLogger(__name__)


def container_type(container_path):
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]
    match container_ext:
        case '.zarr':
            return 'zarr'
        case '.n5':
            return 'n5'
        case '.tif' | '.tiff':
            return 'tiff'
        case _:
            raise ValueError(f'Unsupported container type for {container_path}. It only supports .zarr|.tif')


def write_zarray_as(zarr_arr, container_path, dataset_subpath):
    """
    Persist zarr array at the specified container_path

    Parameters
    ==========
    zarr_arr - the zarr array that needs to saved
    container_path
    dataset_subpath
    """
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]
    if container_ext == '.tif' or container_ext == '.tiff':
        logger.info(f'Persist data as tiff {container_path} ({real_container_path})')
        _write_as_tiff(zarr_arr, container_path)
    else:
        logger.info((
            f'Cannot persist data using {container_path} '
            f'({real_container_path}): {dataset_subpath} '
        ))


def _write_as_tiff(zarr_array, output_path):
    _, _, z, _, _ = zarr_array.shape
    metadata = {
        "axes": "TCZYX",
    }
    with tifffile.TiffWriter(output_path, bigtiff=True) as tw:
        for ti in range(t):
            for ci in range(c):
                for zi in range(z):
                    tw.write(
                        zarr_array[ti, ci, zi],  # (y, x) plane
                        metadata=metadata if (ti,ci,zi)==(0,0,0) else None
                    )
