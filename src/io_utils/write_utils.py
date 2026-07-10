import logging
import os
import tifffile


logger = logging.getLogger(__name__)


def container_type(container_path):
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]
    match container_ext:
        case '.zarr':
            return 'zarr'
        case '.zarr2':
            return 'zarr2'
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
        _write_zarr_to_tiff(zarr_arr, container_path)
    else:
        logger.info((
            f'Cannot persist data using {container_path} '
            f'({real_container_path}): {dataset_subpath} '
        ))


def _write_zarr_to_tiff(zarr_array, output_path):
    array_shape = zarr_array.shape
    match len(array_shape):
        case 2:
            _write_zarr_to_2D_tiff(zarr_array, output_path)
        case 3:
            _write_zarr_to_3D_tiff(zarr_array, output_path)
        case 4:
            _write_zarr_to_4D_tiff(zarr_array, output_path)
        case 5:
            _write_zarr_to_5D_tiff(zarr_array, output_path)
        case _:
            raise ValueError(f'Cannot save as tiff {array_shape} arrays. It only up to 5D zarr arrays')


def _write_zarr_to_2D_tiff(zarr_array, output_path):
    with tifffile.TiffWriter(output_path) as tw:
        tw.write(zarr_array[...])


def _write_zarr_to_3D_tiff(zarr_array, output_path):
    z, _, _ = zarr_array.shape
    with tifffile.TiffWriter(output_path, bigtiff=True) as tw:
        for zi in range(z):
            tw.write(zarr_array[zi])


def _write_zarr_to_4D_tiff(zarr_array, output_path):
    c, z, _, _ = zarr_array.shape
    metadata = {
        "axes": "CZYX",
    }
    with tifffile.TiffWriter(output_path, bigtiff=True) as tw:
        for ci in range(c):
            for zi in range(z):
                tw.write(
                    zarr_array[ci, zi],  # (y, x) plane
                    metadata=metadata if (ci,zi)==(0,0,0) else None
                )


def _write_zarr_to_5D_tiff(zarr_array, output_path):
    t, c, z, _, _ = zarr_array.shape
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
