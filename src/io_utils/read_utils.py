import logging
import nrrd
import numpy as np
import os
import zarr

from tifffile import TiffFile

from . import zarr_utils


logger = logging.getLogger(__name__)


def get_voxel_spacing(attrs, default_value=[1., 1., 1.]):
    """
    Get voxel spacing always in the [TC]ZYX order.

    The method takes into consideration that
    * OME ZARR has the coords in TCZYX order
    * N5 has them in XYZ order
    """
    if attrs.get('coordinateTransformations'):
        # this typically is the OME-ZARR case
        scale_metadata = list(filter(lambda t: t.get('type') == 'scale', attrs['coordinateTransformations']))
        if len(scale_metadata) > 0:
            dataset_scale = scale_metadata[0].get('scale', [1., 1., 1., 1., 1.])
            # get voxel spatial resolution as [dx, dy, dz]
            xyz_voxel_spacing = np.array(dataset_scale[-3:])[::-1]
        else:
            xyz_voxel_spacing = np.array(default_value) if default_value is not None else None
    elif (attrs.get('downsamplingFactors') is not None and
          attrs.get('pixelResolution') is not None):
        # this is the N5 case for a scale != S0
        xyz_voxel_spacing = (np.array(attrs['pixelResolution']) *
                             np.array(attrs['downsamplingFactors']))
    elif (attrs.get('pixelResolution')):
        # this is the N5 case for S0
        pixel_resolution = attrs['pixelResolution']
        if type(pixel_resolution) is list:
            xyz_voxel_spacing = np.array(pixel_resolution)
        else:
            raise ValueError((
                f'Unknown pixelResolution: {pixel_resolution} of type {type(pixel_resolution)} '
                f'found in {attrs} '
            ))
    else:
        xyz_voxel_spacing = np.array(default_value) if default_value is not None else None

    if xyz_voxel_spacing is not None:
        # flip the coordinates as zyx
        return xyz_voxel_spacing[::-1]
    else:
        return None


def open(container_path, subpath, data_timeindex=None, data_channels=None, block_coords=None):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        logger.info(f'Open nrrd {container_path} ({real_container_path})')
        return read_nrrd(container_path, block_coords=block_coords)
    elif container_ext == '.tif' or container_ext == '.tiff':
        logger.info(f'Open tiff {container_path} ({real_container_path})')
        im = read_tiff(container_path, block_coords=block_coords)
        return im, {}
    elif container_ext == '.npy':
        im = np.load(container_path)
        return im, {}
    elif container_ext == '.n5' or container_ext == '.zarr':
        container_store = container_ext[1:]
        logger.info((
            f'Open {container_path} ({real_container_path}):{subpath} '
            f'using {container_store} container store '
        ))
        return zarr_utils.open(container_path, subpath,
                               block_coords=block_coords,
                               data_store_name=container_store,
                               data_timeindex=data_timeindex,
                               data_channels=data_channels)
    else:
        logger.warning(f'Cannot handle {container_path} ({real_container_path}): {subpath}')
        return None, {}


def read_tiff(input_path, block_coords=None):
    with TiffFile(input_path) as tif:
        tif_store = tif.aszarr()
        tif_array = zarr.open(tif_store)
        if block_coords is None:
            img = tif_array
        else:
            img = tif_array[block_coords]
        return img


def read_nrrd(input_path, block_coords=None):
    im, dict = nrrd.read(input_path)
    return im[block_coords] if block_coords is not None else im, dict
