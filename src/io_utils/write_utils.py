import dask.array as da
import functools
import logging
import nrrd
import numpy as np
import os
import tifffile

from . import zarr_utils


logger = logging.getLogger(__name__)


def save(container_path, dataset_subpath,
         data, shape,
         blocksize=None,
         container_attributes={},
         **dataset_attributes,
):
    """
    Persist distributed data - typically a dask array to the specified
    container

    Parameters
    ==========
    container_path
    dataset_subpath
    data - the dask array that needs
    shape - the shape of the saved data.
            This may differ from the data.shape because
            in we want to save the labels which typically are 3D as a 5D array
            to have the same number of dimensions as the input
    """
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]
    persist_block = None
    if container_ext == '.nrrd':
        logger.info(f'Persist data as nrrd {container_path} ({real_container_path})')
        output_dir = os.path.dirname(container_path)
        output_name = os.path.basename(path_comps[0])
        persist_block = functools.partial(_save_block_to_nrrd,
                                          output_dir=output_dir,
                                          output_name=output_name,
                                          ext=container_ext)
    elif container_ext == '.tif' or container_ext == '.tiff':
        logger.info(f'Persist data as tiff {container_path} ({real_container_path})')
        output_dir = os.path.dirname(container_path)
        output_name = os.path.basename(path_comps[0])
        persist_block = functools.partial(_save_block_to_tiff,
                                          output_dir=output_dir,
                                          output_name=output_name,
                                          ext=container_ext,
                                          container_attributes=container_attributes)
    elif (container_ext == '.n5' or container_ext == '') and dataset_subpath and dataset_subpath != '.':
        logger.info((
            f'Persist data as N5 {container_path} '
            f'({real_container_path}):{dataset_subpath} '
            f'with shape {shape} and blocksize {blocksize} '
        ))
        data_store = 'n5'
        zarr_data = zarr_utils.create_dataset(
            container_path,
            dataset_subpath,
            shape,
            blocksize,
            data.dtype,
            data_store_name='n5',
            parent_attrs=container_attributes,
            **dataset_attributes,
        )
        persist_block = functools.partial(_save_block_to_zarr,
                                          zarr_output=zarr_data)
    elif container_ext == '.zarr' or not dataset_subpath or dataset_subpath == '.':
        logger.info((
            f'Persist {data.dtype} data as zarr {container_path} '
            f'({real_container_path}):{dataset_subpath} '
            f'with shape {shape} and blocksize {blocksize} '
        ))
        data_store = 'zarr'
        zarr_data = zarr_utils.create_dataset(
            container_path,
            dataset_subpath,
            shape,
            blocksize,
            data.dtype,
            data_store_name=data_store,
            parent_attrs=container_attributes,
            **dataset_attributes,
        )
        persist_block = functools.partial(_save_block_to_zarr,
                                          zarr_output=zarr_data)
    else:
        logger.info((
            f'Cannot persist data using {container_path} '
            f'({real_container_path}): {subpath} '
        ))

    if persist_block is not None:
        return _save_blocks(data, persist_block)
    else:
        return None


def _save_blocks(dimage, persist_block):
    return da.map_blocks(persist_block,
                         dimage,
                         dtype=bool,
                         chunks=(),
                         drop_axis=tuple(range(dimage.ndim)))


def _save_block_to_nrrd(block, output_dir=None, output_name=None,
                        block_info=None,
                        ext='.nrrd'):
    if block_info is not None:
        output_coords = _block_coords_from_block_info(block_info, 0)
        block_coords = tuple([slice(s.start-s.start, s.stop-s.start)
                              for s in output_coords])

        saved_blocks_count = np.prod(block_info[None]['num-chunks'])
        if saved_blocks_count > 1:
            filename = (output_name + '-' +
                        '-'.join(map(str, block_info[0]['chunk-location'])) +
                        ext)
        else:
            filename = output_name + ext

        full_filename = os.path.join(output_dir, filename)
        logger.info((
            f'Write block {block.shape} '
            f'block_info: {block_info} '
            f'output_coords: {output_coords} '
            f'block_coords: {block_coords} '
        ))
        nrrd.write(full_filename, block[block_coords].transpose(2, 1, 0),
                   compression_level=2)
        return True
    else:
        return False


def _save_block_to_tiff(block, output_dir=None, output_name=None,
                        block_info=None,
                        ext='.tif',
                        container_attributes={}
                        ):
    res_shape = tuple([1 for r in range(0, block.ndim)])
    if block_info is not None:
        output_coords = _block_coords_from_block_info(block_info, 0)
        block_coords = tuple([slice(s.start-s.start, s.stop-s.start)
                              for s in output_coords])

        saved_blocks_count = np.prod(block_info[None]['num-chunks'])
        if saved_blocks_count > 1:
            filename = (output_name + '-' +
                        '-'.join(map(str, block_info[0]['chunk-location'])) +
                        ext)
        else:
            filename = output_name + ext

        full_filename = os.path.join(output_dir, filename)
        logger.info((
            f'Write block {block.shape} '
            f'block_info: {block_info} '
            f'output_coords: {output_coords} '
            f'block_coords: {block_coords} '
            f'to {full_filename} '
        ))
        tiff_metadata = {
            'axes': 'ZYX',
        }
        tiff_metadata.update(container_attributes)

        tifffile.imwrite(full_filename, block[block_coords],
                         metadata=tiff_metadata)
        return True
    else:
        return False


def _save_block_to_zarr(block, zarr_output=np.empty, block_info=None):
    if block_info is not None and zarr_output is not None:
        logger.info(f'Save {block.shape} block {block_info} to {zarr_output.shape}')
        output_coords = _block_coords_from_block_info(block_info, 0)
        block_coords = tuple([slice(s.start-s.start, s.stop-s.start)
                              for s in output_coords])
        if len(block_coords) < len(zarr_output.shape):
            missing_dims = len(zarr_output.shape) - len(block_coords)
            output_coords = (0,) * missing_dims + output_coords

        logger.info((
            f'Write block {block.shape} '
            f'output_coords: {output_coords} '
            f'block_coords: {block_coords} '
        ))
        try:
            zarr_output[output_coords] = block[block_coords]
        except Exception as e:
            logger.exception((
                f'Error while writing {block.shape} block {block_coords} '
                f'to {output_coords} '
            ))
            raise e

        return True #np.ones(res_shape, dtype=np.uint32)
    else:
        return False #np.zeros(res_shape, dtype=np.uint32)


def _block_coords_from_block_info(block_info, block_index):
    return tuple([slice(c[0],c[1])
                  for c in block_info[block_index]['array-location']])
