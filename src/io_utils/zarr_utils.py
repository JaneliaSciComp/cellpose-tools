import logging
import numcodecs as codecs
import os
import re
import zarr

from ome_zarr_models.v04.image import (Dataset)


logger = logging.getLogger(__name__)


def create_dataset(container_path, dataset_subpath, shape, chunks, dtype,
                   data=None, data_store_name=None,
                   compressor=None,
                   parent_attrs={},
                   **dataset_attrs):
    try:
        real_container_path = os.path.realpath(container_path)
        store = _get_data_store(real_container_path, data_store_name)
        if dataset_subpath and dataset_subpath != '.':
            logger.info((
                f'Create dataset {container_path}:{dataset_subpath} '
                f'compressor={compressor}, shape: {shape}, chunks: {chunks} '
                f'parent attrs: {parent_attrs} '
                f'{dataset_attrs} '
            ))
            root_group = zarr.open_group(store=store, mode='a')
            codec = (None if compressor is None
                     else codecs.get_codec(dict(id=compressor)))
            if dataset_subpath in root_group:
                # if the dataset already exists, get its shape
                dataset_shape = root_group[dataset_subpath].shape
                logger.info((
                    f'Dataset {container_path}:{dataset_subpath} '
                    f'already exists with shape {dataset_shape} '
                ))
            else:
                dataset_shape = shape

            dataset = root_group.require_dataset(
                dataset_subpath,
                shape=dataset_shape,
                chunks=chunks,
                dtype=dtype,
                compressor=codec,
                data=data)

            _update_dataset_attrs(root_group, dataset,
                                  parent_attrs=parent_attrs,
                                  **dataset_attrs)
            return dataset
        else:
            logger.info(f'Create root array {container_path} {dataset_attrs}')
            zarr_data = zarr.open(store=store, mode='a',
                                  shape=shape, chunks=chunks)

            _update_dataset_attrs(zarr_data, zarr_data,
                                  parent_attrs=parent_attrs,
                                  **dataset_attrs)

            return zarr_data

    except Exception as e:
        logger.error(f'Error creating a dataset at {container_path}:{dataset_subpath}: {e}')
        raise e


def _update_dataset_attrs(root_container, dataset,
                          parent_attrs={}, **dataset_attrs):
    if dataset.path:
        dataset_parent = os.path.dirname(dataset.path)
        parent_container = (root_container if not dataset_parent
                            else root_container.require_group(dataset_parent))
    else:
        parent_container = root_container

    parent_container.attrs.update(parent_attrs)
    dataset.attrs.update(dataset_attrs)


def open(data_path, data_subpath, data_store_name=None,
         data_timeindex=None, data_channels=None, 
         mode='r',
         block_coords=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath, data_store_name)
        data_store = _get_data_store(zarr_container_path, data_store_name)
        logger.debug(f'Open zarr container: {data_store}:{data_subpath} ({data_store_name})')
        data_container = zarr.open(store=data_store, mode=mode)
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
            logger.debug((
                f'Open OME ZARR {zarr_container_path}:{zarr_subpath} '
                f'timeindex: {data_timeindex} '
                f'channels: {data_channels} '
            ))
            return _open_ome_zarr(data_container, zarr_subpath,
                                  data_timeindex=data_timeindex,
                                  data_channels=data_channels,
                                  block_coords=block_coords)
        else:
            a = (data_container[zarr_subpath] 
                if zarr_subpath and zarr_subpath != '.'
                else data_container)
            if block_coords is not None:
                logger.info(f'Get block at {block_coords}')
                ba = a[block_coords]
            else:
                ba = a
            return ba, a.attrs.asdict()
    except Exception as e:
        logger.error(f'Error opening {data_path}:{data_subpath} {e}')
        raise e


def prepare_parent_group_attrs(container_path,
                               dataset_path,
                               axes=None,
                               coordinateTransformations=None,
                               image_label_attrs=None,
                               ome_ngff_version='0.4'):
    if ((coordinateTransformations is None or coordinateTransformations == []) and
        axes is None):
        return {}

    if dataset_path:
        dataset_path_comps = [c for c in dataset_path.split('/') if c]
        # take the last component of the dataset path to be the scale path
        dataset_scale_subpath = dataset_path_comps.pop()
    else:
        # No subpath was provided - I am using '.', but
        # this may be problematic - I don't know yet how to handle it properly
        dataset_scale_subpath = '.'

    scales, translations = None, None
    if coordinateTransformations is not None:
        for t in coordinateTransformations:
            if t['type'] == 'scale':
                scales = t['scale']
            elif t['type'] == 'translation':
                translations = t['translation']

    multiscale_attrs = {
        'name': os.path.basename(container_path),
        'axes': axes if axes is not None else [],
        'version': ome_ngff_version,
    }

    if scales is not None:
        dataset = Dataset.build(path=dataset_scale_subpath, scale=scales, translation=translations)
        multiscale_attrs.update({
            'datasets': (dataset.dict(exclude_none=True),),
        })

    return {
        'image-label': image_label_attrs,
        'multiscales': [ multiscale_attrs ],
    }


def _adjust_data_paths(data_path, data_subpath, data_store_name):
    """
    This methods adjusts the container and dataset paths such that
    the container paths always contains a .attrs file
    """
    if data_store_name == 'n5' or data_path.endswith('.n5') or data_path.endswith('.N5'):
        # N5 container path is the same as the data_path
        # and the subpath is the dataset path
        return data_path, data_subpath

    dataset_path_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_path_arg.split('/') if c]
    dataset_comps_index = 0

    # Look for the first subpath that containes .zattrs file
    while dataset_comps_index < len(dataset_comps):
        container_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        container_path = f'{data_path}/{container_subpath}'
        if (os.path.exists(f'{container_path}/.zattrs') or
            os.path.exists(f'{container_path}/attributes.json')):
            break
        dataset_comps_index = dataset_comps_index + 1

    appended_container_path = '/'.join(dataset_comps[0:dataset_comps_index])
    container_path = f'{data_path}/{appended_container_path}'
    new_subpath = '/'.join(dataset_comps[dataset_comps_index:])

    return container_path, new_subpath


def _extract_numeric_comp(v):
    match = re.match(r'^(\D*)(\d+)$', v)
    if match:
        return int(match.groups()[1])
    else:
        raise ValueError(f'Invalid component: {v}')


def _get_data_store(data_path, data_store_name):
    if data_store_name is None or data_store_name == 'n5':
        return zarr.N5Store(data_path)
    else:
        return zarr.DirectoryStore(data_path, dimension_separator='/')


def _is_ome_zarr(data_container_attrs: dict | None) -> bool:
    if data_container_attrs is None:
        return False

    # test if bioformats_layout or multiscales attribute exists - if it does assume OME-ZARR
    bioformats_layout = data_container_attrs.get("bioformats2raw.layout", None)
    multiscales = data_container_attrs.get('multiscales', [])
    return bioformats_layout == 3 or len(multiscales) > 0


def _find_ome_multiscales(data_container, data_subpath):
    logger.info(f'Find OME multiscales group within {data_subpath}')
    dataset_subpath_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_subpath_arg.split('/') if c]

    dataset_comps_index = 0
    while dataset_comps_index < len(dataset_comps):
        group_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        dataset_item = data_container[group_subpath]
        dataset_item_attrs = dataset_item.attrs.asdict()
        if dataset_item_attrs.get('multiscales', []) == []:
            dataset_comps_index = dataset_comps_index + 1
        else:
            logger.debug(f'Found multiscales at {group_subpath}: {dataset_item_attrs}')
            # found a group that has attributes which contain multiscales list
            return dataset_item, '/'.join(dataset_comps[dataset_comps_index:]), dataset_item_attrs

    data_container_attrs = data_container.attrs.asdict()
    if data_container_attrs.get('multiscales', []) == []:
        return None, None, {}
    else:
        # the container itself has multiscales attributes
        return data_container, '', data_container_attrs


def _open_ome_zarr(data_container, data_subpath,
                   data_timeindex=None, data_channels=None, block_coords=None):
    multiscales_group, dataset_subpath, multiscales_attrs  = _find_ome_multiscales(data_container, data_subpath)

    if multiscales_group is None:
        a = (data_container[data_subpath]
             if data_subpath and data_subpath != '.'
             else data_container)
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()

    logger.debug((
        f'Open dataset {dataset_subpath}, timeindex: {data_timeindex}, '
        f'channels: {data_channels}, block_coords {block_coords} '
    ))

    dataset_comps = [c for c in dataset_subpath.split('/') if c]
    multiscale_metadata = multiscales_attrs.get('multiscales', [])[0]
    dataset_metadata = None
    # lookup the dataset by path
    for ds in multiscale_metadata.get('datasets', []):
        ds_path = ds.get('path', '')
        current_ds_path_comps = [c for c in ds_path.split('/') if c]
        logger.debug((
            f'Compare current dataset path: {ds_path} ({current_ds_path_comps}) '
            f'with {dataset_subpath} ({dataset_comps}) '
        ))
        if (len(current_ds_path_comps) <= len(dataset_comps) and
            tuple(current_ds_path_comps) == tuple(dataset_comps[-len(current_ds_path_comps):])):
            # found a dataset that has a path matching a suffix of the dataset_subpath arg
            dataset_metadata = ds
            currrent_dataset_path = ds.get('path')
            # drop the matching suffix
            dataset_comps = dataset_comps[-len(current_ds_path_comps):]
            logger.debug((
                f'Found dataset: {currrent_dataset_path}, '
                f'remaining dataset components: {dataset_comps}'
            ))
            break

    if dataset_metadata is None:
        # could not find a dataset using the subpath 
        # look at the last subpath component and get the dataset index from there
        # e.g., if the subpath looks like:
        #       '/s<n>' => datasets[n] if n < len(datasets) otherwise datasets[0]
        dataset_index_comp = dataset_comps[-1]
        logger.info(f'No dataset was found using {dataset_subpath} - try to use: {dataset_index_comp}')
        dataset_index = _extract_numeric_comp(dataset_index_comp)
        if dataset_index < len(multiscale_metadata.get('datasets', [])):
            dataset_metadata = multiscale_metadata['datasets'][dataset_index]
        else:
            dataset_metadata = multiscale_metadata['datasets'][0]

    dataset_axes = multiscale_metadata.get('axes')
    dataset_path = dataset_metadata.get('path')
    dataset_transformations = dataset_metadata.get('coordinateTransformations')
    logger.debug(f'Get array using array path: {dataset_path}:{data_timeindex}:{data_channels}')
    a = multiscales_group[dataset_path]
    # a is potentially a 5-dim array: [timepoint?, channel?, z, y, x]
    if block_coords is not None:
        ba = _get_array_selection(a, dataset_axes, data_timeindex, data_channels, block_coords)
    else:
        ba = _get_array_selection(a, dataset_axes, data_timeindex, data_channels, None)
    multiscales_attrs.update(a.attrs.asdict())
    multiscales_attrs.update({
        'dataset_path': dataset_path,
        'axes': dataset_axes,
        'dimensions': a.shape,
        'dataType': a.dtype,
        'blockSize': a.chunks,
        'timeindex': data_timeindex,
        'channels': data_channels,
        'coordinateTransformations': dataset_transformations,
    })
    return ba, multiscales_attrs


def _get_array_selection(arr, axes, timeindex: int | None,
                         ch:int | list[int] | None,
                         block_coords: tuple | None):
    ndim = arr.ndim

    if block_coords is None:
        coords_param = (slice(None,None),) * ndim
    elif len(block_coords) < ndim:
        coords_param = (slice(None,None),) * (ndim - len(block_coords)) + block_coords
    else:
        coords_param = block_coords

    selector = []
    selection_exists = False

    for ai, a in enumerate(axes):
        if a.get('type') == 'time':
            if timeindex is not None:
                selector.append(timeindex)
                selection_exists = True
            else:
                selector.append(coords_param[ai])
        elif a.get('type') == 'channel':
            if ch is None or ch == []:
                selector.append(coords_param[ai])
            else:
                selector.append(ch)
                selection_exists = True
        else:
            selector.append(coords_param[ai])

        selection_exists = (selection_exists or
                            coords_param[ai].start is not None or
                            coords_param[ai].stop is not None)

    if selection_exists:
        try:
            # try to select the data using the selector
            block_slice_coords = tuple(selector)
            logger.debug(f'Get block at {block_slice_coords}')
            return arr[block_slice_coords]
        except Exception  as e:
            logger.exception(f'Error selecting data with selector {tuple(selector)}')
            raise e
    else:
        return arr
