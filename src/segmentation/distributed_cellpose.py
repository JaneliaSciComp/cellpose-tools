"""
This is code contributed by Greg Fleishman to run Cellpose on a Dask cluster.
"""
import dask_image.ndmeasure as di_ndmeasure
import logging
import numpy as np
import os
import scipy
import time
import torch
import traceback
import zarr

from cellpose import transforms
from cellpose.models import assign_device, CellposeModel
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import as_completed, Client
from typing import List

from .block_utils import (get_block_crops, get_nblocks,
                          prepare_blocksize, prepare_overlaps,
                          remove_overlaps)


logger = logging.getLogger(__name__)


def distributed_eval(
        input_zarr: zarr.Array,
        input_timeindex: int|None,
        input_channels: List[int]|None,
        blocksize,
        output_dir,
        labels_zarr: zarr.Array,
        dask_client: Client,
        blockoverlaps=(),
        mask=None,
        roi=None,
        preprocessing_steps=[],
        cellpose_model_args={},
        normalize_args={},
        cellpose_eval_args={},
        label_dist_th=1.0,
        skip_merge_labels=False,
):
    """
    Evaluate a cellpose model on overlapping blocks of a big image.
    Distributed over workstation or cluster resources with Dask.
    Optionally run preprocessing steps on the blocks before running cellpose.
    Optionally use a mask or roi to ignore background regions in image.

    The dask client must be present but it can be either a remote client that references
    a Dask Scheduler's IP or a local client.

    Parameters
    ----------
    input_zarr : zarr.Array
        Image data as a zarr array

    timeindex : string
        if the image is a 5-D TCZYX ndarray specify which timeindex to use

    input_channels : sequence[int] | None
        channels used for segmentation. If not set, it uses all channels
                
    blocksize : iterable
        The size of blocks in voxels. E.g. [128, 256, 256]

    dask_client : dask.distributed.Client
        A remote or locakl dask client.

        mask : numpy.ndarray (default: None)
        A foreground mask for the image data; may be at a different resolution
        (e.g. lower) than the image data. If given, only blocks that contain
        foreground will be processed. This can save considerable time and
        expense. It is assumed that the domain of the input_zarr image data
        and the mask is the same in physical units, but they may be on
        different sampling/voxel grids.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image blocks before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    Returns
    -------
    Two values are returned:
    (1) A reference to a dask array containing the stitched cellpose
        segments for your entire image
    (2) Bounding boxes for every segment. This is a list of tuples of slices:
        [(slice(z1, z2), slice(y1, y2), slice(x1, x2)), ...]
        The list is sorted according to segment ID. That is the smallest segment
        ID is the first tuple in the list, the largest segment ID is the last
        tuple in the list.
    """
    image_shape = input_zarr.shape
    logger.info((
        f'3D: {cellpose_eval_args.get("do_3D")}, '
        f'shape: {image_shape}, '
        f'process blocks: {blocksize} with {blockoverlaps} overlaps '
        f'timeindex: {input_timeindex} '
        f'image channels {input_channels} '
    ))
    diameter = cellpose_eval_args.get('diameter')
    blocksize = prepare_blocksize(image_shape, blocksize)
    blockoverlaps = prepare_overlaps(image_shape, blocksize, blockoverlaps,
                                     default_overlap=2 * diameter if diameter is not None else None)
    block_indices, block_crops = get_block_crops(
        image_shape, blocksize, blockoverlaps, mask, roi,
    )

    if len(block_indices) == 0:
        logger.info('No block was selected for segmentation')
        return labels_zarr, []

    logger.info((
        f'Start segmenting: {len(block_indices)} {blocksize} blocks '
        f'with overlap {blockoverlaps} '
        f'from a {image_shape} image '
    ))

    futures = dask_client.map(
        _process_block,
        block_indices,
        block_crops,
        input_zarr=input_zarr,
        input_timeindex=input_timeindex,
        input_channels=input_channels,
        blocksize=blocksize,
        blockoverlaps=blockoverlaps,
        labels_zarr=labels_zarr,
        preprocessing_steps=preprocessing_steps,
        cellpose_model_args=cellpose_model_args,
        normalize_args=normalize_args,
        cellpose_eval_args=cellpose_eval_args,
    )

    label_block_indices, faces, boxes = [], [], []
    all_label_ids = np.array([], dtype=np.uint32)

    for f, r in as_completed(futures, with_results=True):
        if f.cancelled():
            tb = f.traceback()
            logger.error(f'Block segmenting error: {''.join(traceback.format_tb(tb))}')
        else:
            bi, bfs, bboxes, blids = r
            logger.debug(f'Finished segmenting block {bi} (found {len(blids)} labels) ')
            label_block_indices.append(bi)
            faces.append(bfs)
            boxes.extend(bboxes)
            all_label_ids = np.concatenate([all_label_ids, blids]).astype(np.uint32)

    logger.info((
        f'Finished segmenting: {len(block_indices)} {blocksize} blocks '
        f'with overlap {blockoverlaps}'
        ' - start label merge process'
    ))

    logger.info((
        'Segmentation results contain '
        f'faces: {len(faces)}, boxes: {len(boxes)}, box_ids: {len(all_label_ids)}'
    ))

    if skip_merge_labels:
        logger.info((
            'Skip label merge process '
            f'returning {len(boxes)} labels '
        ))
        return labels_zarr, boxes
    else:
        return _merge_labels(label_block_indices, faces, boxes, all_label_ids,
                             labels_zarr, labels_zarr, dask_client, output_dir, label_dist_th)


def local_eval(
        input_image,
        input_timeindex,
        input_channels,
        labels_zarr,
        preprocessing_steps=[],
        cellpose_model_args={},
        normalize_args={},
        cellpose_eval_args={},
):
    """
    Evaluate a cellpose model on the entire image
    without distributing it to any dask cluster
    Optionally run preprocessing steps on the blocks before running cellpose.

    Parameters
    ----------
    input_image : ndarray
        Input image array

    input_timeindex : string
        if the image is a 5-D TCZYX ndarray specify which timeindex to use

    input_channels : sequence[int] | None
        channels used for segmentation. If not set, it uses all channels

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image blocks before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    Returns
    -------
    A reference to a dask array containing the stitched cellpose
    segments for your entire image.
    The reason for returning a dask array and not just the labels numpy array
    is to make the output compatible to the one returned by the distributed version
    
    """
    image_shape = input_image.shape
    logger.info((
        f'Segment (locally) {image_shape} image'
        f'3D: {cellpose_eval_args.get("do_3D")}, '
        f'timeindex: {input_timeindex} '
        f'image channels {input_channels} '
    ))
    labels = _read_preprocess_and_segment(
        input_image,
        input_timeindex,
        input_channels,
        None, # no cropping => entire image
        preprocessing_steps=preprocessing_steps,
        cellpose_model_args=cellpose_model_args,
        normalize_args=normalize_args,
        cellpose_eval_args=cellpose_eval_args,
    )
    _, nlabels = np.unique(labels, return_counts=True)
    if labels.ndim == labels_zarr.ndim:
        labels_zarr[...] = labels
    else:
        labels_zarr[0,...] = labels
    return labels_zarr, nlabels


def _process_block(
    block_index,
    crop,
    input_zarr,
    input_timeindex,
    input_channels,
    blocksize,
    blockoverlaps,
    labels_zarr,
    preprocessing_steps=[],
    cellpose_model_args={},
    normalize_args={},
    cellpose_eval_args={},
    max_labels_per_block=99999,
):
    """
    Preprocess and segment one block, of many, with eventual merger
    of all blocks in mind. The block is processed as follows:

    (1) Read block from disk, preprocess, and segment.
    (2) Remove overlaps.
    (3) Get bounding boxes for every segment.
    (4) Remap segment IDs to globally unique values.
    (5) Write segments to disk.
    (6) Get segmented block faces.

    (5) return remapped segments as a numpy array, boxes, and box_ids

    Parameters
    ----------
    block_index : tuple
        The (i, j, k, ...) index of the block in the overall block grid

    crop : tuple of slice objects
        The bounding box of the data to read from the input_zarr array

    image_container_path : string
        Path to image container.

    image_subpath : string
        Dataset path relative to image container.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image block before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    blocksize : iterable (list, tuple, np.ndarray)
        The number of voxels (the shape) of blocks without overlaps

    blocksoverlap : iterable (list, tuple, np.ndarray)
        The number of voxels added to the blocksize to provide context
        at the edges

    labels_output_zarr : zarr.core.Array
        A location where segments can be stored temporarily before
        merger is complete

    Returns
    -------
    faces : a list of numpy arrays - the faces of the block segments
    boxes : a list of crops (tuples of slices), bounding boxes of segments
    box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                boxes
    """
    logger.info((
        f'RUNNING BLOCK: {block_index}, '
        f'REGION: {crop}, '
        f'blocksize: {blocksize}, '
        f'blocksoverlap: {blockoverlaps}, '
        f'cellpose eval opts: {cellpose_eval_args}, '
        f'cellpose model opts: {cellpose_model_args}, '
    ))
    segmentation = _read_preprocess_and_segment(
        input_zarr,
        input_timeindex,
        input_channels,
        crop, 
        preprocessing_steps=preprocessing_steps,
        cellpose_model_args=cellpose_model_args,
        normalize_args=normalize_args,
        cellpose_eval_args=cellpose_eval_args,
    )
    seg_ndim = segmentation.ndim
    # labels are single channel so if the input was multichannel remove the channel coords
    labels_shape = labels_zarr.shape[-seg_ndim:]
    labels_block_index = block_index[-seg_ndim:]
    labels_coords = crop[-seg_ndim:]
    labels_overlaps = blockoverlaps[-seg_ndim:]
    labels_blocksize = blocksize[-seg_ndim:]

    logger.debug((
        f'adjusted labels image shape to {labels_shape} '
        f'labels block index to {labels_block_index} '
        f'labels block coords to {labels_coords} '
        f'labels block overlaps to {labels_overlaps} '
        f'labels block size to {labels_blocksize} '
    ))
    logger.info(f'Remove {labels_overlaps} overlaps from {segmentation.shape} labels')
    segmentation, labels_coords = remove_overlaps(segmentation, labels_coords, labels_overlaps, labels_blocksize)

    nblocks = get_nblocks(labels_shape, labels_blocksize)
    segmentation, label_ids = _global_segment_ids(segmentation, labels_block_index, nblocks,
                                                  max_labels_per_block=max_labels_per_block)
    if label_ids[0] == 0:
        label_ids = label_ids[1:]
    if labels_zarr.ndim != seg_ndim:
        labels_zarr_coords = (0,)*(labels_zarr.ndim-seg_ndim)+tuple(labels_coords)
        logger.debug(f'Write {segmentation.shape} labels for block {block_index} at {labels_zarr_coords} (zarr slice: {labels_coords})')
        labels_zarr[labels_zarr_coords] = segmentation
    else:
        logger.debug(f'Write {segmentation.shape} labels for block {block_index} at {labels_coords}')
        labels_zarr[tuple(labels_coords)] = segmentation
    boxes = _bounding_boxes_in_global_coordinates(segmentation, labels_coords)
    faces = _block_faces(segmentation)
    return labels_block_index, faces, boxes, label_ids


# ----------------------- component functions ---------------------------------#
def _read_preprocess_and_segment(
    input_zarr,
    input_timeindex,
    input_channels,
    crop,
    preprocessing_steps=[],
    cellpose_model_args={},
    normalize_args={},
    cellpose_eval_args={},
):
    """Read block from zarr array, run all preprocessing steps, run cellpose"""

    input_channel_axis = cellpose_eval_args.get('channel_axis')

    block_coords_list = [c for c in crop]
    if input_channel_axis is not None and input_channels:
        block_coords_list[input_channel_axis] = input_channels

    if input_timeindex is not None:
        # this should only be set for OME images if timepoints are present
        # and timepoints if present are the first dimension
        block_coords_list[0] = input_timeindex

    block_coords = tuple(block_coords_list)
    logger.info((
        f'Reading {block_coords} block from the input zarr '
        f'based on the input crop: {crop} '
        f'timeindex {input_timeindex} '
        f'channels {input_channels} '
        f'input channel axis {input_channel_axis} '
    ))

    image_block = input_zarr[block_coords]
    block_shape = image_block.shape
    block_ndim = image_block.ndim

    do_3D = cellpose_eval_args.get('do_3D', False)
    input_z_axis = cellpose_eval_args.get('z_axis')
    spatial_dims = 3 if do_3D else 2

    if input_z_axis is not None:
        # z axis is specified
        if input_timeindex is not None and input_z_axis > 0:
            z_axis = input_z_axis - 1
        else:
            z_axis = input_z_axis
    else:
        # z_axis is not specified
        if not do_3D:
            z_axis = None
        else:
            if block_ndim >= spatial_dims:
                z_axis = -3
            else:
                raise ValueError(f'Cannot handle {spatial_dims}-D segmentation for block of shape {block_shape}')

    if input_channel_axis is not None:
        # channel axis is specified
        if input_timeindex is not None and input_channel_axis > 0:
            channel_axis = input_channel_axis - 1
        else:
            channel_axis = input_channel_axis
    else:
        # channel axis is not specified
        if block_ndim == spatial_dims:
            # append a dimension for the channel if channel dimension is missing
            new_block_shape = (1,) + (block_shape)
            image_block = np.reshape(image_block, new_block_shape)
            channel_axis = 0 # channel is the first dimension
        else:
            # assume the channel axis is before the spatial axes
            channel_axis = block_ndim - spatial_dims - 1
    cellpose_eval_args['channel_axis'] = channel_axis
    cellpose_eval_args['z_axis'] = z_axis

    start_time = time.time()

    for pp_step in preprocessing_steps:
        logger.debug(f'Apply preprocessing step: {pp_step}')
        image_block = pp_step[0](image_block, **pp_step[1])

    model = _get_segmentation_model(cellpose_model_args)

    if normalize_args.get('normalize'):
        logger.info(f'Normalize {image_block.shape} block at {crop} params: {normalize_args}')
        image_block = transforms.normalize_img(image_block, axis=channel_axis,
                                               **normalize_args)
    logger.info(f'Eval {image_block.shape} block at {crop} args: {cellpose_eval_args}')
    try:
        labels = model.eval(image_block, **cellpose_eval_args)[0].astype(np.uint32)
    except Exception as e:
        logger.error((
            f'ERROR eval {image_block.shape} block at {crop} args: {cellpose_eval_args} '
            f'err={e} {traceback.format_exception(e)}'
        ))
        raise e

    end_time = time.time()
    unique_labels = np.unique(labels)
    logged_block_message = (f'for block: {crop}' 
                            if crop is not None
                            else 'for entire image')
    logger.info((
        'Finished model eval '
        f'{logged_block_message} => '
        f'found {len(unique_labels)} unique labels '
        f'in the {labels.shape} image '
        f'in {end_time-start_time}s '
    ))
    return labels


def _get_segmentation_model(cellpose_model_args):
    logger.info(f'Get segmentation model: {cellpose_model_args}')
    use_gpu = cellpose_model_args.get('use_gpu', True)
    gpu_device = cellpose_model_args.get('gpu_device', 0)
    if use_gpu:
        available_gpus = torch.cuda.device_count()
        logger.info(f'Found {available_gpus} GPUs')
        if available_gpus > 1:
            # if multiple gpus are available try to find one that can be used
            segmentation_device, gpu = None, False
            for gpui in range(available_gpus):
                try:
                    logger.debug(f'Try GPU: {gpui}')
                    segmentation_device, gpu = assign_device(gpu=use_gpu, device=gpui)
                    logger.debug(f'Result for GPU: {gpui} => {segmentation_device}:{gpu}')
                    if gpu:
                        break
                    # because of a bug in cellpose trying the other devices explicitly here
                    torch.cuda.set_device(gpui)
                    segmentation_device = torch.device(f'cuda:{gpui}')
                    logger.info(f'Device {segmentation_device} present and usable')
                    _ = torch.zeros((1,1)).to(segmentation_device)
                    logger.info(f'Device {segmentation_device} tested and it is usable')
                    gpu = True
                    break
                except Exception as e:
                    logger.warning(f'cuda:{gpui} present but not usable: {e}')
        else:
            segmentation_device, gpu = assign_device(gpu=use_gpu, device=gpu_device)

    else:
        segmentation_device, gpu = assign_device(gpu=use_gpu, device=gpu_device)
    return CellposeModel(
        gpu=gpu,
        device=segmentation_device,
        pretrained_model=cellpose_model_args.get('pretrained_model'),
    )


def _bounding_boxes_in_global_coordinates(segmentation, crop):
    """
    bounding boxes (tuples of slices) are super useful later
    best to compute them now while things are distributed
    """
    boxes = scipy.ndimage.find_objects(segmentation)
    boxes = [b for b in boxes if b is not None]

    def _translate(a, b):
        return slice(a.start+b.start, a.start+b.stop)

    for iii, box in enumerate(boxes):
        boxes[iii] = tuple(_translate(a, b) for a, b in zip(crop, box))
    return boxes


def _global_segment_ids(segmentation, block_index, nblocks, max_labels_per_block=99999):
    """
    Pack the block index into the segment IDs so they are
    globally unique. Everything gets remapped to [1..N] later.
    A label is split into 5 digits on left and 5 digits on right.
    This creates limits: 42950 maximum number of blocks and
    99999 maximum number of segments per block
    """
    unique, unique_inverse = np.unique(segmentation, return_inverse=True)
    logger.info((
        f'Block {block_index} out of {nblocks} blocks '
        f'- has {len(unique)} unique labels '
    ))

    max_local_label = np.max(unique)
    if max_local_label > max_labels_per_block:
        logger.error(f'Block {block_index} has more than 99999 labels ({np.max(unique)}) so this may generate label conflicts - use a smaller block')
        raise ValueError(f'Max label in block {block_index} is {max_local_label} may create possible clashes')

    p = str(np.ravel_multi_index(block_index, nblocks))
    max_label_digits = len(str(max_labels_per_block))
    remap = [int(p+str(x).zfill(max_label_digits)) for x in unique]
    if unique[0] == 0:
        remap[0] = 0  # 0 should just always be 0
    logger.debug(f'Remap: {remap}')
    segmentation = np.array(remap, dtype=np.uint32)[unique_inverse.reshape(segmentation.shape)]
    return segmentation, remap


def _block_faces(segmentation):
    """Slice faces along every axis"""
    faces = []
    for iii in range(segmentation.ndim):
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(0, 1)
        faces.append(segmentation[tuple(a)])
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(-1, None)
        faces.append(segmentation[tuple(a)])
    return faces


def _merge_labels(label_block_indices, faces, boxes, all_label_ids,
                  source_labels_zarr, target_labels_zarr, dask_client, output_dir, label_dist_th):
    logger.info((
        f'Relabel {all_label_ids.shape} labels of type {all_label_ids.dtype} - '
        f'use {len(faces)} faces for merging labels'
    ))
    new_labeling = _determine_merge_relabeling(label_block_indices, faces, all_label_ids,
                                               label_dist_th=label_dist_th)
    new_labeling_path = f'{output_dir}/new_labeling.npy'
    _write_new_labeling(new_labeling_path, new_labeling)

    logger.info(f'Relabel {all_label_ids.shape} blocks from {new_labeling_path}')
    label_slices = slices_from_chunks(
        normalize_chunks(target_labels_zarr.chunks, shape=target_labels_zarr.shape)
    )
    relabel_futures = dask_client.map(
        _relabel_block,
        label_slices,
        new_labeling=new_labeling_path,
        source=source_labels_zarr,
        target=target_labels_zarr,
    )
    relabel_res = True
    for f, r in as_completed(relabel_futures, with_results=True):
        if f.cancelled():
            tb = f.traceback()
            logger.error(f'Block relabel error: {''.join(traceback.format_tb(tb))}')
            relabel_res = False
        else:
            relabel_res = relabel_res and r
    logger.info(f'Relabeling final result: {relabel_res}')
    merged_boxes = _merge_all_boxes(boxes, new_labeling[all_label_ids.astype(np.int32)])
    return target_labels_zarr, merged_boxes


def distributed_merge(
        unmerged_labels_zarr: zarr.Array,
        blocksize,
        merged_labels_zarr: zarr.Array,
        output_dir,
        dask_client: Client,
        mask=None,
        roi=None,
        label_dist_th=1.0,
):
    labels_shape = unmerged_labels_zarr.shape
    blocksize = prepare_blocksize(labels_shape, blocksize)
    block_indices, block_crops = get_block_crops(
        labels_shape, blocksize, None, mask, roi,
    )
    if len(block_indices) == 0:
        # nothing to do, but we may still want to copy
        # unmerged_labels to merged_labels
        return unmerged_labels_zarr, []

    logger.info((
        f'Start merging: {len(block_indices)} {blocksize} label blocks '
        f'from a {labels_shape} segmented image '
    ))

    futures = dask_client.map(
        _get_label_merge_info,
        block_indices,
        block_crops,
        labels_zarr=unmerged_labels_zarr,
    )

    label_block_indices, faces, boxes = [], [], []
    all_label_ids = np.array([], dtype=np.uint32)

    for f, r in as_completed(futures, with_results=True):
        if f.cancelled():
            tb = f.traceback()
            logger.error(f'Block label extract error: {''.join(traceback.format_tb(tb))}')
        else:
            bi, bfs, bboxes, blids = r
            logger.debug(f'Finished getting label info for block {bi} (found {len(blids)} labels) ')
            label_block_indices.append(bi)
            faces.append(bfs)
            boxes.extend(bboxes)
            all_label_ids = np.concatenate([all_label_ids, blids]).astype(np.uint32)

    logger.info((
        f'Finished getting label info for {len(block_indices)} {blocksize} blocks '
        ' - start label merge process '
    ))

    logger.info((
        'Unmerged segmentation contains '
        f'faces: {len(faces)}, boxes: {len(boxes)}, box_ids: {len(all_label_ids)} '
    ))
    return _merge_labels(label_block_indices, faces, boxes, all_label_ids,
                         unmerged_labels_zarr, merged_labels_zarr, dask_client, output_dir, label_dist_th)



def _get_label_merge_info(
    block_index,
    crop,
    labels_zarr
):
    None # TODO



def _determine_merge_relabeling(block_indices, faces, labels,
                               label_dist_th=1.0):
    """Determine boundary segment mergers, remap all label IDs to merge
       and put all label IDs in range [1..N] for N global segments found"""
    faces = _adjacent_faces(block_indices, faces)
    logger.debug(f'Determine relabeling for {labels.shape} labels of type {labels.dtype}')
    used_labels = labels.astype(int)
    label_range = int(np.max(used_labels) + 1)
    label_groups = _block_face_adjacency_graph(faces, label_range,
                                               label_dist_th=label_dist_th)
    logger.debug((
        f'Build connected components for {label_groups.shape} label groups'
        f'{label_groups}'
    ))
    new_labeling = scipy.sparse.csgraph.connected_components(label_groups,
                                                             directed=False)[1]
    logger.debug(f'Initial {new_labeling.shape} connected labels:, {new_labeling}')
    # XXX: new_labeling is returned as int32. Loses half range. Potentially a problem.
    unused_labels = np.ones(label_range, dtype=bool)
    unused_labels[used_labels] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique))[unique_inverse]
    logger.debug(f'Re-arranged {len(new_labeling)} connected labels:, {new_labeling}')
    return new_labeling


def _adjacent_faces(block_indices, faces):
    """Find faces which touch and pair them together in new data structure"""
    face_pairs = []
    faces_index_lookup = {bi: f for bi, f in zip(block_indices, faces)}
    for block_index in block_indices:
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = faces_index_lookup[block_index][2*ax + 1]
                b = faces_index_lookup[neighbor_index][2*ax]
                face_pairs.append(np.concatenate((a, b), axis=ax))
            except KeyError:
                continue
    return face_pairs


def _block_face_adjacency_graph(faces, labels_range, label_dist_th=1.0):
    """
    Shrink labels in face plane, then find which labels touch across the face boundary
    """
    logger.info(f'Create adjacency graph for labels with a maximum range of {labels_range}')
    all_mappings = [np.empty((2, 0), dtype=np.uint32)]
    structure = scipy.ndimage.generate_binary_structure(3, 1)
    for face in faces:
        sl0 = tuple(slice(0, 1) if d == 2 else slice(None) for d in face.shape)
        sl1 = tuple(slice(1, 2) if d == 2 else slice(None) for d in face.shape)
        a = _shrink_labels(face[sl0], label_dist_th)
        b = _shrink_labels(face[sl1], label_dist_th)
        face = np.concatenate((a, b), axis=np.argmin(a.shape))
        mapped = di_ndmeasure._utils._label._across_block_label_grouping(
            face,
            structure
        )
        all_mappings.append(mapped)
    i, j = np.concatenate(all_mappings, axis=1)
    v = np.ones_like(i)
    csr_mat = scipy.sparse.coo_matrix((v, (i, j)),
                                      shape=(labels_range,labels_range)).tocsr()
    logger.debug(f'Labels mapping as csr matrix {csr_mat}')
    return csr_mat


def _shrink_labels(plane, threshold):
    """
    Shrink labels in plane by some distance from their boundary
    """
    gradmag = np.linalg.norm(np.gradient(plane.squeeze()), axis=0)
    shrunk_labels = np.copy(plane.squeeze())
    shrunk_labels[gradmag > 0] = 0
    distances = scipy.ndimage.distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def _merge_all_boxes(boxes, box_ids):
    """Merge all boxes that map to the same box_ids"""
    merged_boxes = []
    boxes_array = np.array(boxes, dtype=object)
    for iii in np.unique(box_ids):
        merge_indices = np.argwhere(box_ids == iii).squeeze()
        if merge_indices.shape:
            merged_box = _merge_boxes(boxes_array[merge_indices])
        else:
            merged_box = boxes_array[merge_indices]
        merged_boxes.append(merged_box)
    return merged_boxes


def _merge_boxes(boxes):
    """Take union of two or more parallelpipeds"""
    box_union = boxes[0]
    for iii in range(1, len(boxes)):
        local_union = []
        for s1, s2 in zip(box_union, boxes[iii]):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            local_union.append(slice(start, stop))
        box_union = tuple(local_union)
    return box_union


def _write_new_labeling(new_labeling_path, new_labeling):
    new_labeling_dir = os.path.dirname(new_labeling_path)
    os.makedirs(new_labeling_dir, exist_ok=True)
    logger.info(f'Save new label assignment to {new_labeling_path}')
    np.save(new_labeling_path, new_labeling)


def _relabel_block(block_coords, new_labeling=None,
                   source=[], target=[]):
    if new_labeling is None:
        return False
    else:
        logger.info(f'Relabel block: {block_coords}')
        block = source[block_coords]
        new_labeling_array = np.load(new_labeling)
        logger.info(f'Apply {len(new_labeling_array)} to {block.shape} at {block_coords}')
        target[block_coords] = new_labeling_array[block]
        return True
