import logging
import numpy as np

from typing import Tuple, List


logger = logging.getLogger(__name__)


def get_block_crops(shape, blocksize, overlaps, mask, roi):
    """
    Given a voxel grid shape, blocksize, and overlap size, construct
       tuples of slices for every block; optionally only include blocks
       that contain foreground in the mask. Returns parallel lists,
       the block indices and the slice tuples.
    """
    blocksize = np.array(blocksize, dtype=int)
    blockoverlaps = np.array(overlaps, dtype=int) if overlaps else 0

    if mask is not None:
        mask_ratio = np.array(mask.shape) / shape[-3:] # shape may be 5-D so only consider the spatial shape
        logger.info(f'Mask ratio for {mask.shape} mask and {shape} image: {mask_ratio}')
    else:
        mask_ratio = 0

    indices, crops = [], []
    nblocks = get_nblocks(shape, blocksize)
    for index in np.ndindex(*nblocks):
        start_block = blocksize * index
        stop_block = np.minimum(start_block + blocksize, shape)
        block_crop = tuple(slice(x, y) for x, y in zip(start_block, stop_block))
        foreground = is_foreground_block(block_crop, mask, mask_ratio, roi, shape)
        if foreground:
            start_crop = np.maximum(0, start_block - blockoverlaps)
            stop_crop = np.minimum(stop_block + blockoverlaps, shape)
            crop = tuple(slice(x, y) for x, y in zip(start_crop, stop_crop))
            if roi is not None:
                logger.debug(f'Block {index} at {block_crop} (with overlaps {crop}) intersects the defined roi {roi}')
            indices.append(index)
            crops.append(crop)


        start = np.maximum(0, start)
        stop = np.minimum(shape, stop)
        crop = tuple(slice(x, y) for x, y in zip(start, stop))


    return indices, crops


def is_foreground_block(block, mask, mask_image_ratio, roi, image_shape):
    if mask is None and roi is None:
        return True

    spatial_crop = block[-3:] # consider only the spatial shape
    if mask is not None:
        mask_crop = tuple(
            slice(
                int(np.floor(s.start * r)),
                min(int(np.ceil(s.stop * r)), int(ms)),
            )
            for s, r, ms in zip(spatial_crop, mask_image_ratio, mask.shape)
        )
        return np.any(mask[mask_crop])
    else:
        # roi is a tuple (xmin,ymin,zmin[,xmax,ymax,zmax]) in XYZ order
        # block crop slices are in ZYX order; reverse mask coords to match
        roi_min_zyx = tuple(reversed(roi[:3]))
        if len(roi) == 6:
            roi_max_zyx = tuple(
                s if v < 0 else v
                for v, s in zip(reversed(roi[3:6]), image_shape[-3:])
            )
        else:
            roi_max_zyx = image_shape[-3:]
        # two intervals [a, b) and [c, d) intersect iff a < d and b > c
        return all(
            s.start < b_max and s.stop > b_min
            for s, b_min, b_max in zip(spatial_crop, roi_min_zyx, roi_max_zyx)
        )


def get_nblocks(shape, blocksize):
    """Given a shape and blocksize determine the number of blocks per axis"""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def prepare_blocksize(shape: Tuple[int, ...]|List[int],
                      blocksize: Tuple[int, ...]|List[int]) -> List[int]:
    ndim = len(shape)
    blocksize_ndim = len(blocksize)
    final_blocksize = []

    # the blocksize may have fewer elements than the image shape
    # in that case we right align it to shape
    # if somehow blocksize has more elements than shape
    # we drop the first elements until the sizes match
    offset = ndim - blocksize_ndim
    
    for si in range(ndim):
        final_blocksize.append(shape[si] if si < offset else blocksize[si - offset])

    return final_blocksize


def prepare_overlaps(shape: Tuple[int, ...], 
                     blocksize: Tuple[int, ...]|List[int],
                     blockoverlaps: Tuple[int, ...]|List[int]|None,
                     default_overlap: float|Tuple[float,...]|None=None) -> List[int]:
    shape_ndim = len(shape)
    blocksize_ndim = len(blocksize)

    def _get_default_overlap(dim):
        if isinstance(default_overlap, (int, float)):
            return int(default_overlap)
        elif isinstance(default_overlap, (list, tuple)):
            if len(default_overlap) > dim:
                return int(default_overlap[dim])
        # default to 10% of the corresponding blocksize
        return int(blocksize[dim] * 0.1)

    # If overlaps not provided (None / empty), compute defaults for every blocksize dim
    if not blockoverlaps:
        offset = shape_ndim - blocksize_ndim  # blocksize is right-aligned to shape
        return [
            0 if blocksize[i] == shape[i + offset] else _get_default_overlap(i)
            for i in range(blocksize_ndim)
        ]

    # If overlaps provided as list/tuple, right-align overlaps to blocksize
    if isinstance(blockoverlaps, (list, tuple)):
        bo_ndim = len(blockoverlaps)
        offset_shape = shape_ndim - blocksize_ndim
        offset_ov = max(blocksize_ndim - bo_ndim, 0)  # how much overlaps lags behind blocksize
        return [
            0 if blocksize[i] == shape[i + offset_shape]
            else (int(blockoverlaps[i - offset_ov])
                  if i >= offset_ov 
                  else _get_default_overlap(i))
            for i in range(blocksize_ndim)
        ]

    raise ValueError(f"Invalid block overlaps argument: {blockoverlaps}")


def remove_overlaps(array, crop, overlaps, blocksize):
    """
    Overlaps are only there to provide context for boundary voxels
    and can be removed after segmentation is complete
    reslice array to remove the overlaps
    """
    logger.debug((
        f'Remove overlaps: {overlaps} '
        f'crop: {crop} '
        f'blocksize is {blocksize} '
        f'block shape: {array.shape} '
    ))
    crop_trimmed = list(crop)
    for axis in range(array.ndim):
        # left side
        if crop[axis].start != 0:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(overlaps[axis], None)
            loverlap_index = tuple(slc)
            logger.debug((
                f'Remove left overlap on axis {axis}: {loverlap_index} ({type(loverlap_index)}) '
                f'from labeled block of shape: {array.shape} '
            ))
            array = array[loverlap_index]
            a, b = crop[axis].start, crop[axis].stop
            crop_trimmed[axis] = slice(a + overlaps[axis], b)
        # right side
        if array.shape[axis] > blocksize[axis]:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(None, blocksize[axis])
            roverlap_index = tuple(slc)
            logger.debug((
                f'Remove right overlap on axis {axis}: {roverlap_index} ({type(roverlap_index)}) '
                f'from labeled block of shape: {array.shape} '
            ))
            array = array[roverlap_index]
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, crop_trimmed
