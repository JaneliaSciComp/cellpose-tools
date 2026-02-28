import json
import logging
import numpy as np
import os
import sys
import traceback
import zarr

from cellpose.cli import get_arg_parser

from dask.distributed import (Client, LocalCluster)

from io_utils.read_utils import open_array, read_array_attrs
from io_utils.write_utils import container_type, write_zarray_as

from math import floor

from segmentation.distributed_cellpose import (distributed_eval, local_eval)
from segmentation.preprocessing import get_preprocessing_steps

from utils.configure_logging import (configure_logging)
from utils.configure_dask import (load_dask_config, ConfigureWorkerPlugin)

from zarr_tools.ngff.ngff_utils import (create_ome_metadata, get_axes_dictindex,
                                        get_non_spatial_axes, get_spatial_voxel_spacing)
from zarr_tools.io.zarr_io import create_zarr_array

from .cli import dictfromjson, floattuple, inttuple, intlist, stringlist
from .download_models import download_cellpose_models


logger:logging.Logger


def _define_args():
    args_parser = get_arg_parser()
    args_parser.add_argument('-i','--input',
                             dest='input',
                             type=str,
                             required=False,
                             help = "Input image to be segmented - it can be a directory path for zarr or N5 or file path for tiff")
    args_parser.add_argument('--input-subpath', '--input_subpath',
                             dest='input_subpath',
                             type=str,
                             help = "Input dataset subpath in case the input image is a zarr or N5 container")
    args_parser.add_argument('--timeindex',
                             dest='input_timeindex',
                             type=int,
                             default=None,
                             help = "Time index in case the input is an OME-ZARR container")
    args_parser.add_argument('--input-channels', '--input_channels',
                             dest='input_channels',
                             type=intlist,
                             help = "Input segmentation channels")

    args_parser.add_argument('--voxel-spacing', '--voxel_spacing',
                             dest='voxel_spacing',
                             type=floattuple,
                             metavar='X,Y,Z',
                             help = "Spatial voxel spacing as X,Y,Z")

    args_parser.add_argument('--mask',
                             dest='mask',
                             type=str,
                             help = "Mask directory")
    args_parser.add_argument('--mask-subpath', '--mask_subpath',
                             dest='mask_subpath',
                             type=str,
                             help = "mask subpath")

    args_parser.add_argument('--roi',
                             dest='roi',
                             type=inttuple,
                             metavar="xmin,ymin,zmin,xmax,ymax,zmax",
                             help='Fixed volume mask descriptor a tuple of 6 values representing min and max voxel coordinates')

    args_parser.add_argument('-o','--output',
                             dest='output',
                             required=False,
                             type=str,
                             help = "output image name - it can be a directory path for zarr or N5 or file path for TIFF")
    args_parser.add_argument('--output-subpath', '--output_subpath',
                             dest='output_subpath',
                             type=str,
                             help = "output dataset subpath")
    args_parser.add_argument('--output-chunk-size', '--output_chunk_size',
                             dest='output_chunk_size',
                             default=128,
                             type=int,
                             help='Output chunk size as a single int')
    args_parser.add_argument('--output-blocksize', '--output_blocksize',
                             dest='output_blocksize',
                             type=inttuple,
                             metavar='X,Y,Z',
                             help='Output chunk size as a tuple (x,y,z).')
    args_parser.add_argument('--compressor', '--compression',
                             dest='compressor',
                             default='zstd',
                             help='Zarr array compression algorithm')
    args_parser.add_argument('--compressor-opts', '--compression-opts',
                             dest='compressor_opts',
                             type=dictfromjson,
                             default={},
                             help='Zarr array compression options')
    args_parser.add_argument('--with-ome-labels',
                             dest='with_label_values',
                             action='store_true',
                             default=False,
                             help='If set output label values in OME metadata')
    args_parser.add_argument('--zarr-format', '--zarr_format',
                             type=int,
                             default=2,
                             dest='zarr_format',
                             help='Zarr format (2 or 3 for v2 or v3)')

    args_parser.add_argument('--working-dir', '--working_dir',
                             dest='working_dir',
                             default='.',
                             type=str,
                             help = "output file")

    args_parser.add_argument('--process-blocksize', '--process_blocksize',
                             dest='process_blocksize',
                             type=inttuple,
                             help='Output chunk size as a tuple (x,y,z).')
    args_parser.add_argument('--blocks-overlaps', '--blocks_overlaps',
                             dest='blocks_overlaps',
                             type=inttuple,
                             metavar='dX,dY,dZ',
                             help='Blocks overlaps as a tuple (x,y,z).')
    args_parser.add_argument('--max-size-fraction', '--max_size_fraction',
                             dest='max_size_fraction',
                             type=float,
                             default=0.4,
                             help='Fraction of the total image for which the masks are discarded')
    args_parser.add_argument('--norm-lowhigh', '--norm_lowhigh',
                             dest='norm_lowhigh',
                             nargs=2,  # Require exactly two values
                             metavar=('VALUE1', 'VALUE2'),
                             help="Provide two values to set low and high normalize value")
    args_parser.add_argument('--normalize-sharpen-radius', '--normalize_sharpen_radius',
                             dest='normalize_sharpen_radius',
                             type=float,
                             default=0,
                             help='Sharpen radius used for normalization')
    args_parser.add_argument('--normalize-smooth-radius', '--normalize_smooth_radius',
                             dest='normalize_smooth_radius',
                             type=float,
                             default=0,
                             help='Smooth radius used for normalization')
    args_parser.add_argument('--normalize-invert', '--normalize_invert',
                             dest='normalize_invert',
                             action='store_true',
                             default=False,
                             help="Normalize invert")
    args_parser.add_argument('--expansion-factor', '--expansion_factor',
                             dest='expansion_factor',
                             type=float,
                             default=0.,
                             help='Sample expansion factor')

    distributed_args = args_parser.add_argument_group("Distributed Arguments")
    distributed_args.add_argument('--dask-scheduler', '--dask_scheduler',
                                  dest='dask_scheduler',
                                  type=str,
                                  default=None,
                                  help='The TCP/IP address (tcp://x.x.x.x:port) of the Dask scheduler used for distributing cellpose over an existing dask cluster')
    distributed_args.add_argument('--dask-config', '--dask_config',
                                  dest='dask_config',
                                  type=str,
                                  default=None,
                                  help='Dask configuration yaml file')
    distributed_args.add_argument('--local-dask-workers', '--local_dask_workers',
                                  dest='local_dask_workers',
                                  type=int,
                                  default=0,
                                  help='Number of workers when using a local cluster')
    distributed_args.add_argument('--worker-cpus', '--worker_cpus',
                                  dest='worker_cpus',
                                  type=int,
                                  default=1,
                                  help='Number of cpus allocated to a dask worker')
    distributed_args.add_argument('--device',
                                  type=str,
                                  default='0',
                                  dest='device',
                                  help='which device to use, use an integer for torch, or mps for M1')    
    distributed_args.add_argument('--models-dir', '--models_dir',
                                  dest='models_dir',
                                  type=str,
                                  help='cache cellpose models directory')
    distributed_args.add_argument('--model', '--pretrained-model',
                                  dest='segmentation_model',
                                  type=str,
                                  default='cpsam',
                                  help='A builtin segmentation model or a model added to the cellpose models directory')
    distributed_args.add_argument('--label-distance-threshold', '--label-dist-th',
                                  dest='label_dist_th',
                                  type=float,
                                  default=1.0,
                                  help='Label distance transform threshold used for merging labels')
    
    distributed_args.add_argument('--skip-merge-labels', '--skip_merge_labels',
                                  dest='skip_merge_labels',
                                  action='store_true',
                                  default=False,
                                  help='Skip label merging across blocks and save boxes/box_ids to the working dir')

    distributed_args.add_argument('--preprocessing-steps', '--preprocessing_steps',
                                  dest='preprocessing_steps',
                                  type=stringlist,
                                  default=[],
                                  help='Preprocessing steps to run before cellpose')

    distributed_args.add_argument('--preprocessing-config', '--preprocessing_config',
                                  dest='preprocessing_config',
                                  type=str,
                                  help='YAML file containing parameters for the preprocessing steps')
    
    distributed_args.add_argument('--logging-config', dest='logging_config',
                                  type=str,
                                  help='python log file configuration')

    return args_parser


def _run_segmentation(args):
    load_dask_config(args.dask_config)
    if args.models_dir is not None:
        models_dir = os.path.realpath(args.models_dir)
        os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = models_dir
    elif os.environ.get('CELLPOSE_LOCAL_MODELS_PATH'):
        models_dir = os.environ['CELLPOSE_LOCAL_MODELS_PATH']
    else:
        models_dir = None


    if models_dir is not None and args.segmentation_model:
        pretrained_model = os.path.join(models_dir, args.segmentation_model)
    else:
        pretrained_model = args.segmentation_model

    logger.info(f'Download cellpose models to {models_dir} ({os.environ.get('CELLPOSE_LOCAL_MODELS_PATH')})')
    download_cellpose_models(models_dir, args.segmentation_model)

    if args.dask_scheduler:
        logger.info(f'Create dask client for {args.dask_scheduler}')
        dask_client = Client(address=args.dask_scheduler)
    elif args.local_dask_workers > 0:
        # use a local asynchronous client
        logger.info(f'Create local dask client with {args.local_dask_workers} local workers')
        dask_client = Client(LocalCluster(n_workers=args.local_dask_workers,
                                          threads_per_worker=args.worker_cpus))
    else:
        logger.info('Use in process cellpose segmentation')
        dask_client = None

    if dask_client is not None:
        logger.info(f'Initialize Dask Worker plugin with: {models_dir}, {args.logging_config}')
        worker_config = ConfigureWorkerPlugin(models_dir,
                                              args.logging_config,
                                              args.verbose,
                                              worker_cpus=args.worker_cpus)
        dask_client.register_plugin(worker_config, name='WorkerConfig')

    input_image_attrs = read_array_attrs(args.input, args.input_subpath)
    input_image_ndim = input_image_attrs['array_ndim']
    input_image_shape = input_image_attrs['array_shape']
    input_image_dtype = input_image_attrs['array_dtype']

    if args.voxel_spacing is not None:
        # voxel spacing is specified in the command line, so use this value
        voxel_spacing = args.voxel_spacing[::-1] # this is specified as XYZ and we want it as ZYX
    else:
        voxel_spacing = get_spatial_voxel_spacing(input_image_attrs)

    if voxel_spacing is not None:
        if args.expansion_factor > 0:
            voxel_spacing = [c / args.expansion_factor for c in voxel_spacing]
    else:
        voxel_spacing = (1,) * (3 if args.do_3D else 2)

    logger.info(f'Image data shape/dim/dtype: {input_image_shape}, {input_image_ndim}, {input_image_dtype}')
    
    if args.output:
        try:
            if args.anisotropy and args.anisotropy != 1.0:
                anisotropy = args.anisotropy
            else:
                if voxel_spacing is not None:
                    anisotropy = voxel_spacing[0] / voxel_spacing[1]
                else:
                    anisotropy = None

            preprocessing_steps = get_preprocessing_steps(args.preprocessing_steps, 
                                                          args.preprocessing_config,
                                                          voxel_spacing=voxel_spacing)
            logger.info(f'Preprocessing steps: {preprocessing_steps}')

            image_axes = get_axes_dictindex(input_image_attrs)
            if args.input_timeindex is not None:
                input_timeindex = args.input_timeindex
            else:
                # if no input timeindex arg was provided
                # and if it's an OME that has time index, default it to 0
                if image_axes.get('t') is not None:
                    input_timeindex = 0
                else:
                    # assume the input image has no timepoints dimension
                    input_timeindex = None

            # prepare the channel_axis and the z_axis arg
            if args.channel_axis is not None:
                channel_axis = args.channel_axis
            else:
                channel_axis = image_axes.get('c', None)

            if args.z_axis is not None:
                z_axis = args.z_axis
            else:
                z_axis = image_axes.get('z', None)

            normalize_lowhigh = ((int(args.norm_lowhigh[0]), int(args.norm_lowhigh[1]))
                                    if args.norm_lowhigh is not None else None)
            normalize_percentile = ((int(args.norm_percentile[0]), int(args.norm_percentile[1]))
                                    if args.norm_percentile is not None else None)
            cellpose_model_args = {
                'use_gpu': args.use_gpu,
                'gpu_device': args.gpu_device,
                'pretrained_model': pretrained_model,
            }
            normalize_args = {
                'normalize': not args.no_norm,
                'lowhigh': normalize_lowhigh,
                'percentile': normalize_percentile,
                'norm3D': args.do_3D,
                'sharpen_radius': args.normalize_sharpen_radius,
                'smooth_radius': args.normalize_smooth_radius,
                'tile_norm_blocksize': 0,
                'tile_norm_smooth3D': 1,
                'invert': args.normalize_invert,
            }
            cellpose_eval_args = {
                'diameter': args.diameter,
                'do_3D': args.do_3D,
                'min_size': args.min_size,
                'max_size_fraction': args.max_size_fraction,
                'niter': args.niter,
                'anisotropy': anisotropy,
                'z_axis': z_axis,
                'channel_axis': channel_axis,
                'flow_threshold': args.flow_threshold,
                'cellprob_threshold': args.cellprob_threshold,
                'stitch_threshold': args.stitch_threshold,
                'flow3D_smooth': args.flow3D_smooth,
                'batch_size': 8,
            }
            input_image_array = open_array(input_image_attrs['array_storepath'], input_image_attrs['array_subpath'])
            labels_zarr = _create_output_labels_zarr(args, input_image_attrs)
            if dask_client is not None:
                # set the process size and the blocks overlap
                if args.process_blocksize is not None:
                    # process_blocksize are specified (as X,Y,Z) so revert them
                    process_blocksize = args.process_blocksize[::-1]
                    output_chunks = np.array(labels_zarr.chunks[-3:])
                    if np.any(output_chunks > process_blocksize):
                        logger.error(f'Process size {process_blocksize} must be >= spatial values of {labels_zarr.chunks}')
                        raise ValueError((
                            f'Processing block size {process_blocksize} is too small '
                            f'compared to chunk size: {output_chunks}'
                        ))
                else:
                    process_blocksize = input_image_shape # process the whole image

                if args.blocks_overlaps is not None:
                    # blocks_overlaps are also specified as dX,dY,dZ overlaps 
                    # so we need to revert them
                    blocks_overlaps = args.blocks_overlaps[::-1]
                else:
                    blocks_overlaps = ()

                logger.info((
                    f'Invoke distributed segmentation {input_image_attrs['array_storepath']}:{input_image_attrs['array_subpath']} '
                    f'timeindex: {input_timeindex}, input channels: {args.input_channels} '
                    f'process block size: {process_blocksize}, blocks overlaps: {blocks_overlaps}'
                ))

                output_labels, boxes = distributed_eval(
                    input_image_array,
                    input_timeindex,
                    args.input_channels,
                    process_blocksize,
                    args.working_dir,
                    labels_zarr,
                    dask_client,
                    blockoverlaps=blocks_overlaps,
                    mask=None,
                    roi=args.roi,
                    preprocessing_steps=preprocessing_steps,
                    cellpose_model_args=cellpose_model_args,
                    normalize_args=normalize_args,
                    cellpose_eval_args=cellpose_eval_args,
                    label_dist_th=args.label_dist_th,
                    skip_merge_labels=args.skip_merge_labels,
                )
                nlabels = len(boxes)
            else:
                input_image_array = open_array(input_image_attrs['array_storepath'], input_image_attrs['array_subpath'])
                output_labels, nlabels = local_eval(
                    input_image_array,
                    args.input_timeindex,
                    args.input_channels,
                    labels_zarr,
                    preprocessing_steps=preprocessing_steps,
                    cellpose_model_args=cellpose_model_args,
                    normalize_args=normalize_args,
                    cellpose_eval_args=cellpose_eval_args,
                )

            logger.info(f'Finished segmentation process. Found {nlabels-1} labels')
            output__container_type = container_type(args.output)
            if output__container_type != 'zarr':
                logger.info(f'Save output labels as {output__container_type} at {args.output}')
                write_zarray_as(output_labels, args.output, args.output_subpath)
            elif args.with_label_values and nlabels > 0:
                _update_image_label_attrs(output_labels, nlabels)

            if dask_client is not None:
                dask_client.close()

        except:
            raise


def _print_version_and_exit():
    from cellpose import version_str as cellpose_version

    print(cellpose_version)
    sys.exit(0)


def _create_output_labels_zarr(args, image_attrs, labels_dtype='uint32'):
    """
    Create a zarr array with the same dimensionality as the input image,
    not necessarily the same shape but the labels zarr
    should have the same number of dimensions
    """
    output_labels_container_type = container_type(args.output)
    output_subpath = args.output_subpath if args.output_subpath else args.input_subpath

    if output_labels_container_type != 'zarr':
        # since the output is not a zarr - create a temporary zarr to hold the labels
        labels_zarr_path = f'{args.working_dir}/segmentation.zarr'
        labels_array_subpath = 'block_labels'
        ome_metadata = {}
    else:
        labels_zarr_path = args.output
        labels_array_subpath = output_subpath
        image_transforms = image_attrs.get('array_transforms', {})
        ome_metadata = create_ome_metadata(
            os.path.basename(labels_zarr_path),
            labels_array_subpath,
            image_attrs.get('array_axes'),
            image_transforms.get('scale'),
            image_transforms.get('translation'),
            image_attrs.get('array_ndim'),
            ome_version='0.4'
        )

    image_ndim = image_attrs['array_ndim']
    non_spatial_axes = get_non_spatial_axes(image_attrs).values()
    # the output labels image should have the same dimensions as the input image
    input_image_shape = image_attrs['array_shape']
    labels_shape = tuple(input_image_shape[di] 
                         if di not in non_spatial_axes else 1
                         for di in range(len(input_image_shape)))

    if args.output_blocksize is not None:
        # if output blocksize is specified, use it but 
        # ensure that it has the same number of dimensions as the input image
        if len(args.output_blocksize) < image_ndim:
            # make the chunksize 1 for missing dimensions
            # also since the output blocksize is specified as X,Y,Z - revert it to Z,Y,X
            output_blocksize = (args.output_blocksize + (1,) * (image_ndim - len(args.output_blocksize)))[::-1]
        else:
            # the blocksize already has the same number of dimensions as the input,
            # just reverse it because in the command line we specify as x,y,z[,c,t]
            output_blocksize = args.output_blocksize[::-1]
    else:
        # default to output_chunk_size for spatial axes or 1 for the other axes
        if non_spatial_axes == ():
            output_blocksize = (args.output_chunk_size,) * image_ndim
        else:
            output_blocksize = (1,) * len(non_spatial_axes) + (args.output_chunk_size,) * (image_ndim - len(non_spatial_axes))

    logger.info((
        f'Create labels zarr {labels_zarr_path}:{labels_array_subpath} '
        f'zarr format: {args.zarr_format}, shape: {labels_shape}, chunksize: {output_blocksize} '
        f'compressor: {args.compressor} '
    ))

    return create_zarr_array(
        labels_zarr_path,
        labels_array_subpath,
        labels_shape,
        output_blocksize,
        labels_dtype,
        compressor=args.compressor,
        compression_opts=args.compressor_opts,
        parent_array_attrs=ome_metadata,
        zarr_format=args.zarr_format,
    )


def _update_image_label_attrs(labels_zarr:zarr.Array, nlabels):
    image_label_attrs = _create_image_label_attrs(nlabels)
    parent_path = os.path.dirname(labels_zarr.path)
    if parent_path != '':
        # array not at the root
        parent_group = zarr.open_group(store=labels_zarr.store, path=parent_path, mode="a")
        parent_group.attrs.update({
            'image-label': image_label_attrs,
        })


def _create_image_label_attrs(nlabels:int, ome_ngff_version='0.4', source_image_path='..'):
    """
    Create image label attributes for <nlabels> labels
    """
    # set HSV parameters
    s = 1.0
    v = 1.0
    colors = []
    for lnum in range(nlabels-1):
        h = lnum/nlabels
        r, g, b = _hsv2rgb(h, s, v)
        colors.append({
            'label-value': lnum + 1,
            'rgba': [r, g, b, 255]
        })

    return {
        'colors': colors,
        'version': ome_ngff_version,
        'source': {
            'image': source_image_path
        }
    }


def _hsv2rgb(h:float, s:float, v:float):
    i = floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if i % 6 == 0:
        r = v
        g = t
        b = p
    elif i % 6 == 1:
        r = q
        g = v
        b = p
    elif i % 6 == 2:
        r = p
        g = v
        b = t
    elif i % 6 == 3:
        r = p
        g = q
        b = v
    elif i % 6 == 4:
        r = t
        g = p
        b = v
    else: # i % 6 == 5
        r = v
        g = p
        b = q

    return round(r * 255), round(g * 255), round(b * 255)


def _main():
    args_parser = _define_args()
    args = args_parser.parse_args()

    try:
        if args.version:
            _print_version_and_exit()

        # validate args
        if not args.input:
            args_parser.error("--input is required")
        if not args.output:
            args_parser.error("--output is required")
        if args.roi is not None and len(args.roi) not in (3, 6):
            args_parser.error(f"--roi must have 3 - minx,miny,minz only, or 6 minx,miny,minz and maxx,maxy,maxz values, got {len(args.roi)} ({args.roi})")

        # prepare logging
        global logger
        logger = configure_logging(args.logging_config, args.verbose)
        logger.info(f'Invoked cellpose segmentation with: {args}')

    except Exception as err:
        print('Logging configuration error:', err)
        traceback.print_exception(err)
        sys.exit(1)

    # run segmentation
    _run_segmentation(args)


if __name__ == '__main__':
    _main()
