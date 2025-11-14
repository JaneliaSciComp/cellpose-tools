from math import floor
import os
import sys
import traceback

import io_utils.read_utils as read_utils
import io_utils.write_utils as write_utils

from cellpose.cli import get_arg_parser

from dask.distributed import (Client, LocalCluster)

from segmentation.cellpose import (distributed_eval, local_eval)
from segmentation.preprocessing import get_preprocessing_steps

from io_utils.zarr_utils import prepare_parent_group_attrs

from utils.configure_logging import (configure_logging)
from utils.configure_dask import (load_dask_config, ConfigureWorkerPlugin)


def _floattuple(arg):
    if arg is not None and arg.strip():
        return tuple([float(d) for d in arg.split(',')])
    else:
        return ()


def _inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def _intlist(arg):
    if arg is not None and arg.strip():
        return [int(d) for d in arg.split(',')]
    else:
        return []


def _stringlist(arg):
    if arg is not None and arg.strip():
        return list(filter(lambda x: x, [s.strip() for s in arg.split(',')]))
    else:
        return []


def _define_args():
    args_parser = get_arg_parser()
    args_parser.add_argument('-i','--input',
                             dest='input',
                             type=str,
                             required=True,
                             help = "Input image to be segmented - it can be a directory path for zarr or N5 or file path for tiff")
    args_parser.add_argument('--input-subpath', '--input_subpath',
                             dest='input_subpath',
                             type=str,
                             help = "Input dataset subpath in case the input image is a zarr or N5 container")
    args_parser.add_argument('--timeindex',
                             dest='input_timeindex',
                             type=int,
                             default=0,
                             help = "Time index in case the input is an OME-ZARR container")
    args_parser.add_argument('--input-channels', '--input_channels',
                             dest='input_channels',
                             type=_intlist,
                             help = "Input segmentation channels")

    args_parser.add_argument('--voxel-spacing', '--voxel_spacing',
                             dest='voxel_spacing',
                             type=_floattuple,
                             help = "voxel spacing")

    args_parser.add_argument('--mask',
                             dest='mask',
                             type=str,
                             help = "mask directory")
    args_parser.add_argument('--mask-subpath', '--mask_subpath',
                             dest='mask_subpath',
                             type=str,
                             help = "mask subpath")

    args_parser.add_argument('-o','--output',
                             dest='output',
                             required=True,
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
                             type=_inttuple,
                             help='Output chunk size as a tuple (x,y,z).')

    args_parser.add_argument('--working-dir', '--working_dir',
                             dest='working_dir',
                             default='.',
                             type=str,
                             help = "output file")

    args_parser.add_argument('--process-blocksize', '--process_blocksize',
                             dest='process_blocksize',
                             type=_inttuple,
                             help='Output chunk size as a tuple (x,y,z).')
    args_parser.add_argument('--blocks-overlaps', '--blocks_overlaps',
                             dest='blocks_overlaps',
                             type=_inttuple,
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
                                  type=str, default=None,
                                  help='The TCP/IP address (tcp://x.x.x.x:port) of the Dask scheduler used for distributing cellpose over an existing dask cluster')
    distributed_args.add_argument('--dask-config', '--dask_config',
                                  dest='dask_config',
                                  type=str, default=None,
                                  help='Dask configuration yaml file')
    distributed_args.add_argument('--local-dask-workers', '--local_dask_workers',
                                  dest='local_dask_workers',
                                  type=int,
                                  default=0,
                                  help='Number of workers when using a local cluster')
    distributed_args.add_argument('--worker-cpus', '--worker_cpus',
                                  dest='worker_cpus',
                                  type=int, default=1,
                                  help='Number of cpus allocated to a dask worker')
    distributed_args.add_argument('--device', required=False, default='0', type=str,
                                  dest='device',
                                  help='which device to use, use an integer for torch, or mps for M1')    
    distributed_args.add_argument('--models-dir', '--models_dir',
                                  dest='models_dir',
                                  type=str,
                                  help='cache cellpose models directory')
    distributed_args.add_argument('--model',
                                  dest='segmentation_model',
                                  type=str,
                                  default='cpsam',
                                  help='A builtin segmentation model or a model added to the cellpose models directory')
    distributed_args.add_argument('--label-distance-threshold', '--label-dist-th',
                                  dest='label_dist_th',
                                  type=float,
                                  default=1.0,
                                  help='Label distance transform threshold used for merging labels')
    
    distributed_args.add_argument('--preprocessing-steps', '--preprocessing_steps',
                                  dest='preprocessing_steps',
                                  type=_stringlist,
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

    if models_dir:
        from cellpose.models import get_user_models

        logger.info(f'Download cellpose models to {models_dir}')
        get_user_models()

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

    image_data, image_attrs = read_utils.open(args.input, args.input_subpath)
    image_ndim = image_data.ndim
    image_shape = image_data.shape
    image_dtype = image_data.dtype
    image_data = None

    if args.voxel_spacing is not None:
        voxel_spacing = read_utils.get_voxel_spacing({}, args.voxel_spacing)
    else:
        voxel_spacing = read_utils.get_voxel_spacing(image_attrs, (1.,) * image_ndim)

    if voxel_spacing is not None:
        if args.expansion_factor > 0:
            expansion = args.expansion_factor
        else:
            expansion = 1.
        voxel_spacing /= expansion
        spatial_ndims = len(voxel_spacing)
    else:
        spatial_ndims = 3 # assume 3D for now

    logger.info(f'Image data shape/dim/dtype: {image_shape}, {image_ndim}, {image_dtype}')
    
    if args.output:
        output_subpath = args.output_subpath if args.output_subpath else args.input_subpath

        if args.process_blocksize is not None:
            if len(args.process_blocksize) < image_ndim:
                # append 0s
                process_blocksize_arg = (args.process_blocksize +
                                        (0,) * (image_ndim - len(args.process_blocksize)))
            else:
                process_blocksize_arg = args.process_blocksize
            zyx_process_size = process_blocksize_arg[::-1] # make it zyx
            process_blocksize = tuple([d if d > 0 else image_shape[di]
                                        for di,d in enumerate(zyx_process_size)])
        else:
            process_blocksize = image_shape # process the whole image

        if (args.blocks_overlaps is not None and
            len(args.blocks_overlaps) > 0):
            blocks_overlaps = args.blocks_overlaps[::-1] # make it zyx
        else:
            blocks_overlaps = ()

        try:
            logger.info(f'Invoke segmentation for {image_shape} with process blocksize {process_blocksize}')

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

            if args.z_axis is not None:
                z_axis = args.z_axis
            else:
                z_axis = None
            if args.channel_axis is not None:
                channel_axis = args.channel_axis
            else:
                channel_axis = None
            if dask_client is not None:
                # ignore bounding boxes
                output_labels, boxes = distributed_eval(
                    args.input,
                    args.input_subpath,
                    image_shape,
                    args.input_timeindex,
                    args.input_channels,
                    args.segmentation_model,
                    process_blocksize,
                    args.working_dir,
                    dask_client,
                    diameter=args.diameter,
                    spatial_ndims=spatial_ndims,
                    do_3D=args.do_3D,
                    blocksoverlap=blocks_overlaps,
                    min_size=args.min_size,
                    max_size_fraction=args.max_size_fraction,
                    niter=args.niter,
                    anisotropy=anisotropy,
                    z_axis=z_axis,
                    channel_axis=channel_axis,
                    normalize=not args.no_norm,
                    normalize_lowhigh=args.norm_lowhigh,
                    normalize_percentile=args.norm_percentile,
                    normalize_norm3D=True,
                    normalize_sharpen_radius=args.normalize_sharpen_radius,
                    normalize_smooth_radius=args.normalize_smooth_radius,
                    normalize_invert=args.normalize_invert,
                    flow_threshold=args.flow_threshold,
                    cellprob_threshold=args.cellprob_threshold,
                    stitch_threshold=args.stitch_threshold,
                    flow3D_smooth=args.flow3D_smooth,
                    label_dist_th=args.label_dist_th,
                    preprocessing_steps=preprocessing_steps,
                    use_gpu=args.use_gpu,
                    gpu_device=args.gpu_device,
                )
                nlabels = len(boxes)
            else:
                output_labels, nlabels = local_eval(
                    args.input,
                    args.input_subpath,
                    args.input_timeindex,
                    args.input_channels,
                    args.segmentation_model,
                    diameter=args.diameter,
                    spatial_ndims=spatial_ndims,
                    do_3D=args.do_3D,
                    min_size=args.min_size,
                    max_size_fraction=args.max_size_fraction,
                    niter=args.niter,
                    anisotropy=anisotropy,
                    z_axis=z_axis,
                    channel_axis=channel_axis,
                    normalize=not args.no_norm,
                    normalize_lowhigh=args.norm_lowhigh,
                    normalize_percentile=args.norm_percentile,
                    normalize_norm3D=True,
                    normalize_sharpen_radius=args.normalize_sharpen_radius,
                    normalize_smooth_radius=args.normalize_smooth_radius,
                    normalize_invert=args.normalize_invert,
                    flow_threshold=args.flow_threshold,
                    cellprob_threshold=args.cellprob_threshold,
                    stitch_threshold=args.stitch_threshold,
                    flow3D_smooth=args.flow3D_smooth,
                    preprocessing_steps=preprocessing_steps,
                    use_gpu=args.use_gpu,
                    gpu_device=args.gpu_device,
                )

            logger.info(f'Finished segmentation process. Found {nlabels-1} labels')

            labels_group_attrs = prepare_parent_group_attrs(
                os.path.basename(args.output),
                output_subpath,
                axes=image_attrs.get('axes'),
                coordinateTransformations=image_attrs.get('coordinateTransformations'),
                image_label_attrs=_create_image_label_attrs(nlabels),
            )

            if args.output_blocksize is not None:
                if len(args.output_blocksize) < output_labels.ndim:
                    # append 0s which later will be replaced
                    # with corresponding output_labels dimension
                    output_blocksize_arg = (args.output_blocksize +
                                            output_labels[0:(output_labels.ndim-len(args.output_blocksize))][::-1])
                else:
                    output_blocksize_arg = args.output_blocksize
                zyx_blocksize = output_blocksize_arg[::-1] # make it zyx
                output_blocks = tuple([d if d > 0 else output_labels.shape[di]
                                    for di,d in enumerate(zyx_blocksize)])
            else:
                # default to output_chunk_size
                output_blocks = (args.output_chunk_size,) * output_labels.ndim

            if len(output_labels.shape) < len(image_shape):
                output_shape = (1,) * (len(image_shape) - len(output_labels.shape)) + output_labels.shape
            else:
                output_shape = output_labels.shape

            if len(output_blocks) < len(image_shape):
                output_blocks = (1,) * (len(image_shape) - len(output_blocks)) + output_blocks

            persisted_labels = write_utils.save(
                args.output, output_subpath,
                output_labels, output_shape,
                blocksize=output_blocks,
                container_attributes=labels_group_attrs,
                pixelResolution=image_attrs.get('pixelResolution'),
                downsamplingFactors=image_attrs.get('downsamplingFactors'),
            )

            if persisted_labels is not None:
                if dask_client is not None:
                    r = dask_client.compute(persisted_labels).result()
                else:
                    r = persisted_labels.compute()
                logger.info(f'DONE ({r})!')
            else:
                logger.warning('No segmentation labels were generated')

            if dask_client is not None:
                dask_client.close()

        except:
            raise


def _print_version_and_exit():
    from cellpose import version_str as cellpose_version

    print(cellpose_version)
    sys.exit(0)


def _create_image_label_attrs(nlabels:int, ome_ngff_version='0.4', source_image_path='..'):
    # set HSV parameters
    s = 1.0
    v = 1.0
    colors = []
    for l in range(nlabels-1):
        h = l/nlabels
        r, g, b = _hsv2rgb(h, s, v)
        colors.append({
            'label-value': l + 1,
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
