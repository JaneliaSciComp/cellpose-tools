import argparse
import logging
from math import ceil
import os
import sys
import traceback

from dask.distributed import Client, LocalCluster

from io_utils.read_utils import open_array, read_array_attrs

from pathlib import Path

from segmentation.distributed_cellpose import distributed_merge

from utils.configure_logging import configure_logging
from utils.configure_dask import load_dask_config, ConfigureWorkerPlugin

from zarr_tools.io.zarr_io import create_zarr_array, derive_shard_shape
from zarr_tools.ngff.ngff_utils import create_ome_metadata, get_non_spatial_axes

from .cli import dictfromjson, inttuple

logger: logging.Logger


def _define_args():
    args_parser = argparse.ArgumentParser(
        description=(
            'Merge labels across blocks for a segmentation zarr that was produced '
            'with --skip-merge-labels. Reads the saved block metadata files and '
            'writes the merged result to a destination zarr.'
        )
    )
    args_parser.add_argument('-i', '--input',
                             dest='input',
                             type=str,
                             required=True,
                             help='Input segmentation zarr with unmerged block labels')
    args_parser.add_argument('--input-subpath', '--input_subpath',
                             dest='input_subpath',
                             type=str,
                             help='Dataset subpath inside the input zarr/N5 container')
    args_parser.add_argument('-o', '--output',
                             dest='output',
                             type=str,
                             help='Destination zarr path for merged labels')
    args_parser.add_argument('--output-subpath', '--output_subpath',
                             dest='output_subpath',
                             type=str,
                             help='Dataset subpath inside the output zarr/N5 container')
    args_parser.add_argument('--compressor', '--compression',
                             dest='compressor',
                             help='Zarr array compression algorithm')
    args_parser.add_argument('--compressor-opts', '--compression-opts',
                             dest='compressor_opts',
                             type=dictfromjson,
                             default={},
                             help='Zarr array compression options')
    args_parser.add_argument('--zarr-format', '--zarr_format',
                             type=int,
                             default=2,
                             dest='zarr_format',
                             help='Zarr format (2 or 3 for v2 or v3)')
    args_parser.add_argument('--output-blocksize', '--output_blocksize',
                             dest='output_blocksize',
                             type=inttuple,
                             metavar='X,Y,Z',
                             help='Output chunk size as a tuple (x,y,z).')
    args_parser.add_argument('--sharding-factor', '--sharding_factor',
                             dest='sharding_factor',
                             type=inttuple,
                             default=(),
                             metavar='SX,SY,SZ',
                             help='Sharding factor per spatial axis (zarr v3 only). '
                                  'Shard size = input_chunksize * factor.')

    args_parser.add_argument('--working-dir', '--working_dir',
                             dest='working_dir',
                             type=str,
                             help='Directory for saving label re-assignment')
    args_parser.add_argument('--process-blocksize', '--process_blocksize',
                             dest='process_blocksize',
                             type=inttuple,
                             help='Output chunk size as a tuple (x,y,z).')
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
                             help='Volume ROI descriptor a tuple of 6 values representing min and max voxel coordinates')
    args_parser.add_argument('--label-distance-threshold', '--label-dist-th',
                             dest='label_dist_th',
                             type=float,
                             default=1.0,
                             help='Label distance transform threshold used for merging labels')

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

    args_parser.add_argument('--logging-config',
                             dest='logging_config',
                             type=str,
                             help='Python log file configuration')
    args_parser.add_argument('-v', '--verbose',
                             dest='verbose',
                             action='store_true',
                             default=False,
                             help='Verbose logging')
    return args_parser


def _run_merge(args):
    load_dask_config(args.dask_config)

    if args.dask_scheduler:
        logger.info(f'Create dask client for {args.dask_scheduler}')
        dask_client = Client(address=args.dask_scheduler)
    elif args.local_dask_workers > 0:
        logger.info(f'Create local dask client with {args.local_dask_workers} local workers')
        dask_client = Client(LocalCluster(n_workers=args.local_dask_workers,
                                          threads_per_worker=args.worker_cpus))
    else:
        logger.info('Create default local dask client')
        dask_client = Client(LocalCluster())

    if dask_client is not None:
        worker_config = ConfigureWorkerPlugin(None, args.logging_config, args.verbose,
                                              worker_cpus=args.worker_cpus)
        dask_client.register_plugin(worker_config, name='WorkerConfig')

    input_labels_attrs = read_array_attrs(args.input, args.input_subpath)
    logger.info(f'Labels image {args.input}:{args.input_subpath} attributes: {input_labels_attrs}')

    input_labels_shape = input_labels_attrs['array_shape']

    non_spatial_axes = get_non_spatial_axes(input_labels_attrs).values()
    labels_shape = tuple(input_labels_shape[di]
                         if di not in non_spatial_axes else 1
                         for di in range(len(input_labels_shape)))
    output_path = args.output if args.output else args.input
    output_subpath = args.output_subpath if args.output_subpath else args.input_subpath

    input_labels_array = open_array(input_labels_attrs['array_storepath'], input_labels_attrs['array_subpath'])
    labels_ndim = input_labels_attrs['array_ndim']
    shard_shape = None
    logger.info((
        f'Create output labels zarr {args.output}:{output_subpath} '
        f'ndim: {labels_ndim}, shape: {labels_shape} '
    ))
    if output_path == args.input and output_subpath == args.input_subpath:
        logger.info(f'Merged labels will overwrite {output_path}:{output_subpath}')
        output_labels_array = input_labels_array
        output_chunksize = input_labels_array.chunks
        logger.debug(f'Use same chunk size {output_chunksize} for merged labels output')
        try:
            shard_shape = input_labels_array.shards
            logger.debug(f'Use shard shape: {shard_shape}')
        except (AttributeError, NotImplementedError):
            shard_shape = None
            logger.debug(f'No sharding was used for {args.input}:{args.input_subpath}')
    else:
        if args.output_blocksize is not None:
            # the output chunksize can only be overwritten if it creates new file
            if len(args.output_blocksize) < len(input_labels_array.chunks):
                output_chunksize = args.output_blocksize[::-1]
            else:
                output_chunksize = input_labels_array.chunks[0:len(input_labels_array.chunks)-len(args.output_blocksize)] + args.output_blocksize[::-1]
        else:
            output_chunksize = input_labels_array.chunks
        logger.debug(f'Use {output_chunksize} for merged labels output')
        input_labels_transforms = input_labels_attrs.get('array_transforms', {})
        logger.debug(f'Input labels transforms: {input_labels_transforms}')
        ome_version = '0.4' if args.zarr_format == 2 else '0.5'
        logger.info(f'Create OME {ome_version}')
        ome_metadata = create_ome_metadata(
            os.path.basename(output_path),
            output_subpath,
            input_labels_attrs.get('array_axes'),
            input_labels_transforms.get('scale'),
            input_labels_transforms.get('translation'),
            input_labels_attrs.get('array_ndim'),
            ome_version=ome_version
        )
        if args.sharding_factor:
            if len(args.sharding_factor) < len(output_chunksize):
                sharding_factor = (args.sharding_factor + (1,) * (len(output_chunksize) - len(args.sharding_factor)))[::-1]
            else:
                # just like for the blocksize we already have the correct number of dimensions
                # just reverse the sharding factor because in the command line we specify it as x,y,z[,c,t]
                sharding_factor = args.sharding_factor[::-1]
        else:
            sharding_factor = None

        logger.info(f'Output blocksize: {output_chunksize}, sharding factor: {sharding_factor}')
        shard_shape = derive_shard_shape(
            output_chunksize,
            args.zarr_format,
            sharding_factor,
        )

        logger.info((
            f'Create labels zarr {output_path}:{output_subpath} '
            f'zarr format: {args.zarr_format}, shape: {labels_shape}, chunksize: {output_chunksize} '
            f'shard_shape: {shard_shape} '
            f'compressor: {args.compressor} '
        ))

        output_labels_array = create_zarr_array(
            output_path,
            output_subpath,
            labels_shape,
            output_chunksize,
            input_labels_array.dtype,
            compressor=args.compressor,
            compression_opts=args.compressor_opts,
            parent_array_attrs=ome_metadata,
            zarr_format=args.zarr_format,
            shard_shape=shard_shape,
        )
    if shard_shape is not None:
        spatial_shard = tuple(int(x) for x in shard_shape[-3:])
        if args.process_blocksize is not None:
            pb = tuple(int(x) for x in args.process_blocksize[::-1])
            if all(p % s == 0 for p, s in zip(pb, spatial_shard)):
                process_blocksize = pb
            else:
                process_blocksize = tuple(int(ceil(p / s)) * s for p, s in zip(pb, spatial_shard))
                logger.warning(
                    f'process_blocksize {args.process_blocksize[::-1]} is not an exact multiple of '
                    f'shard_shape {spatial_shard}, rounding up to {process_blocksize}'
                )
        else:
            process_blocksize = spatial_shard
    else:
        if args.process_blocksize is not None:
            process_blocksize = args.process_blocksize[::-1]
        else:
            process_blocksize = output_chunksize

    if args.mask and Path(args.mask).exists():
        # read the mask
        logger.info(f'Read foreground mask from {args.mask}:{args.mask_subpath}')
        mask = open_array(args.mask, args.mask_subpath)
    else:
        logger.info('No foreground mask')
        mask = None

    # call distributed label merge
    logger.info(f'Merge labels using process size: {process_blocksize}')
    _, boxes = distributed_merge(
        input_labels_array,
        process_blocksize,
        output_labels_array,
        args.working_dir,
        dask_client,
        mask=mask,
        roi=args.roi,
        label_dist_th=args.label_dist_th,
    )
    nlabels = len(boxes)
    logger.info(f'Finished labels merge process. Found {nlabels-1} labels')

    dask_client.close()


def _main():
    args_parser = _define_args()
    args = args_parser.parse_args()

    try:
        global logger
        logger = configure_logging(args.logging_config, args.verbose)
        logger.info(f'Invoked merge labels with: {args}')
    except Exception as err:
        print('Logging configuration error:', err)
        traceback.print_exception(err)
        sys.exit(1)

    _run_merge(args)


if __name__ == '__main__':
    _main()
