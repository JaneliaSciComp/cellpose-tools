import argparse
import logging
import numpy as np
import os
import sys
import traceback

from dask.distributed import Client, LocalCluster

from io_utils.write_utils import container_type, write_zarray_as
from io_utils.read_utils import read_array_attrs

from segmentation.distributed_cellpose import merge_labels

from utils.configure_logging import configure_logging
from utils.configure_dask import load_dask_config, ConfigureWorkerPlugin

from zarr_tools.io.zarr_io import create_zarr_array
from zarr_tools.ngff.ngff_utils import get_non_spatial_axes


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
                             required=True,
                             help='Destination zarr path for merged labels')
    args_parser.add_argument('--output-subpath', '--output_subpath',
                             dest='output_subpath',
                             type=str,
                             help='Dataset subpath inside the output zarr/N5 container')
    args_parser.add_argument('--working-dir', '--working_dir',
                             dest='working_dir',
                             type=str,
                             required=True,
                             help='Directory containing the saved block metadata files '
                                  '(label-block-indices.npy, block-faces.npy, '
                                  'label-boxes.npy, label-boxes-ids.npy)')
    args_parser.add_argument('--label-distance-threshold', '--label-dist-th',
                             dest='label_dist_th',
                             type=float,
                             default=1.0,
                             help='Label distance transform threshold used for merging labels')
    args_parser.add_argument('--output-chunk-size', '--output_chunk_size',
                             dest='output_chunk_size',
                             default=128,
                             type=int,
                             help='Output chunk size')
    args_parser.add_argument('--compressor', '--compression',
                             dest='compressor',
                             default='zstd',
                             help='Zarr array compression algorithm')
    args_parser.add_argument('--compressor-opts', '--compression-opts',
                             dest='compressor_opts',
                             type=_dictfromjson,
                             default={},
                             help='Zarr array compression options')
    args_parser.add_argument('--dask-scheduler', '--dask_scheduler',
                             dest='dask_scheduler',
                             type=str,
                             default=None,
                             help='TCP/IP address of an existing Dask scheduler')
    args_parser.add_argument('--dask-config', '--dask_config',
                             dest='dask_config',
                             type=str,
                             default=None,
                             help='Dask configuration yaml file')
    args_parser.add_argument('--local-dask-workers', '--local_dask_workers',
                             dest='local_dask_workers',
                             type=int,
                             default=0,
                             help='Number of workers when using a local cluster')
    args_parser.add_argument('--worker-cpus', '--worker_cpus',
                             dest='worker_cpus',
                             type=int,
                             default=1,
                             help='Number of CPUs allocated to a dask worker')
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

    working_dir = args.working_dir
    logger.info(f'Loading block metadata from {working_dir}')
    label_block_indices = np.load(f'{working_dir}/label-block-indices.npy', allow_pickle=True)
    faces = np.load(f'{working_dir}/block-faces.npy', allow_pickle=True)
    boxes = list(np.load(f'{working_dir}/label-boxes.npy', allow_pickle=True))
    all_box_ids = np.load(f'{working_dir}/label-boxes-ids.npy')

    input_image_attrs = read_array_attrs(args.input, args.input_subpath)
    input_image_shape = input_image_attrs['array_shape']
    image_ndim = input_image_attrs['array_ndim']

    non_spatial_axes = get_non_spatial_axes(input_image_attrs).values()
    labels_shape = tuple(input_image_shape[di]
                         if di not in non_spatial_axes else 1
                         for di in range(len(input_image_shape)))

    output_subpath = args.output_subpath if args.output_subpath else args.input_subpath

    if non_spatial_axes:
        output_blocksize = (1,) * len(non_spatial_axes) + (args.output_chunk_size,) * (image_ndim - len(non_spatial_axes))
    else:
        output_blocksize = (args.output_chunk_size,) * image_ndim

    logger.info((
        f'Create output labels zarr {args.output}:{output_subpath} '
        f'shape: {labels_shape}, chunksize: {output_blocksize}'
    ))
    labels_zarr = create_zarr_array(
        args.output,
        output_subpath,
        labels_shape,
        output_blocksize,
        'uint32',
        compressor=args.compressor,
    )

    import zarr as _zarr
    input_labels_zarr = _zarr.open(args.input, mode='r', path=args.input_subpath)

    logger.info('Copy unmerged labels to output zarr before relabeling')
    labels_zarr[:] = input_labels_zarr[:]

    logger.info('Start label merge process')
    _, merged_boxes = merge_labels(
        label_block_indices, faces, boxes, all_box_ids,
        labels_zarr, dask_client, working_dir, args.label_dist_th,
    )
    logger.info(f'Merge complete. Found {len(merged_boxes)} labels')

    output_container_type = container_type(args.output)
    if output_container_type != 'zarr':
        logger.info(f'Save output labels as {output_container_type} at {args.output}')
        write_zarray_as(labels_zarr, args.output, output_subpath)

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
