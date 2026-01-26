import os
import sys
import traceback

from cellpose import version_str as cellpose_version
from cellpose.cli import get_arg_parser


def _define_args():
    args_parser = get_arg_parser()


    args_parser.add_argument('--models-dir', dest='models_dir',
                             type=str,
                             help='cache cellpose models directory')
    args_parser.add_argument('--model', dest='segmentation_model',
                             type=str,
                             default='cpsam',
                             help='segmentation model')
    return args_parser


def download_cellpose_models(models_dir, model_name):

    if models_dir is not None:
        os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = models_dir

    from cellpose.models import get_user_models, model_path

    try:
        print('Cache cellpose models', model_name, flush=True)
        get_user_models()
        model_path(model_name)
    except:
        raise


def _print_version_and_exit():
    print(cellpose_version)
    sys.exit(0)


if __name__ == '__main__':
    args_parser = _define_args()
    args = args_parser.parse_args()
    print('Get cellpose models:', args, flush=True)
    try:
        if args.version:
            _print_version_and_exit()

        download_cellpose_models(args.models_dir, args.segmentation_model)
        sys.exit(0)
    except Exception as err:
        print('Cellpose labeling error:', err)
        traceback.print_exception(err)
        sys.exit(1)
