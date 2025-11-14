import os
import yaml

from dask.distributed import Worker
from distributed.diagnostics.plugin import WorkerPlugin

from flatten_json import flatten

from .configure_logging import configure_logging


class ConfigureWorkerPlugin(WorkerPlugin):
    def __init__(self, models_dir, logging_config, verbose,
                 worker_cpus=0):
        self.models_dir = models_dir
        self.logging_config = logging_config
        self.verbose = verbose
        self.worker_cpus = worker_cpus
        self.logger = None

    def setup(self, worker: Worker):
        self.logger = configure_logging(self.logging_config, self.verbose,
                                        logger_name='dask_worker')
        _set_cpu_resources(self.worker_cpus)
        if self.models_dir:
            self.logger.info(f'Set cellpose models path: {self.models_dir}')
            os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = self.models_dir

    def teardown(self, worker: Worker):
        pass

    def transition(self, key: str, start: str, finish: str, **kwargs):
        pass

    def release_key(self, key: str, state: str, cause: str | None, reason: None, report: bool):
        pass


def load_dask_config(config_file):
    if (config_file):
        import dask.config

        print(f'Use dask config: {config_file}', flush=True)
        
        with open(config_file) as f:
            dask_config = flatten(yaml.safe_load(f))
            dask.config.set(dask_config)


def _set_cpu_resources(cpus:int):
    if cpus:
        os.environ['MKL_NUM_THREADS'] = str(cpus)
        os.environ['NUM_MKL_THREADS'] = str(cpus)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpus)
        os.environ['OPENMP_NUM_THREADS'] = str(cpus)
        os.environ['OMP_NUM_THREADS'] = str(cpus)

    return cpus
