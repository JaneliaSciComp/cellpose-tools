import logging
import numpy as np
import yaml
import pydantic.v1.utils as pu
from scipy.ndimage import gaussian_filter


logger = logging.getLogger(__name__)

default_preprocessing_params_str="""
unsharp:
    sigma_one: 1.0
    weight: 0.1
    iterations: 5
    sigma_two: 0.1
"""


def get_preprocessing_steps(steps, preprocessing_config_file, voxel_spacing=None):
    if steps is None or len(steps) == 0:
        return []

    _preprocessing_methods = {
        'unsharp': _unsharp
    }

    preprocessing_steps = []
    logger.info(f'Get preprocessing steps: {steps}')
    preprocessing_config = _get_preprocessing_config(preprocessing_config_file)

    for step in steps:
        logger.info(f'Check preprocessing step: {step}')
        step_method = _preprocessing_methods.get(step)
        if step_method is not None:
            logger.debug(f'Found method {step_method} for step {step}')
            step_params = preprocessing_config.get(step)
            if step_params is not None:
                if voxel_spacing is not None:
                    step_params['voxel_spacing'] = voxel_spacing
                logger.info(f'Add preprocessing step: {step}:{step_params}')
                preprocessing_steps.append((step_method, step_params))

    return preprocessing_steps


def _get_preprocessing_config(preprocessing_config_file):
    default_config = yaml.safe_load(default_preprocessing_params_str)
    if preprocessing_config_file:
        with open(preprocessing_config_file) as f:
            external_config = yaml.safe_load(f)
            logger.info((
                'Read external config from '
                f'{preprocessing_config_file}: {external_config}'
            ))
            config = pu.deep_update(default_config, external_config)
            logger.info(f'Final config {config}')
    else:
        logger.info(f'Use default preprocessing config: {default_config}')
        config = default_config

    return config


def _unsharp(image, sigma_one=1., weight=1., iterations=5, sigma_two=0.1, voxel_spacing=None):
    image = image.astype(np.float32)
    if voxel_spacing is not None:
        s1 = sigma_one / voxel_spacing
        s2 = sigma_two / voxel_spacing
    else:
        s1 = sigma_one
        s2 = sigma_two
    for i in range(iterations):
        high_frequency_image = image - gaussian_filter(image, s1)
        image = (1. - weight) * image + weight * high_frequency_image

    image = gaussian_filter(image, s2)

    image[image < 0] = 0
    image[image > 65500] = 65500

    return np.round(image).astype(np.uint16)
