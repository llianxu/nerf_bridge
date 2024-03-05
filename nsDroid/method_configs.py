# Code slightly adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/configs/method_configs.py

"""
Method configurations
"""

from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from nsDroid.droid_datamanager import DroidDataManagerConfig
from nsDroid.droid_dataparser import DroidDataParserConfig
from nsDroid.droid_trainer import DroidTrainerConfig

from nerfstudio.plugins.types import MethodSpecification

DroidNerfacto = MethodSpecification( 
    DroidTrainerConfig(
        method_name="droid_nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=30000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=DroidDataManagerConfig(
                dataparser=DroidDataParserConfig(
                    aabb_scale = 8.0,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            # origin 
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
            # model=NerfactoModelConfig(),
            # model=InstantNGPModelConfig(),
            
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=20000),
        vis="viewer",
    ),
    description="Run the nerfstudio nerfacto method on droid data streamed online."
)
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
