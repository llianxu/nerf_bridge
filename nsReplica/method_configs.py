
# Code slightly adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/configs/method_configs.py

"""
NerfBridge Method Configs
"""

from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from nsReplica.Replica_datamanager import ReplicaDataManagerConfig, ReplicaDataManager
from nsReplica.Replica_dataparser import ReplicaDataParserConfig
from nsReplica.Replica_trainer import ReplicaTrainerConfig
from nsReplica.Replica_dataset import ReplicaDepthDataset

from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig

SimNerfacto = MethodSpecification(
    config=ReplicaTrainerConfig(
        method_name="sim_nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=1000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ReplicaDataManagerConfig(
                _target=ReplicaDataManager[ReplicaDepthDataset],
                dataparser=ReplicaDataParserConfig(
                    aabb_scale=1.0,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
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
    description="Run NerfBridge with the Nerfacto model, and train with streamed RGB images.",
)

SimDepthNerfacto = MethodSpecification(
    config=ReplicaTrainerConfig(
        method_name="sim_depth_nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=1000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ReplicaDataManagerConfig(
                _target=ReplicaDataManager[ReplicaDepthDataset],
                pixel_sampler=PairPixelSamplerConfig(),
                dataparser=ReplicaDataParserConfig(
                    aabb_scale=1.0,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            model=DepthNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
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
    description="Run NerfBridge with the DepthNerfacto model, and train with streamed RGB and depth images.",
)
