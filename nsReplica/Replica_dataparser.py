"""Data parser for loading online parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

import os

@dataclass
class ReplicaDataParserConfig(DataParserConfig):
    """ROS config file parser config."""

    _target: Type = field(default_factory=lambda: ReplicaDataParser)
    """target class to instantiate"""
    data: Path = Path("data/replica_room0")
    """ Path to configuration JSON. """
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    aabb_scale: float = 2.0
    """ SceneBox aabb scale."""


@dataclass
class ReplicaDataParser(DataParser):
    """online DataParser"""

    config: ReplicaDataParserConfig

    def __init__(self, config: ReplicaDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.aabb = config.aabb_scale

    def get_dataparser_outputs(self, split="train", num_images: int = 500):
        dataparser_outputs = self._generate_dataparser_outputs(split, num_images)
        return dataparser_outputs

    def _generate_dataparser_outputs(self, split="train", num_images: int = 500):
        """
        This function generates a DataParserOutputs object. Typically in Nerfstudio
        this is used to populate the training and evaluation datasets, but since with
        NSROS Bridge our aim is to stream the data then we only have to worry about
        loading the proper camera parameters and image-pose pair.

        Args:
            split: Determines the data split (not used, but left in place for consistency
                with Nerfstudio)

            num_images: The size limit of the training image dataset. This is used to
                pre-allocate tensors for the Cameras object that tracks camera pose.
        """
        meta = load_from_json(self.data / "transforms.json")

        image_height = meta["h"]
        image_width = meta["w"]
        fx = meta["fl_x"]
        fy = meta["fl_y"]
        cx = meta["cx"]
        cy = meta["cy"]

        k1 = meta["k1"] if "k1" in meta else 0.0
        k2 = meta["k2"] if "k2" in meta else 0.0
        k3 = meta["k3"] if "k3" in meta else 0.0
        k4 = meta["k4"] if "k4" in meta else 0.0
        p1 = meta["p1"] if "p1" in meta else 0.0
        p2 = meta["p2"] if "p2" in meta else 0.0
        distort = torch.tensor([k1, k2, k3, k4, p1, p2], dtype=torch.float32)

        camera_to_world = []
        image_filenames = []
        depth_filenames = []
        for i, frame in enumerate(meta["frames"]):
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            camera_to_world.append(c2w)
            image_filenames.append(frame["image_file_path"])
            if "depth_file_path" in frame:
                depth_filenames.append(frame["depth_file_path"])
                
            
        camera_to_world = torch.stack(camera_to_world, dim=0)[
            :, :-1, :
        ]

        # in x,y,z order
        scene_size = self.aabb
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-scene_size, -scene_size, -scene_size],
                    [scene_size, scene_size, scene_size],
                ],
                dtype=torch.float32,
            )
        )

        num_images = len(meta["frames"]) if len(meta["frames"]) < num_images else num_images
        # Create a dummy Cameras object with the appropriate number
        # of placeholders for poses.
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=image_height,
            width=image_width,
            distortion_params=distort,
            camera_type=CameraType.PERSPECTIVE,
        )

    
        metadata = {
            "num_images": num_images,
            "image_height": image_height,
            "image_width": image_width,
            "data_dir": self.data,
        }

        if depth_filenames != []:
            metadata["depth_filenames"] = depth_filenames
            metadata["depth_scale_factor"] = meta["depth_scale_factor"]
            
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,  # This is empty
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs

    