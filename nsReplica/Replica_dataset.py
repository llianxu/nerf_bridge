from typing import Union
import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from PIL import Image
import cv2
import os
import numpy as np

class ReplicaDataset(InputDataset):
    """
    This is a tensor dataset that keeps track of all of the data streamed by online.
    It's main purpose is to conform to the already defined workflow of nerfstudio:
        (dataparser -> inputdataset -> dataloader).

    In reality we could just store everything directly in onlineDataloader, but this
    would require rewritting more code than its worth.

    Images are tracked in self.image_tensor with uninitialized images set to
    all white (hence torch.ones).
    Poses are stored in self.cameras.camera_to_worlds as 3x4 transformation tensors.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert "num_images" in dataparser_outputs.metadata.keys()
        self.num_images = self.metadata["num_images"]
        assert self.num_images > 0
        self.image_height = self.metadata["image_height"]
        self.image_width = self.metadata["image_width"]
        self.data_dir = self.metadata["data_dir"]
        
        self.device = device

        self.cameras = self.cameras.to(device=self.device)

        self.image_tensor = []
        for image_filename in self.image_filenames:
            current_image_filename = self.data_dir / image_filename
            image = Image.open(current_image_filename)
            # COPY the image data into the data tensor
            self.image_tensor.append(torch.from_numpy(np.array(image, dtype=np.float32) / 255.0))
        self.image_tensor = torch.stack(self.image_tensor, dim=0)
        self.indices = torch.arange(self.num_images)

        self.updated_indices = []

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: int):
        """
        This returns the data as a dictionary which is not actually how it is
        accessed in the dataloader, but we allow this as well so that we do not
        have to rewrite the several downstream functions.
        """
        data = {"image_idx": idx, "image": self.image_tensor[idx]}
        return data


class ReplicaDepthDataset(ReplicaDataset):
    """
    This is a tensor dataset that keeps track of all of the data streamed by ROS.
    It's main purpose is to conform to the already defined workflow of nerfstudio:
        (dataparser -> inputdataset -> dataloader).

    This is effectively the same as the regular ROSDataset, but includes additional
    tensors for depth.

    Images are tracked in self.image_tensor with uninitialized images set to
    all white (hence torch.ones). Similarly, we store depth data in self.depth_tensor.
    Poses are stored in self.cameras.camera_to_worlds as 3x4 transformation tensors.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__(dataparser_outputs, scale_factor, device)
        assert "depth_filenames" in dataparser_outputs.metadata.keys()
        assert "depth_scale_factor" in dataparser_outputs.metadata.keys()

        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_tensor = []
        for depth_filename in self.depth_filenames:
            current_depth_filename = self.data_dir / depth_filename
            depth = cv2.imread(str(current_depth_filename), cv2.IMREAD_UNCHANGED)
            self.depth_tensor.append(torch.from_numpy(np.array(depth, dtype=np.float32))/ self.metadata["depth_scale_factor"])
        self.depth_tensor = torch.stack(self.depth_tensor, dim=0)

    def __getitem__(self, idx: int):
        """
        This returns the data as a dictionary which is not actually how it is
        accessed in the dataloader, but we allow this as well so that we do not
        have to rewrite the several downstream functions.
        """

        data = {"depth_idx": idx, "image": self.image_tensor[idx], "depth": self.depth_tensor[idx]}
        return data