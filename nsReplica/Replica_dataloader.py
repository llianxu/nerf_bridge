# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/data/utils/dataloaders.py

import time
import warnings
from typing import Union

import numpy as np
from rich.console import Console
import torch
from torch.utils.data.dataloader import DataLoader


from nsReplica.Replica_dataset import ReplicaDataset, ReplicaDepthDataset


import threading
import concurrent.futures

CONSOLE = Console(width=120)

# Suppress a warning from torch.tensorbuffer about copying that
# does not apply in this case.
warnings.filterwarnings("ignore", "The given buffer")


class ReplicaDataloader(DataLoader):
    """
    Creates batches of the dataset return type. In this case of nerfstudio this means
    that we are returning batches of full images, which then are sampled using a
    PixelSampler. For this class the image batches are progressively growing as
    more images are recieved from ROS, and stored in a pytorch tensor.

    Args:
        dataset: Dataset to sample from.
        publish_posearray: publish a PoseArray to a ROS topic that tracks the poses of the
            images that have been added to the training set.
        data_update_freq: Frequency (wall clock) that images are added to the training
            data tensors. If this value is less than the frequency of the topics to which
            this dataloader subscribes (pose and images) then this subsamples the ROS data.
            Otherwise, if the value is larger than the ROS topic rates then every pair of
            messages is added to the training bag.
        device: Device to perform computation.
    """

    dataset: ReplicaDataset

    def __init__(
        self,
        dataset: ReplicaDataset,
        data_update_freq: float,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        # This is mostly a parameter placeholder, and manages the cameras
        self.dataset = dataset

        # Image meta data
        self.device = device
        self.num_images = len(self.dataset)
        self.H = self.dataset.image_height
        self.W = self.dataset.image_width
        self.n_channels = 3

        # sim the online process
        self.current_idx = 0
        self.updated = True
        self.update_period = 1 / data_update_freq
        self.poselist = []

        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.data_dict = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.indices,
        }

        if isinstance(self.dataset, ReplicaDepthDataset):
            self.data_dict["depth_image"] = self.dataset.depth_tensor

        super().__init__(dataset=dataset, **kwargs)
        
        self.loader_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.loader_thread_pool.submit(self.set_loader_thread())
        
    def set_loader_thread(self):
        self.loader = threading.Timer(self.update_period, self.load_next_image_pose)
        self.loader.daemon = True
        self.loader.start()
        
    def msg_status(self, num_to_start):
        """
        Check if any image-pose pairs have been successfully streamed from
        online, and return True if so.
        """
        # print(self.current_idx)
        return self.current_idx >= (num_to_start - 1)

    def load_next_image_pose(self):
        """
        The callback triggered when time synchronized image and pose messages
        are published to the onlineDataParser.
        """
       
        if self.current_idx < self.num_images :
            # ----------------- Handling the IMAGE ----------------
            self.dataset.updated_indices.append(self.current_idx)
            self.updated = True
            self.current_idx += 1
        
        self.loader_thread_pool.submit(self.set_loader_thread())
     

  

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_updated_batch(self):
        batch = {}
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[:self.current_idx, ...]
        return batch

    def __iter__(self):
        while True:
            if self.updated:
                self.batch = self._get_updated_batch()
                self.updated = False

            batch = self.batch
            yield batch
    
    def __del__(self):
        self.loader.cancel()
        self.loader_thread_pool.shutdown()
        super().__del__()
