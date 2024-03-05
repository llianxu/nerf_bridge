# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/data/utils/dataloaders.py

"""
Defines the droidDataloader object that subscribes an image tensor and Cameras object with values from droid-slam.
Image and pose pairs are added at a prescribed frequency and intermediary images
are discarded (could be used for evaluation down the line).
"""
import time
import warnings
from typing import Union

import numpy as np
import scipy.spatial.transform as transform
from rich.console import Console
import torch
from torch.utils.data.dataloader import DataLoader

from nerfstudio.process_data.colmap_utils import qvec2rotmat
import nerfstudio.utils.poses as pose_utils

from nsDroid.droid_dataset import DroidDataset

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray
from message_filters import TimeSynchronizer, Subscriber

from flask import request, jsonify
from gevent import pywsgi

import os
import cv2

from threading import Thread


CONSOLE = Console(width=120)

# Suppress a warning from torch.tensorbuffer about copying that
# does not apply in this case.
warnings.filterwarnings("ignore", "The given buffer")


    

def droid_pose_to_nerfstudio(quat_w2c):
    """
    Takes a droid Pose message(quaternion) and converts it to the
    3x4 transform format used by nerfstudio.
    """
    trans = quat_w2c[:3]
    quat =  quat_w2c[3:]
    rot = transform.Rotation.from_quat(quat).as_matrix()
    matrix_w2c = np.eye(4, dtype=np.float32)
    matrix_w2c[:3, :3] = rot
    matrix_w2c[:3, 3] = trans
    matrix_c2w = np.ascontiguousarray(np.linalg.inv(matrix_w2c)[:3, :4])
    ### opencv2bleander
    matrix_c2w[:, 1] *= -1
    matrix_c2w[:, 2] *= -1
    return matrix_c2w


class DroidDataloader(DataLoader):
    """
    Creates batches of the dataset return type. In this case of nerfstudio this means
    that we are returning batches of full images, which then are sampled using a
    PixelSampler. For this class the image batches are progressively growing as
    more images are recieved from droid, and stored in a pytorch tensor.

    Args:
        dataset: Dataset to sample from.
       
        device: Device to perform computation.
    """

    dataset: DroidDataset

    def __init__(
        self,
        dataset: DroidDataset,
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

        self.exp_path = self.dataset.exp_path
        
        self.current_idx = 0
        


        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.data_dict = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.image_indices,
        }

        super().__init__(dataset=dataset, **kwargs)


    def msg_status(self, num_to_start):
        """
        Check if any image-pose pairs have been successfully streamed from
        droid, and return True if so.
        """
        return self.current_idx >= (num_to_start - 1)

    def get_droid_data(self):
        """
        The callback triggered when time synchronized image and pose messages
        are published on the topics specifed in the config JSON passed to
        the droidDataParser.
        """
        
        if request.method == 'POST' and self.current_idx < self.num_images:
       
            # ----------------- Handling the IMAGE AND POSE ----------------
            received_data = request.get_json()
            quat_w2c = np.array(received_data['pose'])
            rgb_img = np.array(received_data['rgb'], dtype=np.float32) / 255.0
            droid_finished = received_data['droid_finished']
            if len(rgb_img.shape) == 4:
                rgb_img = rgb_img.transpose(0, 2, 3, 1)[..., ::-1]
            else:
                rgb_img = rgb_img.transpose(1, 2, 0)[..., ::-1]
            if droid_finished == "False":
                if self.current_idx == 0:
                    os.makedirs(f'{self.exp_path}/transfer_data', exist_ok=True)   
                    for i in range(8):
                        rgb_img_np = np.ascontiguousarray(rgb_img[self.current_idx])
                        c2w_np = droid_pose_to_nerfstudio(quat_w2c[self.current_idx])
                        rgb_img_item = cv2.cvtColor(rgb_img_np, cv2.COLOR_RGB2BGR)
                        np.savetxt(f'{self.exp_path}/transfer_data/{self.current_idx}_pose.txt', c2w_np)
                        cv2.imwrite(f"{self.exp_path}/transfer_data/{self.current_idx}_main.png", (rgb_img_item * 255).astype(np.uint8))
                        device = self.dataset.cameras.device
                        self.dataset.cameras.camera_to_worlds[self.current_idx] = torch.from_numpy(c2w_np).to(device)
                        self.dataset.image_tensor[self.current_idx] = torch.from_numpy(rgb_img_np).to(device)
                        self.dataset.updated_indices.append(self.current_idx)
                        self.current_idx += 1
                else:
                    rgb_img_np = np.ascontiguousarray(rgb_img)
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    c2w_np = droid_pose_to_nerfstudio(quat_w2c)
                    device = self.dataset.cameras.device
                    self.dataset.cameras.camera_to_worlds[self.current_idx] = torch.from_numpy(c2w_np).to(device)
                    self.dataset.image_tensor[self.current_idx] = torch.from_numpy(rgb_img_np).to(device)
                    self.dataset.updated_indices.append(self.current_idx)               
                    np.savetxt(f'{self.exp_path}/transfer_data/{self.current_idx}_pose.txt', c2w_np)
                    cv2.imwrite(f"{self.exp_path}/transfer_data/{self.current_idx}_main.png", (rgb_img*255).astype(np.uint8))
                    self.current_idx += 1
            
            else:
                for i in range(self.current_idx):
                    rgb_img_np = np.ascontiguousarray(rgb_img[i])
                    c2w_np = droid_pose_to_nerfstudio(quat_w2c[i])
                    rgb_img_item = cv2.cvtColor(rgb_img_np, cv2.COLOR_RGB2BGR)
                    os.makedirs(f'{self.exp_path}/global_ba_data', exist_ok=True)   
                    np.savetxt(f'{self.exp_path}/global_ba_data/{i}_pose.txt', c2w_np)
                    cv2.imwrite(f"{self.exp_path}/global_ba_data/{i}_main.png", (rgb_img_item * 255).astype(np.uint8))
                    device = self.dataset.cameras.device
                    self.dataset.cameras.camera_to_worlds[i] = torch.from_numpy(c2w_np).to(device)
                    self.dataset.image_tensor[i] = torch.from_numpy(rgb_img_np).to(device)
            self.updated = True
            return jsonify({'status': 'transfer success'})
           

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_updated_batch(self):
        batch = {}
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[: self.current_idx, ...]
        return batch

    def __iter__(self):
        while True:
            if self.updated:
                self.batch = self._get_updated_batch()
                self.updated = False

            batch = self.batch
            yield batch
