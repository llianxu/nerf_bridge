"""
A datamanager for the NSdroid Bridge.
"""

from dataclasses import dataclass, field
from typing import Type, Dict, Tuple

from rich.console import Console

from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.rays import RayBundle

from nsDroid.droid_dataset import DroidDataset
from nsDroid.droid_dataloader import DroidDataloader
from nsDroid.droid_dataparser import DroidDataParserConfig


CONSOLE = Console(width=120)

from flask import Flask
from gevent import pywsgi

from threading import Thread

nerf_server = Flask(__name__)
def run_server():
    # must use absolute ip address not localhost
    server = pywsgi.WSGIServer(('127.0.0.1', 7400), nerf_server)
    server.serve_forever()
    
    
@dataclass
class DroidDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A droid datamanager that handles a streaming dataloader."""

    _target: Type = field(default_factory=lambda: DroidDataManager)
    dataparser: DroidDataParserConfig = DroidDataParserConfig()
    """ Must use only the droidDataParser here """
    publish_training_posearray: bool = True
    """ Whether the dataloader should publish an pose array of the training image poses. """
    data_update_freq: float = 5.0
    """ Frequency, in Hz, that images are added to the training dataset tensor. """
    num_training_images: int = 500
    """ Number of images to train on (for dataset tensor pre-allocation). """


class DroidDataManager(
    base_datamanager.VanillaDataManager
):  # pylint: disable=abstract-method
    """Essentially the VannilaDataManager from Nerfstudio except that the
    typical dataloader for training images is replaced with one that streams
    image and pose data from droid.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: DroidDataManagerConfig
    train_dataset: DroidDataset

    def create_train_dataset(self) -> DroidDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="train", num_images=self.config.num_training_images
        )
        return DroidDataset(
            dataparser_outputs=self.train_dataparser_outputs, device=self.device
        )

    def setup_train(self):
        assert self.train_dataset is not None
        self.train_image_dataloader = DroidDataloader(
            self.train_dataset,
            device=self.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        
        nerf_server.add_url_rule('/get_droid_data/', methods=['POST'], view_func=self.train_image_dataloader.get_droid_data)
        self.online_thread = Thread(target=run_server, daemon=True)
        self.online_thread.start()
        
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(
            self.train_dataset, self.config.train_num_rays_per_batch
        )
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras,
            self.train_camera_optimizer,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """
        First, checks for updates to the droidDataloader, and then returns the next
        batch of data from the train dataloader.
        """
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def setup_eval(self):
        """
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        pass

    def create_eval_dataset(self):
        """
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        pass

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with droid Streaming."
        )

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with droid Streaming."
        )
