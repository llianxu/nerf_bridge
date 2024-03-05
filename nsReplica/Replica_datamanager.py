"""
A datamanager for the NSOnline Bridge.
"""

from dataclasses import dataclass, field
from typing import Type, Dict, Tuple, Generic, cast, get_origin, get_args
from typing_extensions import TypeVar
from functools import cached_property
from nerfstudio.utils.misc import get_orig_class

from rich.console import Console

from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.rays import RayBundle

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from nsReplica.Replica_dataloader import ReplicaDataloader
from nsReplica.Replica_dataset import ReplicaDataset
from nsReplica.Replica_dataparser import ReplicaDataParserConfig

CONSOLE = Console(width=120)


@dataclass
class ReplicaDataManagerConfig(VanillaDataManagerConfig):
    """A Online datamanager that handles a streaming dataloader."""

    _target: Type = field(default_factory=lambda: ReplicaDataManager)
    dataparser: ReplicaDataParserConfig = ReplicaDataParserConfig()
    data_update_freq: float = 5.0
    """ Frequency, in Hz, that images are added to the training dataset tensor. """
    num_training_images: int = 500
    """ Number of images to train on (for dataset tensor pre-allocation). """

TDataset = TypeVar("TDataset", bound=ReplicaDataset, default=ReplicaDataset)

class ReplicaDataManager(
    VanillaDataManager, Generic[TDataset]
):  # pylint: disable=abstract-method
    """Essentially the VannilaDataManager from Nerfstudio except that the
    typical dataloader for training images is replaced with one that streams
    image and pose data from online.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ReplicaDataManagerConfig
    train_dataset: ReplicaDataset

    def create_train_dataset(self) -> ReplicaDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="train", num_images=self.config.num_training_images
        )
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs, device=self.device
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """
        Returns the dataset type passed as the generic argument. 
        
        NOTE: Hacked from the Vanilla DataManager implementation.
        """
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[ReplicaDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is ReplicaDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is ReplicaDataManager:
            return get_args(orig_class)[0]
        
    def setup_train(self):
        assert self.train_dataset is not None
        self.train_image_dataloader = ReplicaDataloader(
            self.train_dataset,
            self.config.data_update_freq,
            device=self.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
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
        First, checks for updates to the onlineDataloader, and then returns the next
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
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )
