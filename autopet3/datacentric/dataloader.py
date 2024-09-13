import os

import pytorch_lightning as pl
from monai.data import Dataset
from torch.utils.data import DataLoader

from autopet3.datacentric.dataset import RandomPatientDataset
from autopet3.datacentric.transforms import get_transforms
from autopet3.datacentric.utils import get_file_dict_nn, read_split


class AutoPETDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 1,
        target_shape: tuple = None,
        suffix=".nii.gz",
        splits_file: str = None,
        fold: int = 0,
        num_workers_train: int = 2,
        num_workers_val: int = 2,
        data_dir_preprocessed: str = None,

        do_misalign: bool = False,
        max_rotation_sag_cor_ax: list = [ 5, 5, 5 ],
        rad_or_deg: str = "deg",
        prob_rot: float = 0.1,
        max_shiftXYZ: list = [ 2, 2, 1 ],
        prob_shift: float = 0.1,
        do_random_other_transforms: bool = True,
        transforms_name: str = "default",
    ):
        """The AutoPETDataModule class is a PyTorch Lightning DataModule that is responsible for loading and
        preprocessing the data for training, validation, and testing in a PyTorch Lightning pipeline.

        Parameters
        data_dir (str): Directory path where the data is stored.
        batch_size (int): Number of samples in each batch.
        target_shape (tuple): Desired shape of the input data.
        suffix (str): File suffix for the data files.
        splits_file (str): Path to the file containing data splits.
        fold (int): Fold number for cross-validation.
        num_workers_train (int): Number of workers for training data loading.
        num_workers_val (int): Number of workers for validation data loading.
        data_dir_preprocessed (str): Directory path where preprocessed data is stored.

        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.batch_size_val = batch_size
        self.target_shape = tuple(list(target_shape))
        self.fold = fold
        self.splits_file = splits_file
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.suffix = suffix
        self.data_dir_preprocessed = data_dir_preprocessed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.do_misalign = do_misalign
        self.max_rotation_sag_cor_ax = max_rotation_sag_cor_ax
        self.rad_or_deg = rad_or_deg
        self.prob_rot = prob_rot
        self.max_shiftXYZ = max_shiftXYZ
        self.prob_shift = prob_shift

        self.do_random_other_transforms = do_random_other_transforms
        self.transforms_name = transforms_name

    def setup(self, stage: str):
        """Setup method to prepare datasets for training, validation, and testing.
        Args:
            stage (str): Indicates the stage of the setup process, e.g., 'fit', 'test'.

        """
        # Read the split information from the splits file
        if self.fold == "all":
            split = read_split(self.splits_file, 0)
            split["train"] += split["val"]
        else: 
            split = read_split(self.splits_file, self.fold)

        # Dataloader for preprocessed augmented data of example 2
        if self.suffix == ".npz":
            val_files = get_file_dict_nn(self.data_dir, split["val"], suffix=".nii.gz")
            self.train_dataset = RandomPatientDataset(os.path.join(self.data_dir_preprocessed, "train"), split["train"])
            self.val_dataset = RandomPatientDataset(os.path.join(self.data_dir_preprocessed, "val"), split["val"])
            self.batch_size_val = 1
        # Dataloader for example 1
        else:
            train_files = get_file_dict_nn(self.data_dir, split["train"], suffix=self.suffix)
            val_files = get_file_dict_nn(self.data_dir, split["val"], suffix=self.suffix)
            self.train_dataset = Dataset(
                train_files, transform=get_transforms("train", self.target_shape, resample=True,
                                                      do_misalign=self.do_misalign,
                                                      max_rotation_sag_cor_ax=self.max_rotation_sag_cor_ax, rad_or_deg=self.rad_or_deg, prob_rot=self.prob_rot,
                                                      max_shiftXYZ=self.max_shiftXYZ, prob_shift=self.prob_shift,
                                                    do_random_other_transforms=self.do_random_other_transforms, transforms_name=self.transforms_name)
            )
            self.val_dataset = Dataset(
                val_files, transform=get_transforms("val_sampled", self.target_shape, resample=True)
            )
        self.test_dataset = Dataset(val_files, transform=get_transforms("val", self.target_shape, resample=True))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers_train, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_val, num_workers=self.num_workers_val)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers_val)

    def predict_dataloader(self):
        pass

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
