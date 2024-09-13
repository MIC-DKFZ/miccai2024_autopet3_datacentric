import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from autopet3.datacentric.utils import read_split
from autopet3.fixed.utils import plot_ct_pet_label


class RandomPatientDataset(Dataset):
    def __init__(self, data_dir: str, split: str, transform: Optional[Callable] = None):
        """Dataset class for example 2. Returns always patients per epoch with random augmentation sampled.
        Args:
            data_dir (str): The directory where the data is stored.
            split (str): The split of the data (e.g., 'train', 'test').
            transform (Optional[Callable]): A function that transforms the data.
        Returns:
            None

        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_number = self._get_max_number()

    def _get_max_number(self):
        unique, counts = np.unique(["_".join(i.split("_")[:-1]) for i in os.listdir(self.data_dir)], return_counts=True)
        return counts[0] - 1

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        patient_id = self.split[idx]
        random_number = random.randint(0, self.max_number)
        filename = f"{patient_id}_{random_number:03d}.npz"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filename}' does not exist in directory '{self.data_dir}'.")

        # Load data
        data = np.load(filepath)
        image = torch.from_numpy(data["input"])
        label = torch.from_numpy(data["label"])

        if self.transform:
            image, label = self.transform({"ct": image[None, 0], "pet": image[None, 0], "label": label})
        return image, label


if __name__ == "__main__":
    data_dir = "../../test/preprocessed/val"
    splits_file = "../../test/data/splits_final.json"
    split = read_split(splits_file, 0)["train"]
    transform = None

    dataset = RandomPatientDataset(data_dir, split, transform=transform)
    # Accessing a sample
    for i in tqdm(range(2)):
        # Get the first sample
        sample = dataset[i]
        image, label = sample
        plot_ct_pet_label(ct=image[0], pet=image[1].numpy(), label=label[0])
