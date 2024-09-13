import json
import os

import numpy as np
from sklearn.model_selection import GroupKFold


def get_patient(path_list):
    return [path.split("_")[1] for path in path_list]


def make_folds(root_dir: str, output_path: str, num_splits: int = 5, seed: int = 42) -> None:
    """Generate folds for cross-validation based on patient directories.
    Parameters
    root_dir (str): Path to the root directory containing patient data.
    output_path (str): Path to save the generated folds.
    num_splits (int): Number of splits for cross-validation. Default is 5.
    seed (int): Seed for randomization. Default is 42.

    Attention: This function is not completely reproducible. See
    https://stackoverflow.com/questions/41859613/how-to-obtain-reproducible-but-distinct-instances-of-groupkfold

    """
    random_generator = np.random.default_rng(seed)

    patients = os.listdir(root_dir)
    random_generator.shuffle(patients)

    group_ids = get_patient(patients)

    group_kfold = GroupKFold(n_splits=num_splits)
    folds = []

    for fold_idx, (train_indices, test_indices) in enumerate(group_kfold.split(patients, groups=group_ids)):
        folds.append(
            {
                "train": [patients[idx].split(".nii.gz")[0] for idx in train_indices],
                "val": [patients[idx].split(".nii.gz")[0] for idx in test_indices],
            }
        )

    with open(output_path, "w") as json_file:
        json.dump(folds, json_file, sort_keys=False, indent=4)


if __name__ == "__main__":
    root = "../test/data/labelsTr"
    output_path = "../test/data/splits_final.json"
    k = 5
    seed = 42
    make_folds(root, output_path, k, seed)
