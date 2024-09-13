import json
import os
from pathlib import Path
from typing import List

import numpy as np


def get_patients(path_list: List[str]) -> List[str]:
    # extract the patient ID from the path
    return [path.split("_")[1] for path in path_list]


def read_split(splits_file: str, fold: int = 0) -> dict:
    """Read a specific fold from a JSON file containing splits.
    Args:
        splits_file (str): The path to the JSON file containing the splits.
        fold (int): The fold number to read from the splits file. Defaults to 0.
    Returns:
        dict: The dictionary representing the split for the specified fold.

    """
    with open(splits_file) as json_file:
        splits_dict = json.load(json_file)[fold]
    return splits_dict


def get_file_dict_nn(root: str, split: List[str], suffix: str = ".nii.gz") -> List[dict]:
    """Generate a dictionary containing paths to CT, PET, and label files for each element in the split list.
    Args:
        root (str): The root directory path.
        split (List[str]): List of elements to generate paths for.
        suffix (str): Suffix for the file extensions. Default is ".nii.gz".
    Returns:
        List[dict]: A list of dictionaries containing paths to CT, PET, and label files for each element in split.

    """
    root = Path(root)
    data = [
        {
            "ct": root / "imagesTr" / f"{element}_0000{suffix}",
            "pet": root / "imagesTr" / f"{element}_0001{suffix}",
            "label": root / "labelsTr" / f"{element}{suffix}",
        }
        for element in split
    ]
    return data


def extract_paths_containing_tracer(file_dicts: List[dict], key: str = "label") -> np.ndarray:
    # Small helper to extract the tracer from the file name
    tracer_list = []
    for file_dict in file_dicts:
        path = file_dict[key]
        if "fdg" in path.name.lower():
            tracer_list.append("fdg")
        else:
            tracer_list.append("psma")
    return np.array(tracer_list)


def result_parser(net, datamodule, trainer):
    """Save prediction scores to file.
    Args:
        net: LightningModule
        datamodule: LightningDataModule
        trainer: LightningTrainer
    Returns:
        None

    """
    # Save prediction scores to file
    metrics_dice = net.test_aggregator.dice_scores
    metrics_fp = net.test_aggregator.false_positives
    metrics_fn = net.test_aggregator.false_negatives
    results = []

    # Get the file names from the test dataset
    # Change this lines if you don't have a monai dict dataset!
    file_names = datamodule.test_dataset.data

    # Add summary results
    results.append({"Summary": net.test_aggregator.compute()})

    # Add summary results for FDG and PSMA tracers
    tracers = extract_paths_containing_tracer(file_names)
    for tracer in ["fdg", "psma"]:
        if np.any(tracers == tracer):
            results.append(
                {
                    f"Summary {tracer.upper()}": {
                        "false_positives": np.nanmean(np.array(metrics_fp)[tracers == tracer]),
                        "false_negatives": np.nanmean(np.array(metrics_fn)[tracers == tracer]),
                        "dice_score": np.nanmean(np.array(metrics_dice)[tracers == tracer]),
                    }
                }
            )
        else:
            results.append({f"Summary {tracer.upper()}": None})

    # Add individual file results
    for i, file_info in enumerate(file_names):
        # change this line if you don't have a monai dict dataset!
        file_info_str = {key: str(value) for key, value in file_info.items()}
        result = {
            "file_info": file_info_str,
            "metrics": {
                "false_positives": float(metrics_fp[i]),
                "false_negatives": float(metrics_fn[i]),
                "dice_score": float(metrics_dice[i]),
            },
        }
        results.append(result)

    # Write results to JSON file
    output_path = os.path.join(trainer.logger.log_dir, "results.json")
    with open(output_path, "w") as json_file:
        json.dump(results, json_file, sort_keys=False, indent=4)


class SimpleParser:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.dice_scores = []
        self.false_positives = []
        self.false_negatives = []
        self.tracers = []
        self.data = []

    def write(self, file_info: dict, metrics: dict):
        self.data.append({"file_info": {key: str(value) for key, value in file_info.items()}, "metrics": metrics})
        self.dice_scores.append(metrics["dice_score"])
        self.false_positives.append(metrics["fp_volume"])
        self.false_negatives.append(metrics["fn_volume"])
        self.tracers.append("fdg" if "fdg" in Path(file_info["label"]).name.lower() else "psma")

        results = self.aggregate()
        results.extend(self.data)

        with open(self.output_path, "w") as json_file:
            json.dump(results, json_file, sort_keys=False, indent=4)

    def reset(self):
        self.dice_scores = []
        self.false_positives = []
        self.false_negatives = []
        self.tracers = []
        self.data = []

    def aggregate(self) -> List[dict]:
        results = [
            {
                "Summary": {
                    "dice_score": np.nanmean(np.array(self.dice_scores)),
                    "fp_volume": np.nanmean(np.array(self.false_positives)),
                    "fn_volume": np.nanmean(np.array(self.false_negatives)),
                }
            }
        ]
        tracers = np.array(self.tracers)
        for tracer in ["fdg", "psma"]:
            if np.any(tracers == tracer):
                results.append(
                    {
                        f"Summary {tracer.upper()}": {
                            "dice_score": np.nanmean(np.array(self.dice_scores)[tracers == tracer]),
                            "fp_volume": np.nanmean(np.array(self.false_positives)[tracers == tracer]),
                            "fn_volume": np.nanmean(np.array(self.false_negatives)[tracers == tracer]),
                        }
                    }
                )
            else:
                results.append({f"Summary {tracer.upper()}": None})

        return results
