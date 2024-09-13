import os
import shutil
from typing import Union

import matplotlib.pylab as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from autopet3.fixed.evaluation import AutoPETMetricAggregator


class SaveFileToLoggerDirCallback(Callback):
    def __init__(self, config_file: str):
        """Save the provided config file to the log directory.
        Args:
            config_file (str): Path to the configuration file.

        """
        super().__init__()
        self.config_file = config_file

    def on_train_start(self, trainer, pl_module):
        try:
            save_dir = trainer.logger.log_dir
            file_path = os.path.join(save_dir, self.config_file.split("/")[-1])
            shutil.copy(self.config_file, file_path)
        except Exception as e:
            print(f"Failed to copy config file: {e}")


def plot_ct_pet_label(
    ct: Union[np.ndarray, torch.Tensor],
    pet: Union[np.ndarray, torch.Tensor],
    label: Union[np.ndarray, torch.Tensor],
    axis: int = 1,
) -> None:
    """Plot the sum of the label, CT, and PET images along the second dimension.
    Args:
        ct (Union[np.ndarray, torch.Tensor]): The CT image.
        pet (Union[np.ndarray, torch.Tensor]): The PET image.
        label (Union[np.ndarray, torch.Tensor]): The label image.
        axis (int): The axis along which the sum will be computed.

    """
    ct = ct.detach().cpu().numpy() if isinstance(ct, torch.Tensor) else ct
    pet = pet.detach().cpu().numpy() if isinstance(pet, torch.Tensor) else pet
    label = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label

    data = {"ct": ct, "pet": pet, "label": label}
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Plot the sum of the CT image along the second dimension
    axes[0].imshow(np.rot90(data["ct"].squeeze().sum(axis)), cmap="gray")
    axes[0].set_title("Sum of CT Image")

    # Plot the sum of the CT image along the second dimension
    axes[1].imshow(np.rot90(np.amax(data["pet"].squeeze(), axis)), cmap="gray")
    axes[1].set_title("Amax of PET Image")

    # Plot the sum of the label image along the second dimension
    axes[2].imshow(np.rot90(data["label"].squeeze().sum(axis)), cmap="gray")
    axes[2].set_title("Sum of Label Image")
    plt.show()


def plot_results(
    prediction: Union[np.ndarray, torch.Tensor],
    label: Union[np.ndarray, torch.Tensor],
    axis: int = 1,
    print_metrics: bool = False,
) -> None:
    """Plot the results of a prediction compared to the ground truth label.
    Args:
        prediction (Union[np.ndarray, torch.Tensor]): The predicted values.
        label (Union[np.ndarray, torch.Tensor]): The ground truth values.
        axis (int): The axis along which the sum will be computed.
        print_metrics (bool): Whether to print the metrics.
    Returns:
        None

    """
    pred_array = prediction.detach().cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
    label_array = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Plot the sum of the label image along the second dimension
    axes[0].imshow(np.rot90(np.sum(pred_array.squeeze(), axis)), cmap="gray")
    axes[0].set_title("Amax of Prediction")

    # Plot the sum of the CT image along the second dimension
    axes[1].imshow(np.rot90(np.sum(label_array.squeeze(), axis)), cmap="gray")
    axes[1].set_title("Amax of GT")

    # Plot the sum of the CT image along the second dimension
    axes[2].imshow(np.rot90(((label_array.squeeze() - pred_array.squeeze()) ** 2).mean(1)), cmap="Reds")
    axes[2].set_title("Squared error")
    plt.show()

    if print_metrics:
        test_aggregator = AutoPETMetricAggregator()
        test_aggregator.update(pred_array, label_array)
        print(test_aggregator.compute())

def plot_ct_pet_label_results(
    ct: Union[np.ndarray, torch.Tensor],
    pet: Union[np.ndarray, torch.Tensor],
    label: Union[np.ndarray, torch.Tensor],
    prediction: Union[np.ndarray, torch.Tensor],
    print_metrics: bool = False,
    axis: int = 1,
) -> None:
    """Plot the sum of the label, CT, and PET images along the second dimension.
    Args:
        ct (Union[np.ndarray, torch.Tensor]): The CT image.
        pet (Union[np.ndarray, torch.Tensor]): The PET image.
        label (Union[np.ndarray, torch.Tensor]): The label image.
        axis (int): The axis along which the sum will be computed.

    """
    ct = ct.detach().cpu().numpy() if isinstance(ct, torch.Tensor) else ct
    pet = pet.detach().cpu().numpy() if isinstance(pet, torch.Tensor) else pet
    label = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label

    data = {"ct": ct, "pet": pet, "label": label}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    # Plot the sum of the CT image along the second dimension
    axes[0, 0].imshow(np.rot90(data["ct"].squeeze().sum(axis)), cmap="gray")
    axes[0, 0].set_title("Sum of CT Image")

    # Plot the sum of the CT image along the second dimension
    axes[0, 1].imshow(np.rot90(np.amax(data["pet"].squeeze(), axis)), cmap="gray")
    axes[0, 1].set_title("Amax of PET Image")

    # Plot the sum of the label image along the second dimension
    axes[0, 2].imshow(np.rot90(data["label"].squeeze().sum(axis)), cmap="gray")
    axes[0, 2].set_title("Sum of Label Image")
   
    # Plot the sum of the label image along the second dimension
    axes[1, 0].imshow(np.rot90(np.sum(prediction.squeeze(), axis)), cmap="gray")
    axes[1, 0].set_title("Amax of Prediction")

    # Plot the sum of the CT image along the second dimension
    axes[1, 1].imshow(np.rot90(np.sum(label.squeeze(), axis)), cmap="gray")
    axes[1, 1].set_title("Amax of GT")

    # Plot the sum of the CT image along the second dimension
    axes[1, 2].imshow(np.rot90(((label.squeeze() - prediction.squeeze()) ** 2).mean(1)), cmap="Reds")
    axes[1, 2].set_title("Squared error")
    plt.show()

    if print_metrics:
        test_aggregator = AutoPETMetricAggregator()
        test_aggregator.update(prediction, label)
        print(test_aggregator.compute())
    