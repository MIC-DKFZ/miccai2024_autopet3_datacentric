from typing import Dict, List, Union

import cc3d
import numpy as np
import torch
from sklearn import metrics

ArrayOrTensor = Union[np.ndarray, torch.Tensor]
ResDict = Dict[str, Union[float]]


class AutoPETMetricAggregator:
    def __init__(self):
        self.false_positives: List[float] = []
        self.false_negatives: List[float] = []
        self.dice_scores: List[float] = []

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def update(self, prediction: ArrayOrTensor, label: ArrayOrTensor) -> ResDict:
        """Update the false positives, false negatives, and dice scores based on the prediction and label arrays.
        Args:
            prediction (Union[np.ndarray, torch.Tensor]): The predicted array.
            label (Union[np.ndarray, torch.Tensor]): The ground truth array.
        Returns:
            Dict[str, Union[float, np.nan]]: A dictionary containing the dice score, false positive pixels, and false
                                             negative pixels.

        """
        prediction = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
        label = label.cpu().numpy() if isinstance(label, torch.Tensor) else label

        label = np.squeeze(label)
        prediction = np.squeeze(prediction)

        if label.ndim != 3 or prediction.ndim != 3:
            raise ValueError("Both gt_array and pred_array must have 3 dimensions.")

        false_pos = self.count_false_positives(prediction, label)
        false_neg = self.count_false_negatives(prediction, label)
        dice = self.calc_dice_score(prediction, label)
        self.false_positives.append(false_pos)
        self.false_negatives.append(false_neg)
        self.dice_scores.append(dice)
        return {"false_positives": false_pos, "false_negatives": false_neg, "dice_score": dice}

    def reset(self):
        self.false_positives = []
        self.false_negatives = []
        self.dice_scores = []

    def compute(self) -> ResDict:
        """Compute the mean of false positives, false negatives, and dice scores.
        Returns
            Dict[str, Union[float, np.nan]]: A dictionary containing the mean of false positives, false negatives, and
                                             dice scores. If any of the values are NaN, the corresponding key will have
                                             a NaN value.

        """
        # Check if each metric is all NaN
        fp_all_nan = np.isnan(self.false_positives).all()
        fn_all_nan = np.isnan(self.false_negatives).all()
        ds_all_nan = np.isnan(self.dice_scores).all()

        # Prepare the result dictionary
        result = {
            "false_positives": np.nan if fp_all_nan else np.nanmean(self.false_positives),
            "false_negatives": np.nan if fn_all_nan else np.nanmean(self.false_negatives),
            "dice_score": np.nan if ds_all_nan else np.nanmean(self.dice_scores),
        }

        return result

    @staticmethod
    def count_false_positives(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """Count the number of false positive pixel, which do not overlap with the ground truth, based on the prediction
        and ground truth arrays.
        Returns zero if the prediction array is empty.
        Args:
            prediction (np.ndarray): The predicted array.
            ground_truth (np.ndarray): The ground truth array.
        returns:
            float: The number of false positive pixel which do not overlap with the ground truth.

        """
        if prediction.sum() == 0:
            return 0

        if ground_truth.sum() == 0:
            # a little bit faster than calculating connected components
            return prediction.sum()

        connected_components = cc3d.connected_components(prediction.astype(int), connectivity=18)
        false_positives = 0

        for idx in range(1, connected_components.max() + 1):
            component_mask = np.isin(connected_components, idx)
            if (component_mask * ground_truth).sum() == 0:
                false_positives += component_mask.sum()

        return false_positives

    @staticmethod
    def count_false_negatives(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """Count the number of false negative pixel, which do not overlap with the ground truth, based on the prediction
        and ground truth arrays.
        Returns nan if the ground truth array is empty.
        Args:
            prediction (np.ndarray): The predicted array.
            ground_truth (np.ndarray): The ground truth array.
        Returns:
            float: The number of false negative pixel, which do not overlap with the prediction.

        """
        if ground_truth.sum() == 0:
            return np.nan

        gt_components = cc3d.connected_components(ground_truth.astype(int), connectivity=18)
        false_negatives = 0

        for component_id in range(1, gt_components.max() + 1):
            component_mask = np.isin(gt_components, component_id)
            if (component_mask * prediction).sum() == 0:
                false_negatives += component_mask.sum()

        return false_negatives

    @staticmethod
    def calc_dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate the Dice score between the prediction and ground truth arrays.
        Returns nan if the ground truth array is empty.
        Args:
            prediction (np.ndarray): The predicted array.
            ground_truth (np.ndarray): The ground truth array.
        Returns:
            float: The Dice score between the prediction and ground truth arrays.

        """
        if ground_truth.sum() == 0:
            return np.nan

        intersection = (ground_truth * prediction).sum()
        union = ground_truth.sum() + prediction.sum()
        dice_score = 2 * intersection / union

        return dice_score

    @staticmethod
    def calculate_confusion(prediction: np.ndarray, ground_truth: np.ndarray):
        tn, fp, fn, tp = metrics.confusion_matrix(ground_truth.ravel(), prediction.ravel()).ravel()
        return {"TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn, "F1:", (2 * tp) / (2 * tp + fn + fp)}


if __name__ == "__main__":
    aggregator = AutoPETMetricAggregator()
    gt_array = np.zeros((1, 1, 10, 10, 10))
    pred_array = np.zeros((1, 1, 10, 10, 10))

    gt_array[0, 0, 1:5, 1:5, 1:5] = 1
    gt_array[0, 0, 6:10, 6:10, 6:10] = 1

    # Create a cube in the prediction overlapping with one of the cubes in the ground truth
    pred_array[0, 0, 0:4, 0:4, 0:4] = 1

    aggregator.update(pred_array, gt_array)
    results = aggregator.compute()
    aggregator.reset()

    # check fn
    print(results)
    aggregator.calculate_confusion(pred_array, gt_array)

    np.testing.assert_almost_equal(results["false_positives"], 0)
    np.testing.assert_almost_equal(results["false_negatives"], 64)  # 6 from each of the cubes not overlapped
    np.testing.assert_almost_equal(results["dice_score"], 0.28125)  # Overlapping 50%

    # check fp
    aggregator.update(pred_array, gt_array)
    results = aggregator.compute()
    aggregator.reset()
    print(results)

    # check multiple
    aggregator.update(pred_array, gt_array)
    aggregator.update(pred_array, gt_array)
    results = aggregator.compute()
    aggregator.reset()
    print(results)

    # check empty out
    aggregator.update(pred_array, np.zeros(gt_array.shape))
    metrics = aggregator.compute()
    aggregator.reset()
    print(metrics)