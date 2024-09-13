# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time
from typing import Dict, List, Optional, Union

import monai.transforms as mt
import numpy as np
import torch
import typer
from autopet3.datacentric.utils import SimpleParser, get_file_dict_nn, read_split
from autopet3.fixed.dynunet import NNUnet
from autopet3.fixed.evaluation import AutoPETMetricAggregator
from autopet3.fixed.utils import plot_results, plot_ct_pet_label, plot_ct_pet_label_results
from omegaconf import OmegaConf, ListConfig
from tqdm import tqdm
import scipy.ndimage as ndi
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, binary_opening, binary_closing, binary_fill_holes
from skimage.morphology import remove_small_objects, ball
import json

app = typer.Typer()


class PredictModeldev:
    """The PredictModel class is responsible for preprocessing input data, loading and evaluating models, and making
    predictions using an ensemble of models. It also supports test-time augmentation (TTA) for improved predictions.
    The class can run the prediction pipeline on CT and PET image files, save the output, and optionally evaluate the
    results.

    Example Usage
    # Create an instance of PredictModeldev
    model_paths = ["model1.ckpt", "model2.ckpt", "model3.ckpt"]
    predictor = PredictModeldev(model_paths, sw_batch_size=12, tta=True, random_flips=2)

    # Run the prediction pipeline
    ct_file_path = "ct_image.nii.gz"
    pet_file_path = "pet_image.nii.gz"
    label = "label_image.nii.gz"
    save_path = "output_folder"
    metrics = predictor.run(ct_file_path, pet_file_path, label=label, save_path=save_path, verbose=True)
    """

    def __init__(self, model_paths: List[str], sw_batch_size: int = 6, tta: bool = False, random_flips: int = 0):
        """Initialize the class with the given parameters.
        Args:
            model_paths (List[str]): List of model paths.
            sw_batch_size (int, optional): Batch size for the model. Defaults to 6.
            tta (bool, optional): Flag for test-time augmentation. Defaults to False.
            random_flips (int, optional): Number of random flips. Defaults to 0.
        Returns:
            None

        """
        self.ckpts = model_paths
        self.transform = None
        self.sw_batch_size = sw_batch_size
        self.tta = tta
        self.tta_flips = [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        self.random_flips = np.clip(random_flips, 0, len(self.tta_flips))

    @staticmethod
    def preprocess(input: Dict[str, str]) -> torch.Tensor:
        """Preprocesses the input data by applying a series of transformations to the CT and PET images.
        Args:
            input (Dict[str, str]): A dictionary containing the paths to the CT and PET images.
        Returns:
            torch.Tensor: The preprocessed input data as a tensor with a batch dimension.

        """
        # Define the percentile values for CT and PET images
        ct_percentiles = (-832.062744140625, 1127.758544921875)
        ct_norm = torch.Tensor((107.73438968591431, 286.34403119451997))
        pet_percentiles = (1.0433332920074463, 51.211158752441406)
        pet_norm = torch.Tensor((7.063827929027176, 7.960414805306728))

        # Define the spacing for the images
        spacing = (2.0364201068878174, 2.03642010688781740, 3.0)

        # Define a list of transforms
        keys = ["ct", "pet"]
        transforms = [
            mt.LoadImaged(keys=keys),
            mt.EnsureChannelFirstd(keys=keys),
            mt.EnsureTyped(keys=keys),
            mt.Orientationd(keys=keys, axcodes="LAS"),
            mt.Spacingd(
                keys=keys,
                pixdim=spacing,
                mode="bilinear",
            ),
            mt.ScaleIntensityRanged(
                keys=["ct"], a_min=ct_percentiles[0], a_max=ct_percentiles[1], b_min=0, b_max=1, clip=True
            ),
            mt.ScaleIntensityRanged(
                keys=["pet"], a_min=pet_percentiles[0], a_max=pet_percentiles[1], b_min=0, b_max=1, clip=True
            ),
            mt.ScaleIntensityRanged(
                keys=["ct"], a_min=0, a_max=1, b_min=ct_percentiles[0], b_max=ct_percentiles[1], clip=True
            ),
            mt.ScaleIntensityRanged(
                keys=["pet"], a_min=0, a_max=1, b_min=pet_percentiles[0], b_max=pet_percentiles[1], clip=True
            ),
            mt.NormalizeIntensityd(keys=["ct"], subtrahend=ct_norm[0], divisor=ct_norm[1]),
            mt.NormalizeIntensityd(keys=["pet"], subtrahend=pet_norm[0], divisor=pet_norm[1]),
            mt.ConcatItemsd(keys=keys, name="image", dim=0),
            mt.EnsureTyped(keys=keys),
            mt.ToTensord(keys=keys),
        ]

        # Compose and apply the transforms, add batch dimension
        transform = mt.Compose(transforms)
        output = transform(input)["image"][None, ...]
        return output
    
    @staticmethod
    def merge_close_components(segmentation, distance):
        """Merge close connected components based on a given distance."""
        structure = generate_binary_structure(3, 2)  # 3D connectivity
        dilated_segmentation = binary_dilation(segmentation, structure, iterations=distance)
        labeled_array, num_features = ndi.label(dilated_segmentation)
        merged_segmentation = binary_erosion(labeled_array, structure, iterations=distance)
        return merged_segmentation > 0
    
    def postprocess(self,
                    output: torch.Tensor,
                    min_size: int = 5,
                    connectivity: int = 2,
                    merge_distance: int = None,
                    opening_radius: int = 2,
                    closing_radius: int = 0,
                    monai_remove_min_size: int = None,
                    dilation_radius: int = 0,
                    ) -> np.ndarray:
        """Postprocesses the output data by applying a series of transformations to the prediction tensor.
        Args:
            output (torch.Tensor): The predicted output tensor.
            min_size (int, optional): The minimum size of connected components to keep. Defaults to 100.
            connectivity (int, optional): The connectivity for removing small objects. Defaults to 1.
            merge_distance (int, optional): The distance for merging close connected components. Defaults to 3.
            opening_radius (int, optional): The radius for the ball structuring element for binary opening. Defaults to 2.
            closing_radius (int, optional): The radius for the ball structuring element for binary closing. Defaults to 1.
            monai_remove_min_size (int, optional): The minimum size of connected components to keep using MONAI. Defaults to None.
        Returns:
            torch.Tensor: The postprocessed output data as a tensor.

        """
        output_np = output.cpu().numpy().astype(bool)
        print("Prediction shape before postprocessing", output_np.shape[0])
        # print(f"output min: {output.min()}, output max: {output.max()}, sum: {output.sum()}")
        for b in range(output_np.shape[0]):
            # Remove small connected components
            if min_size is not None:
                output_np[b] = remove_small_objects(output_np[b], min_size=min_size, connectivity=connectivity)
            
            # Apply binary opening
            if opening_radius > 0:
                output_np[b] = binary_opening(output_np[b], structure=np.ones((opening_radius, opening_radius, opening_radius)))
            
            # Fill holes in the connected components using binary closing
            if closing_radius > 0:
                output_np[b] = binary_closing(output_np[b], structure=np.ones((closing_radius, closing_radius, closing_radius)))
            
            # Merge close connected components
            if merge_distance is not None:
                output_np[b] = self.merge_close_components(output_np[b], distance=merge_distance)

            # Remove small connected components using MONAI's remove_small_objects
            if monai_remove_min_size is not None:
                trf = mt.RemoveSmallObjectsd(min_size=monai_remove_min_size, by_measure=False)      
                output_np[b] = trf(output_np[b][None, ...])[0] # add channel dimension and remove it after transform

            # Apply dilation
            if dilation_radius > 0:
                output_np[b] = binary_dilation(output_np[b], structure=ball(dilation_radius))
        
        output_np = torch.tensor(output_np, dtype=output.dtype, device=output.device)
        return output_np
    
                            
    def load_and_evaluate_model(self, model_path: str, input: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        """Load a model from a given path and evaluate it on the input tensor.

        Args:
            model_path (str): The path to the model checkpoint.
            input (torch.Tensor): The input tensor for evaluation.
            device (str, optional): The device to run the evaluation on. Defaults to "cuda".

        Returns:
            torch.Tensor: The evaluation result after applying the model.

        """
        #net = NNUnet.load_from_checkpoint(model_path, sw_batch_size=self.sw_batch_size)

        # instead of loading from checkpoint, load the model and load the state dict
        net = NNUnet(sw_batch_size=self.sw_batch_size)
        # filter out unexpected keys
        if device == "cuda":
            state_dict = torch.load(model_path, map_location=torch.device('cuda'))["state_dict"]
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))["state_dict"]
        #state_dict = torch.load(model_path, map_location='cpu')["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if k in net.state_dict()}

        # load the state dict
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        if device == "cuda":
            net.cuda()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            #with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.tta:
                    pred = self.tta_inference(net, input)
                else:
                    pred = net.forward_logits(input)
        else:
            with torch.no_grad():
                print("predicting on cpu")
                if self.tta:
                    pred = self.tta_inference(net, input)
                else:
                    pred = net.forward_logits(input)
        return torch.sigmoid(pred)

    def predict_ensemble(self, input: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        """Predicts the class label for the given input using an ensemble of models.
        Args:
            input (torch.Tensor): The input tensor to be predicted.
        Returns:
            torch.Tensor: The predicted class labels for the input.

        """
        predictions = [self.load_and_evaluate_model(model, input, device=device) for model in self.ckpts]
        averaged_predictions = torch.mean(torch.stack(predictions), dim=0)
        return torch.ge(averaged_predictions, 0.5)

    def tta_inference(self, model: NNUnet, input: torch.Tensor) -> torch.Tensor:
        """Perform test-time augmentation (TTA) inference on the given model and input tensor.
        Args:
            model (NNUnet): The model to perform inference on.
            input (torch.Tensor): The input tensor to be augmented and evaluated.
        Returns:
                torch.Tensor: The predicted output tensor after applying TTA.
        Description:
        This function performs test-time augmentation (TTA) inference on the given model and input tensor.
        It applies a series of transformations to the input tensor and evaluates the model on the augmented tensor.
        The transformations include flipping the input tensor along different axes.
        The augmented predictions are then averaged to obtain the final predicted output tensor.

        """
        prediction = model.forward_logits(input)

        if self.random_flips == 0:
            flips_to_apply = self.tta_flips
        else:
            flips_to_apply = random.sample(self.tta_flips, self.random_flips)

        for flip_idx in flips_to_apply:  # tqdm(flips_to_apply, desc="TTA"):
            prediction += self.flip(model.forward_logits(self.flip(input, flip_idx)), flip_idx)
        prediction /= len(flips_to_apply) + 1
        return prediction

    @staticmethod
    def flip(data, axis):
        return torch.flip(data, dims=axis)

    def run(
        self, ct_file_path: str, pet_file_path: str, label: str = None, save_path: str = None, verbose: bool = False, 
        device: str = "cuda",
        do_postprocess: bool = True,
        body_fg_segmentator: bool = False,
        clip_suv: Union[float, bool] = -1.0,
        **postprocess_kwargs: dict
    ) -> Union[int, dict]:
        """Runs the prediction pipeline on the given CT and PET image files.
        Args:
            ct_file_path (str): The path to the CT image file.
            pet_file_path (str): The path to the PET image file.
            label (str, optional): The path to the ground truth label image file. Defaults to None.
            save_path (str, optional): The path to save the output image. Defaults to None.
            verbose (bool, optional): Whether to print the timings. Defaults to False.
            device (str, optional): The device to run the prediction on. Defaults to "cuda".
            do_postprocess (bool, optional): Whether to perform postprocessing. Defaults to False.
            body_fg_segmentator (bool, optional): Whether to remove false positives outside body segmentation. Defaults to False.
            clip_suv (Union[float, bool], optional): The SUV value to clip the output. Defaults to -1.0. If True, values smaller than 1.0 are clipped.
            **postprocess_kwargs: Additional keyword arguments for postprocessing.
        Returns:
            int: 0 if the pipeline is successfully run or dict containing the metrics if evaluation is enabled.
        This function performs the following steps:
        1. Preprocessing: Loads the PET image, applies preprocessing to the CT and PET images, and stores the
        preprocessed data.
        2. Prediction: Predicts the output using an ensemble of models and TTA if TTA is enabled.
        3. Save Output: Resamples the prediction to match the reference image and saves it as a NIfTI file.
        4. Evaluation (optional): If a label image is provided, loads the ground truth label image, performs orientation
        and spacing adjustment, plots the results, and computes metrics.
        5. Print Timings: Prints the timings for preprocessing, prediction, saving, and total time.
        Note: The function assumes that the necessary models and preprocessing steps have been set up before calling
        this function.

        """
        start_time = time.time()

        # Preprocessing
        reference = mt.LoadImage()(pet_file_path)
        print("Reference image before preprocessing:")
        print(" - shape:", reference.shape)
        print(" - spacing:", reference.meta["pixdim"][1:4])
        print(" - type:", type(reference))

        data = self.preprocess({"ct": ct_file_path, "pet": pet_file_path})
        preprocessing_time = time.time() - start_time

        print("After preprocessing:")
        print(" - shape:", data.shape)
        print(" - spacing:", data.meta["pixdim"][1:4])
        print(" - type:", type(data))

        # Prediction
        start_prediction = time.time()
        if device == "cuda":
            prediction = self.predict_ensemble(data.cuda(), device=device)
        else:
            prediction = self.predict_ensemble(data, device=device)
        prediction_time = time.time() - start_prediction

        # Postprocessing (New)
        if do_postprocess:
            start_postprocessing = time.time()
            prediction[0] = self.postprocess(prediction[0], **postprocess_kwargs)
            postprocessing_time = time.time() - start_postprocessing
    
        output = mt.ResampleToMatch()(prediction[0], reference[None, ...], mode="nearest")
        print("Output dimensions:")
        print(" - shape:", output.shape)
        print(" - spacing:", output.meta["pixdim"][1:4])
        print(" - type:", type(output))

        if body_fg_segmentator:
            start_body_segmentator = time.time()
            ct_reference = mt.LoadImage()(ct_file_path)
            anatomy = ct_reference
            # threshold this image to get the body foreground mask
            anatomy = anatomy > -700
            print("Anatomy dimensions:")
            print(" - shape:", anatomy.shape)
            print(" - type:", type(anatomy))

            anatomy = binary_fill_holes(anatomy)
            anatomy = binary_erosion(anatomy, structure=ball(2))
            anatomy = binary_dilation(anatomy, structure=ball(5))
            
            anatomy = torch.tensor(anatomy, dtype=output.dtype, device=output.device)

            output[0] *= anatomy
            body_segmentator_time = time.time() - start_body_segmentator

        # clip segmentation values where the suv is smaller than value
        clip_suv = 1.0 if clip_suv is True else clip_suv
        if clip_suv > 0.0:
            output[reference[None, ...] < clip_suv] = 0

        # Resample and save output
        start_saving = time.time()

        if save_path is not None:
            mt.SaveImage(
                output_dir=save_path,
                output_ext=".nii.gz",
                output_postfix="pred",
                separate_folder=False,
                output_dtype=np.uint8,
            )(output[0])
        saving_time = time.time() - start_saving

        # Evaluation
        if label is not None:
            ground_truth = mt.LoadImage()(label)
            test_aggregator = AutoPETMetricAggregator()
            test_aggregator(output[0], ground_truth)
            metrics_ = test_aggregator.compute()

            # Multiply with volume spacing -> to voxel_vol in ml
            volume = np.prod(reference.meta["pixdim"][1:4]) / 1000
            metrics = {
                "dice_score": metrics_["dice_score"],
                "fp_volume": metrics_["false_positives"] * volume,
                "fn_volume": metrics_["false_negatives"] * volume,

            }
            if verbose:
                plot_results(output[0], ground_truth)
                plot_ct_pet_label_results(ct=data[0,0], pet=data[0,1], label=ground_truth, prediction=output[0])
                print("Spacing:", reference.meta["pixdim"][1:4])
                print(metrics)
                
                total_time = time.time() - start_time
                print(f"Data preprocessing time: {preprocessing_time} seconds")
                print(f"Prediction time: {prediction_time} seconds")
                if do_postprocess:
                    print(f"Postprocessing time: {postprocessing_time} seconds")
                if body_fg_segmentator:
                    print(f"Body segmentator time: {body_segmentator_time} seconds")
                print(f"Saving time: {saving_time} seconds")
                print(f"Total time: {total_time} seconds")
            return metrics

        if verbose:
            total_time = time.time() - start_time
            print(f"Data preprocessing time: {preprocessing_time} seconds")
            print(f"Prediction time: {prediction_time} seconds")
            if do_postprocess:
                print(f"Postprocessing time: {postprocessing_time} seconds")
            print(f"Saving time: {saving_time} seconds")
            print(f"Total time: {total_time} seconds")
        return 0


@app.command()
def infer(
    ct_file_path: str,
    pet_file_path: str,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    verbose: bool = False,
    model_paths: List[str] = typer.Argument(..., help="Paths to the model checkpoints"),
    sw_batch_size: int = 6,
    tta: bool = False,
    random_flips: int = 0,
    do_postprocess: bool = True,
    body_fg_segmentator: bool = False,
    clip_suv: float = -1.0,
    postprocess_kwargs = None
):
    #new
    if postprocess_kwargs:
        postprocess_kwargs = json.loads(postprocess_kwargs)
    else:
        postprocess_kwargs = {}
        
    predict = PredictModeldev(model_paths, sw_batch_size, tta, random_flips)
    result = predict.run(ct_file_path, pet_file_path, label, save_path, verbose,
                         do_postprocess=do_postprocess, body_fg_segmentator=body_fg_segmentator, clip_suv=clip_suv, **postprocess_kwargs)
    typer.echo(result)

# new infer function for testing
def infer_test(
    ct_file_path: str,
    pet_file_path: str,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    verbose: bool = False,
    model_paths: List[str] = typer.Argument(..., help="Paths to the model checkpoints"),
    sw_batch_size: int = 6,
    tta: bool = False,
    random_flips: int = 0,
    postprocess_kwargs = None, 
    device: str = "cuda",
    do_postprocess: bool = True,
    body_fg_segmentator: bool = False,
    clip_suv: float = -1.0,
    ):
    
    if postprocess_kwargs:
        postprocess_kwargs = json.loads(postprocess_kwargs)
    else:
        postprocess_kwargs = {}

    predict = PredictModeldev(model_paths, sw_batch_size, tta, random_flips)
    result = predict.run(ct_file_path, pet_file_path, label, save_path, verbose, device=device,
                         do_postprocess=do_postprocess, body_fg_segmentator=body_fg_segmentator, clip_suv=clip_suv, **postprocess_kwargs)
    return result

@app.command()
def evaluate(
    config: str = "config/test_predict.yml",
    sw_batch_size: int = 6,
    tta: bool = False,
    random_flips: int = 0,
    result_path: Optional[str] = None,
    postprocess_kwargs = None,
    clip_suv: float = -1.0,
):
    config_file_name = os.path.basename(config)        
    config = OmegaConf.load(config)
    # add other parameters (sw_batch_size, tta, random_flips) to the config file
    config.sw_batch_size = sw_batch_size
    config.tta = tta
    config.random_flips = random_flips
    current_time = time.strftime("%Y%m%d-%H%M%S")
    result_path = result_path if result_path is not None else config.get('result_path', "test/")
    with open(os.path.join(result_path, f"config_{current_time}.json"), "w") as f:
        json.dump(OmegaConf.to_container(config), f)

    do_postprocess = config.get('do_postprocess', False)
    # get postprocess kwargs from config file if not provided
    if postprocess_kwargs is None:
        postprocess_kwargs = config.get('postprocess_kwargs', {})
    elif isinstance(postprocess_kwargs, str):
        postprocess_kwargs = json.loads(postprocess_kwargs)
    elif not isinstance(postprocess_kwargs, dict):
        raise ValueError("postprocess_kwargs must be a dict, a JSON string, or None.")
    
    body_fg_segmentator= config.get('body_fg_segmentator', False)
    clip_suv = config.get('clip_suv', -1.0)

    model_paths = [config.model.ckpt_path] if isinstance(config.model.ckpt_path, str) else config.model.ckpt_path
    predict = PredictModeldev(model_paths, sw_batch_size=sw_batch_size, tta=tta, random_flips=random_flips)
    parser = SimpleParser(os.path.join(result_path, f"results_config_{config_file_name}_f{config.data.fold}.json"))
    split = read_split(config.data.splits_file, config.data.fold)
    files = get_file_dict_nn(config.data.data_dir, split["val"], suffix=".nii.gz")
    for file in tqdm(files, desc="Predicting"):
        result = predict.run(str(file["ct"]), str(file["pet"]), label=str(file["label"]), verbose=False,
                             do_postprocess=do_postprocess, body_fg_segmentator=body_fg_segmentator, clip_suv=clip_suv, **postprocess_kwargs)
        parser.write(file, result)

@app.command()
def evaluatefolds(
    config: str = "config/test_predict.yml",
    sw_batch_size: int = 6,
    tta: bool = False,
    random_flips: int = 0,
    result_path: Optional[str] = None,
    postprocess_kwargs = None,
):
    config_file_name = os.path.basename(config).split(".")[0] 
    typer.echo(f"config_file_name: {config_file_name}")    
    config = OmegaConf.load(config)
    # add other parameters (sw_batch_size, tta, random_flips) to the config file
    config.sw_batch_size = sw_batch_size
    config.tta = tta
    config.random_flips = random_flips

    # Get the current time for unique directory creation
    current_time = time.strftime("%Y%m%d-%H%M%S")
    # Get result path from config file if not provided
    result_path = result_path if result_path is not None else config.get('result_path', "test/")
    run_dir = os.path.join(result_path, f"run_{current_time}_{config_file_name}_tt{tta}_rf{random_flips}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, f"config_{config_file_name}_{current_time}.json"), "w") as f:
        json.dump(OmegaConf.to_container(config), f)

    do_postprocess = config.get('do_postprocess', False)
    # get postprocess kwargs from config file if not provided
    if postprocess_kwargs is None:
        postprocess_kwargs = config.get('postprocess_kwargs', {})
    elif isinstance(postprocess_kwargs, str):
        postprocess_kwargs = json.loads(postprocess_kwargs)
    elif not isinstance(postprocess_kwargs, dict):
        raise ValueError("postprocess_kwargs must be a dict, a JSON string, or None.")
    
    body_fg_segmentator = config.get('body_fg_segmentator', False)
    clip_suv = config.get('clip_suv', -1.0)


    model_paths = [config.model.ckpt_path] if isinstance(config.model.ckpt_path, str) else config.model.ckpt_path
    
    folds = list(config.data.fold) if isinstance(config.data.fold, ListConfig) else [config.data.fold]  
    # check if there is the same number of models as folds
    if len(model_paths) == len(folds):
        for model_path in model_paths:
            config_path = os.path.dirname(os.path.dirname(model_path))
            # get yml file that starts with "trainconfig" in the folder
            trainconfig_files = [f for f in os.listdir(config_path) if f.startswith("trainconfig") and f.endswith(".yml")]
            if len(trainconfig_files) == 0:
                raise ValueError("No trainconfig file found in the model path folder")
            elif len(trainconfig_files) > 1:
                raise ValueError("Multiple trainconfig files found in the model path folder")
            else:
                trainconfig_file = trainconfig_files[0]
                with open(os.path.join(config_path, trainconfig_file), "r") as f:
                    trainconfig = OmegaConf.load(f)
                    fold = trainconfig.data.fold # actual fold of the model
                    # check if fold is in the list of folds
                    if fold not in folds:
                        raise ValueError(f"Fold {fold} is not in the list of folds {folds}")

            predict = PredictModeldev([model_path], sw_batch_size=sw_batch_size, tta=tta, random_flips=random_flips)
            parser = SimpleParser(os.path.join(run_dir, f"results_config_{config_file_name}_f{fold}.json"))
            split = read_split(config.data.splits_file, fold)
            files = get_file_dict_nn(config.data.data_dir, split["val"], suffix=".nii.gz")
            for file in tqdm(files, desc="Predicting"):
                result = predict.run(str(file["ct"]), str(file["pet"]), label=str(file["label"]), verbose=False,
                                     do_postprocess=do_postprocess, body_fg_segmentator=body_fg_segmentator,
                                     clip_suv=clip_suv, **postprocess_kwargs)
                parser.write(file, result)
    else:
        raise ValueError("Number of models should be equal to the number of folds")

if __name__ == "__main__":
    app()
