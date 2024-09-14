# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os

import SimpleITK
import torch

from predict import PredictModel


class Datacentric_baseline:  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces (Automated PET/CT lesion segmentation)
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"
        # according to the specified grand-challenge interfaces (Data centric model)
        self.output_path_category = "/output/data-centric-model.json"
        # where to store the nii files
        self.nii_path = "/opt/algorithm/"
        self.weights_path = "/opt/algorithm/weights/"
        
        self.ckpt_paths = glob.glob(os.path.join(self.weights_path, "*.ckpt"))
        self.tta = True #disable test-time augmentation in the process.py file by setting self.tta = False in the init function.
        self.sw_batch_size = 10 # default 12
        self.random_flips = 5
        self.dynamic_tta = True
        self.max_tta_time = 42 #220
        self.dynamic_ensemble = True
        self.max_ensemble_time = 210

        self.inferer = PredictModel(
            model_paths=self.ckpt_paths,
            sw_batch_size=self.sw_batch_size,
            tta=self.tta,
            random_flips=self.random_flips,
            dynamic_tta=self.dynamic_tta,
            max_tta_time=self.max_tta_time,
            dynamic_ensemble=self.dynamic_ensemble,
            max_ensemble_time=self.max_ensemble_time,
        )

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def save_datacentric(self, value: bool):
        print("Saving datacentric json to " + self.output_path_category)
        with open(self.output_path_category, "w") as json_file:
            json.dump(value, json_file, indent=4)

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "SUV.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "CTres.nii.gz"),
        )
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.output_path, "PRED.nii.gz"),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        """
        Your algorithm goes here
        """
        print("Using weights: ")
        print(self.ckpt_paths)

        self.inferer.run(
            ct_file_path=os.path.join(self.nii_path, "CTres.nii.gz"),
            pet_file_path=os.path.join(self.nii_path, "SUV.nii.gz"),
            save_path=self.output_path,
            verbose=True,
            do_postprocess=False,
            body_fg_segmentator=False,
            clip_suv=True,
            #**postprocess_kwargs,
        )

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.save_datacentric(True)
        self.write_outputs(uuid)


if __name__ == "__main__":
    Datacentric_baseline().process()
