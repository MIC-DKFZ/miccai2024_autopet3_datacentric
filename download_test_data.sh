#!/bin/bash

#
# In case of trouble with git lfs, you can use this script to download the test data
#

# Files to be included from original dataset
declare -a arr_orig=(
    "imagesTr/fdg_2ce074c2ea_02-04-2006-NA-PET-CT Ganzkoerper  primaer mit KM-43376_0000.nii.gz"
    "imagesTr/fdg_2ce074c2ea_02-04-2006-NA-PET-CT Ganzkoerper  primaer mit KM-43376_0001.nii.gz"
    "imagesTr/fdg_3b1c9155f5_01-31-2003-NA-PET-CT Ganzkoerper  primaer mit KM-94971_0000.nii.gz"
    "imagesTr/fdg_3b1c9155f5_01-31-2003-NA-PET-CT Ganzkoerper  primaer mit KM-94971_0001.nii.gz"
    "imagesTr/fdg_af547fa618_03-11-2006-NA-PET-CT Ganzkoerper  primaer mit KM-11358_0000.nii.gz"
    "imagesTr/fdg_af547fa618_03-11-2006-NA-PET-CT Ganzkoerper  primaer mit KM-11358_0001.nii.gz"
    "imagesTr/psma_69ea0c011af2e2d4_2014-05-01_0000.nii.gz"
    "imagesTr/psma_69ea0c011af2e2d4_2014-05-01_0001.nii.gz"
    "imagesTr/psma_95b833d46f153cd2_2017-11-18_0000.nii.gz"
    "imagesTr/psma_95b833d46f153cd2_2017-11-18_0001.nii.gz"
    "imagesTr/psma_95b833d46f153cd2_2018-04-16_0000.nii.gz"
    "imagesTr/psma_95b833d46f153cd2_2018-04-16_0001.nii.gz"
    "labelsTr/fdg_2ce074c2ea_02-04-2006-NA-PET-CT Ganzkoerper  primaer mit KM-43376.nii.gz"
    "labelsTr/fdg_3b1c9155f5_01-31-2003-NA-PET-CT Ganzkoerper  primaer mit KM-94971.nii.gz"
    "labelsTr/fdg_af547fa618_03-11-2006-NA-PET-CT Ganzkoerper  primaer mit KM-11358.nii.gz"
    "labelsTr/psma_69ea0c011af2e2d4_2014-05-01.nii.gz"
    "labelsTr/psma_95b833d46f153cd2_2017-11-18.nii.gz"
    "labelsTr/psma_95b833d46f153cd2_2018-04-16.nii.gz"
)

#files to be included from preprocessed data
declare -a arr_preproc=(
    "preprocessed/train/fdg_2ce074c2ea_02-04-2006-NA-PET-CT Ganzkoerper  primaer mit KM-43376_000.npz"
    "preprocessed/train/fdg_2ce074c2ea_02-04-2006-NA-PET-CT Ganzkoerper  primaer mit KM-43376_001.npz"
    "preprocessed/train/fdg_3b1c9155f5_01-31-2003-NA-PET-CT Ganzkoerper  primaer mit KM-94971_000.npz"
    "preprocessed/train/fdg_3b1c9155f5_01-31-2003-NA-PET-CT Ganzkoerper  primaer mit KM-94971_001.npz"
    "preprocessed/train/fdg_af547fa618_03-11-2006-NA-PET-CT Ganzkoerper  primaer mit KM-11358_000.npz"
    "preprocessed/train/fdg_af547fa618_03-11-2006-NA-PET-CT Ganzkoerper  primaer mit KM-11358_001.npz"
    "preprocessed/train/psma_69ea0c011af2e2d4_2014-05-01_000.npz"
    "preprocessed/train/psma_69ea0c011af2e2d4_2014-05-01_001.npz"
    "preprocessed/train/psma_95b833d46f153cd2_2017-11-18_000.npz"
    "preprocessed/train/psma_95b833d46f153cd2_2017-11-18_001.npz"
    "preprocessed/train/psma_95b833d46f153cd2_2018-04-16_000.npz"
    "preprocessed/train/psma_95b833d46f153cd2_2018-04-16_001.npz"
    "preprocessed/val/fdg_2ce074c2ea_02-04-2006-NA-PET-CT Ganzkoerper  primaer mit KM-43376_000.npz"
    "preprocessed/val/fdg_3b1c9155f5_01-31-2003-NA-PET-CT Ganzkoerper  primaer mit KM-94971_000.npz"
    "preprocessed/val/fdg_af547fa618_03-11-2006-NA-PET-CT Ganzkoerper  primaer mit KM-11358_000.npz"
    "preprocessed/val/psma_69ea0c011af2e2d4_2014-05-01_000.npz"
    "preprocessed/val/psma_95b833d46f153cd2_2017-11-18_000.npz"
    "preprocessed/val/psma_95b833d46f153cd2_2018-04-16_000.npz"
    "epoch=581_fold0.ckpt"
)


for f in "${arr_orig[@]}"; do
    echo "Download: ${f}"
    curl --create-dirs -o "test/data/${f}" "https://syncandshare.lrz.de/dl/fiCJ6mQcjefMTQdKJsBSys/${f// /%20}"
done


for f in "${arr_preproc[@]}"; do
    echo "Download: ${f}"
    curl --create-dirs -o "test/${f}" "https://syncandshare.lrz.de/dl/fiB6N4UaiWJ8m88nPkoN5G/${f// /%20}"
done
