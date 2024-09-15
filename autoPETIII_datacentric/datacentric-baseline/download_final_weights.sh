# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash


#weights to be included
declare -a arr_preproc=(
    "last_f0.ckpt"
    "last_f1.ckpt"
    "last_f2.ckpt"
    "last_f3.ckpt"
    "last_f4.ckpt"
)


for f in "${arr_preproc[@]}"; do
    echo "Download: ${f}"
    curl --create-dirs -o "weights/${f}" "https://hub.dkfz.de/s/R9nDpWCxEwwdDmZ/${f// /%20}"
done
