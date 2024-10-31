#!/bin/bash

source /mnt/workspace/public/Miniconda/etc/profile.d/conda.sh
conda activate lamstare
python /mnt/workspace/public/multitask/LAMstare/lamstare/experiments/plt_lcurve.py
python /mnt/workspace/public/multitask/LAMstare/lamstare/experiments/plt_test.py
