#!/bin/bash

source /mnt/data_nas/penganyang/miniconda3/etc/profile.d/conda.sh
conda activate lamstare
export PYTHONPATH=/mnt/data_nas/public/multitask/LAMstare
export CONDA_ENV=openlam_db
python /mnt/data_nas/public/multitask/LAMstare/lamstare/experiments/plt_ood.py
# python /mnt/data_nas/public/multitask/LAMstare/lamstare/experiments/plt_test.py
