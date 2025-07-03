#!/bin/bash

source /mnt/data_nas/penganyang/miniconda3/etc/profile.d/conda.sh
conda activate lamstare
export PYTHONPATH=/mnt/data_nas/public/multitask/LAMstare
export CONDA_ENV=lamstare
cd /mnt/data_nas/public/multitask/LAMstare/lamstare/utils/
python ./submit_cronjob.py
