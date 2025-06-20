#!/bin/bash

source /mnt/data_nas/public/miniconda3/etc/profile.d/conda.sh
conda activate lamstare
export PYTHONPATH=/mnt/data_nas/public/multitask/LAMstare
export CONDA_ENV=openlam_db
cd /mnt/data_nas/public/multitask/LAMstare/lamstare/utils/
python ./submit_cronjob.py
