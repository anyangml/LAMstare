#!/bin/bash

source /mnt/data_nas/public/Miniconda/etc/profile.d/conda.sh
conda activate lamstare
bash /mnt/data_nas/penganyang/renew_dlc.sh # to renew dlc credential
export PYTHONPATH=/mnt/data_nas/cc/LAMstare_new
export CONDA_ENV=openlam_db
cd /mnt/data_nas/cc/LAMstare_new/lamstare/utils/
python ./submit_cronjob.py
