#!/bin/bash

source /mnt/data_nas/public/Miniconda/etc/profile.d/conda.sh # may need update
conda activate lamstare # may need update
bash /mnt/data_nas/penganyang/renew_dlc.sh # to renew dlc credential
export PYTHONPATH=/mnt/data_nas/cc/LAMstare_new
export CONDA_ENV=openlam_db
python /mnt/data_nas/cc/LAMstare_new/lamstare/utils/submit_cronjob.py # may need update --> update to call dlc_submit
