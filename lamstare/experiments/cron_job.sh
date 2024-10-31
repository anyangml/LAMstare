#!/bin/bash

source /mnt/data_nas/public/Miniconda/etc/profile.d/conda.sh # may need update
conda activate lamsatre # may need update
bash /mnt/data_nas/penganyang/renew_dlc.sh # to renew dlc credential
python /mnt/data_nas/public/multitask/LAMstare/lamstare/utils/dlc_submit.py # may need update --> update to call dlc_submit
