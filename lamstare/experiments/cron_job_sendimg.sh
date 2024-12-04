#!/bin/bash

source /mnt/workspace/public/Miniconda/etc/profile.d/conda.sh
conda activate lamstare
export PYTHONPATH=/mnt/data_nas/cc/LAMstare_new
export CONDA_ENV=openlam_db
python /mnt/workspace/cc/LAMstare_new/lamstare/experiments/plt_ood.py
python /mnt/workspace/cc/LAMstare_new/lamstare/experiments/plt_test.py
