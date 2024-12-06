import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import json

from lamstare.infra.property_database import PropertyRecord
from lamstare.utils.dlc_submit import query_job_numbers, submit_job_to_dlc
from lamstare.utils.property import prepare_property_finetune_folder

def submit_property_test(
    exp_path: str,
    model_version: str,
    step: int,
    property_yml: str = "PROPERTY_TEST.yml",
    output_path: str = "finetune",
) -> None:

    with open(property_yml, 'r') as f:
        tasks = yaml.load(f, Loader=yaml.FullLoader)["TASK_TO_HEAD_MAP"]

    for task_name in tasks:
        run_id = exp_path.split("/")[-1]  # Get basename as id
        run_name = f"{run_id}#{step}#{task_name}"
        record_count = len(PropertyRecord.query_by_name(run_name=run_name))
        if record_count == 0:

            finetune_path = prepare_property_finetune_folder(exp_path, task_name, step, property_yml, output_path)
            command = (
                ". /mnt/data_nas/public/.bashrc; "
                f"conda activate /mnt/data_nas/public/Miniconda/envs/{os.environ.get('CONDA_ENV','openlam_db')};"
                # "export PYTHONPATH=/mnt/data_nas/cc/LAMstare_new;"
                f"cd {Path(__file__).resolve().parent};"
                f"python3 run_property_test.py {exp_path} {task_name} {model_version} {step} {finetune_path} {run_name}"
            ).replace("workspace","data_nas")
            
            job_name = f"PROPERTY-{run_name}"
            
            if query_job_numbers(job_name):
                logging.warning(f"SKIPPED: {job_name} is already running.")
            else:
                logging.debug(f"Job command: \n{command}")
                submit_job_to_dlc(job_name, command)
                logging.info(f"SUBMITTED: {run_name}")
        elif record_count == 1:
            logging.info(f"SKIPPED: {run_name} already exists.")
        else:
            logging.error(f"ERROR: {run_name} has multiple records, please check.")
        break
        





def main(exp_path: str, step: int):
    
    print(f"Running DPTEST for {exp_path} on ckpt-{step}...\n")
    submit_property_test(
        exp_path=exp_path,
        model_version="autotest",
        step=step,
        property_yml=os.path.dirname(__file__) + "/PROPERTY_TEST.yml",
    )

if __name__ == "__main__":
    main("/mnt/data_nas/public/multitask/training_exps/1126_prod_shareft_120GUP_240by3_single_384_96_24", 1000000)