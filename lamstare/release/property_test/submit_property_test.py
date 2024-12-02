import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from lamstare.infra.property_database import PropertyRecord
from lamstare.utils.dlc_submit import query_job_numbers, submit_job_to_dlc




def submit_property_test(
    exp_path: str,
    model_version: str,
    step: Optional[int] = None,
    is_multitask: bool = True,
    mapping_path: str = "OOD_DATASET.yml",
    output_path: str = "./testood",
    overwrite: Optional[bool] = False,
) -> None:
    """Submit OOD test jobs to DLC.

    Usage:

    ```
    source /mnt/data_nas/cc/.bashrc
    export PYTHONPATH=/mnt/data_nas/cc/LAMstare_new:$PYTHONPATH
    cd /mnt/data_nas/cc/LAMstare_new/lamstare/release

    export CONDA_ENV=openlam_db # OPTIONAL; default: ood_test_res
    python3 submit_ood_test.py /mnt/data_nas/public/multitask/training_exps/1103_linear_fitting_medium_l8_atton_37head_tanh_40GPU_bs_auto128
    ```
    """
    # generate test file from input.json
    mapping, is_multitask = get_ood_to_head_map(
        mapping_path=mapping_path, output_path=output_path, overwrite=overwrite
    )  # also generates the dp test input files
    if step is None:
        step = get_latest_ckpt(exp_path)
    if is_multitask:
        avaliable_heads_in_model = set(get_head_weights(exp_path).keys())
        heads_needed = set(mapping.values())
        assert heads_needed.issubset(
            avaliable_heads_in_model
        ), f"Missing heads in model: {heads_needed - avaliable_heads_in_model}"

    USE_DLC = True
    for ood_dataset, head in reversed(mapping.items()):
        testfile = Path(output_path).resolve() / f"{ood_dataset}.txt"
        run_id = exp_path.split("/")[-1]  # Get basename as id
        run_name = f"{run_id}#{step}#{ood_dataset}#{head}"  # the index to identify each run in DB
        record_count = len(OODRecord.query_by_name(run_name=run_name))
        if record_count == 0:
            if USE_DLC:
                command = (
                    ". /mnt/data_nas/public/.bashrc; "
                    f"conda activate /mnt/data_nas/public/Miniconda/envs/{os.environ.get('CONDA_ENV','lamstare')};"
                    "export PYTHONPATH=/mnt/data_nas/cc/LAMstare_new;"
                    f"cd {Path(__file__).resolve().parent} ; "
                    f"python3 run_ood_test.py {exp_path} {ood_dataset} {head} {model_version} {step} {testfile} {run_name}"
                ).replace("workspace","data_nas")
                job_name = f"OOD-{run_name}"
                if query_job_numbers(job_name):
                    logging.warning(f"SKIPPED: {job_name} is already running.")
                else:
                    logging.debug(f"Job command: \n{command}")
                    submit_job_to_dlc(job_name, command)
                    logging.info(f"SUBMITTED: {run_name}")
                # Check submission: https://pai.console.aliyun.com/?regionId=cn-beijing&workspaceId=177142#/dlc/jobs
            else:
                run_ood_test(
                    exp_path,
                    ood_dataset,
                    head,
                    model_version,
                    step,
                    testfile,
                    run_name,
                )
                # os.system(command) # Optional: run locally
        elif record_count == 1:
            logging.info(f"SKIPPED: {run_name} already exists.")
        else:
            logging.error(f"ERROR: {run_name} has multiple records, please check.")


def main(exp_path: str, freq: int = 200000, step: Optional[int] = None):
    if step is None:
        step = find_ckpt_to_test_cron(exp_path, freq, OODRecord)
    if step is not None:
        print(f"Running DPTEST for {exp_path} on ckpt-{step}...\n")
        submit_ood_test(
            exp_path=exp_path,
            model_version="autotest",
            mapping_path=os.path.dirname(__file__) + "/../release/OOD_DATASET.yml",
            step=step,
            is_multitask=False,
        )
    else:
        print("No new ckpt to test.\n")


if __name__ == "__main__":
    path = sys.argv[1]
    logging.basicConfig(level=logging.DEBUG)
    submit_ood_test(path, "b4_release", is_multitask=False)