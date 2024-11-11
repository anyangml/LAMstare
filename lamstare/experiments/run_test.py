import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv  # type: ignore
load_dotenv()

from lamstare.infra import Record
from lamstare.utils.dlc_submit import query_job_numbers, submit_job_to_dlc
from lamstare.utils.dptest import extract_valid_path_from_input, get_head_weights

# Deprecated
temp_file_path = os.environ.get("TEMP_FILE_DIR")
assert temp_file_path is not None, "TEMP_FILE_DIR is not set in .env"


def submit_ind_test(exp_path: str, step: int):
    run_id = exp_path.split("/")[-1]  # Get basename as id
    try:
        heads = list(get_head_weights(exp_path).keys())
    except KeyError:
        heads = [""]  # single task

    for head in heads:
        run_name = f"{run_id}#{step}#{head}"
        record_count = len(Record.query_by_name(run_name=run_name))
        if record_count == 0:
            testfile = extract_valid_path_from_input(exp_path, head)
            job_name = f"AUTOTEST_{run_name}"
            command = (
                f"#!/bin/bash \n"
                f". /mnt/data_nas/public/.bashrc \n"
                f"conda activate /mnt/data_nas/public/Miniconda/envs/{os.environ.get('CONDA_ENV','lamstare')} \n"
                f"export PYTHONPATH=/mnt/data_nas/cc/LAMstare_new \n"
                f"cd {Path(__file__).resolve().parent} \n"
                f"python3 run_ind_test.py {exp_path} {head} {step} {testfile} {run_name} \n"
            )
            logging.debug(f"Job command: \n{command}")
            if query_job_numbers(job_name):
                logging.warning(f"SKIPPED: {job_name} is already running.")
            else:
                submit_job_to_dlc(job_name, command)
        elif record_count == 1:
            logging.info(f"SKIPPED: {run_name} already exists.")
        else:
            logging.error(f"ERROR: {run_name} has multiple records, please check.")


def find_ckpt_to_test_cron(exp_path: str, freq: int) -> Optional[int]:
    ckpt_pth = Path(exp_path)
    with open(ckpt_pth / "checkpoint", "r") as f:
        latest_ckpt = int(f.readlines()[0].split("-")[1].split(".")[0])
    # Get exp_path abs path
    exp_path = str(ckpt_pth.resolve())
    # Get basename
    run_id = exp_path.split("/")[-1]
    if len(Record.query(run_id=run_id)) > 0:
        previous_tested_step = int(Record.query(run_id=run_id)[-1].step)  # type: ignore
    else:
        previous_tested_step = 0
    print(
        f"Latest ckpt tested: {previous_tested_step}, Latest ckpt available: {latest_ckpt}\n"
    )
    if latest_ckpt >= previous_tested_step + freq:
        need_to_test = previous_tested_step + freq
    else:
        need_to_test = None
    return need_to_test


def main(exp_path: str, freq: int = 200000):
    need_to_test = find_ckpt_to_test_cron(exp_path, freq)
    if need_to_test is not None:
        print(f"Running DPTEST for {exp_path} on ckpt-{need_to_test}...\n")
        submit_ind_test(exp_path, need_to_test)
    else:
        print("No new ckpt to test.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1])
    # main("/mnt/workspace/cc/multitask/training_exps/1015_37head_multitask_1gpu_test") # multi task test data
    # main("/mnt/workspace/penganyang/experiments/1018_mptrj_l6_atton_b256_test") # single task test data
