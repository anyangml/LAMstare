import subprocess
from pathlib import Path
from dotenv import load_dotenv # type: ignore
import os
import sys
load_dotenv()

def submit_dptest_job_to_dlc(job_name:str):
    docker_image = os.environ.get("DOCKER_IMAGE")
    data_sources = os.environ.get("DATA_SOURCES")
    workspace_id = os.environ.get("WORKSPACE_ID")
    resource_id = os.environ.get("RESOURCE_ID")
    venv = os.environ.get("VENV")
    lamstare_path = os.environ.get("ROOT_DIR")
    priority = 1
    worker_count = 1
    worker_gpu = 1
    worker_cpu = 12
    worker_memory = 120
    job_name = f"AUTOTEST_{job_name}"

    command = f"#!/bin/bash \n" \
            f". /mnt/data_nas/public/.bashrc \n" \
            f"conda activate /mnt/data_nas/public/Miniconda/envs/{venv} \n" \
            f"cd {lamstare_path} \n" \
            f"python lamstare/experiments/run_dptest.py \n" \
            

    cmd = ['/mnt/data_nas/penganyang/dlc', 'submit', 'pytorchjob']
    cmd += ["--name", job_name]
    cmd += ["--worker_cpu", f'{worker_cpu}']
    cmd += ["--worker_gpu", f'{worker_gpu}']
    cmd += ["--worker_memory", f'{worker_memory}Gi']
    cmd += ["--worker_shared_memory", f'{worker_memory}Gi']
    cmd += ["--worker_image", docker_image]
    cmd += ["--command", f"{command}"]
    cmd += ["--data_sources", f'{data_sources}']
    cmd += ["--priority", f'{priority}']
    cmd += ["--workers", f'{worker_count}']
    cmd += ["--workspace_id", f'{workspace_id}']
    cmd += ["--resource_id", f'{resource_id}']

    try:
        subprocess.run(cmd)
    except:
        raise RuntimeError(f"{job_name} failed")
    
if __name__ == "__main__":
    submit_dptest_job_to_dlc(sys.argv[1])