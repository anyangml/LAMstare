from dotenv import load_dotenv
import os
import logging
load_dotenv()

def submit_job_to_dlc(job_name:str, command:str):
    docker_image = os.environ["DOCKER_IMAGE"]
    data_sources = os.environ["DATA_SOURCES"]
    workspace_id = os.environ["WORKSPACE_ID"]
    resource_id = os.environ["RESOURCE_ID"]
    priority = 1
    worker_count = 1
    worker_gpu = 1
    worker_cpu = 12
    worker_memory = 120
    cmd = f"/mnt/data_nas/penganyang/dlc submit pytorchjob " \
          f"--name {job_name} " \
          f"--worker_cpu {worker_cpu} " \
          f"--worker_gpu {worker_gpu} " \
          f"--worker_memory {worker_memory}Gi " \
          f"--worker_shared_memory {worker_memory}Gi " \
          f"--worker_image {docker_image} " \
          f"--data_sources {data_sources} " \
          f"--priority {priority} " \
          f"--workers {worker_count} " \
          f"--workspace_id {workspace_id} " \
          f"--resource_id {resource_id} " \
          f"--command '{command}' "
    # cmd += ["--interactive"]
    # cmd += ["--envs", ",".join([f"{k}={v}" for k,v in os.environ.items()])]

    try:
        os.system(cmd)
    except Exception as e:
        logging.error(f"An error occurred while submitting the job: {e}")


def submit_dptest_job_to_dlc(exp_path:str):
    job_name = exp_path.split("/")[-1]

    venv = os.environ.get("VENV")
    lamstare_path = os.environ.get("ROOT_DIR")

    job_name = f"AUTOTEST_{job_name}"

    command = f"#!/bin/bash \n" \
            f". /mnt/data_nas/public/.bashrc \n" \
            f"conda activate /mnt/data_nas/public/Miniconda/envs/{venv} \n" \
            f"cd {lamstare_path} \n" \
            f"python lamstare/experiments/run_test.py {exp_path} \n" \

    submit_job_to_dlc(job_name,command)

if __name__ == "__main__":
    for exp_path in ["/mnt/data_nas/penganyang/experiments/1029_omat_batch128_medium_test"]:
        submit_dptest_job_to_dlc(exp_path) # abosulute path to the experiment folder