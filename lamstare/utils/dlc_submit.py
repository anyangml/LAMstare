import subprocess
from dotenv import load_dotenv
import os
import logging
import yaml

load_dotenv()

def query_job_numbers(job_name:str):

    cmd = f"volc ml_task list  --status Queue,Staging,Running -n {job_name} --output json "
    logging.debug(f"Querying job status with command: \n{cmd}")
    ret = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True).stdout
    if "没有匹配条件的任务" in ret:
        logging.debug(f"No jobs found for {job_name}.")
        return 0
    return 1

def submit_job_to_dlc(job_name:str, command:str=None):
    template = yaml.safe_load(open("/aisi/public/multitask/LAMstare/logs/job_template.yaml", "r"))
    template["TaskName"] = job_name
    template["Description"] = ""
    template["ResourceQueueID"] = "q-20250618190306-vzjfq"
    template["Framework"] = "PyTorchDDP"
    template["Storages"] = [
    {
        "Type": "Vepfs",
        "MountPath": "/aisi",
        "VepfsId": "vepfs-cnbj6c0df4e46c0f",
        "SubPath": ""
    }
    ]
    template["TaskRoleSpecs"] = [
    {
        "RoleName":"worker",
        "RoleReplicas": 1,
        "Flavor":"ml.pni2l.3xlarge"
    }
    ]
    template["ImageUrl"] = "dp-ve-registry-cn-beijing.cr.volces.com/aisi/deepmd:0210"

    command = ("ln -s  /aisi /mnt/data_nas\n"
                "source /mnt/data_nas/public/.bashrc\n"
                "conda activate  /mnt/data_nas/public/miniconda3/envs/lamstare\n"
                "nvidia-smi\n\n"
                "set -x\n"
                f"cd /mnt/data_nas/public/multitask/LAMstare/lamstare/utils \n"
                f"echo hellohello > test.txt\n")

    template['Entrypoint'] = command
    yaml_file = f"/aisi/public/multitask/LAMstare/logs/{job_name}.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(template, f)
    cmd = ['volc', 'ml_task', 'submit', '-c', yaml_file]
    try:
        ret = subprocess.check_output(cmd).decode('utf-8')
    except:
        logging.warning('Warning: ret = subprocess.check_output(cmd).decode(\'utf-8\') has failed')
    job_id = ret.split("task_id=")[-1].strip()
    logging.info(f'Job submitted : {job_id}')

if __name__ == "__main__":
    print(submit_job_to_dlc("250618_dpa3_omol25_singletask_96GPU_l24thin"))