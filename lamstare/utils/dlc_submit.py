import subprocess
from dotenv import load_dotenv
import os
import logging
import yaml

load_dotenv()

def query_job_numbers(job_name:str):

    cmd = f"export PATH=/root/.volc/bin/:$PATH; volc ml_task list  --status Queue,Staging,Running -n {job_name.replace('#', '井')} --output json"
    logging.debug(f"Querying job status with command: \n{cmd}")
    ret = subprocess.run(cmd, shell=True, check=False, text=True, capture_output=True)
    logging.info(f"Command output: \n{ret.stdout}")
    if "没有匹配条件的任务" in ret.stdout:
        logging.debug(f"No jobs found for {job_name}.")
        return False
    elif job_name.replace('#', '井') not in ret.stdout:
        logging.debug(f"Job {job_name} not found in the output.")
        return False
    return True

def submit_job_to_dlc(job_name:str, command:str):
    template = yaml.safe_load(open("/aisi/public/multitask/LAMstare/logs/job_template.yaml", "r"))
    template["TaskName"] = job_name.replace("#", "井")
    template["Description"] = ""
    template["ResourceQueueID"] = "q-20250618190306-vzjfq"
    template.pop("UserCodePath", None)  # Remove UserCodePath if it exists
    template.pop("ActiveDeadlineSeconds", None)  # Remove ActiveDeadlineSeconds if it exists
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

    # combined_command = ("ln -s  /aisi /mnt/data_nas\n"
    #         f"{command}")

    template['Entrypoint'] = command
    yaml_file = f"/aisi/public/multitask/LAMstare/logs/{job_name}.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(template, f)
    cmd = f"export PATH=/root/.volc/bin/:$PATH; volc ml_task submit -c {yaml_file}"
    try:
        ret = subprocess.run(cmd, shell=True, check=False, text=True, capture_output=True).stdout
        job_id = ret.split("task_id=")[-1].strip()
        logging.info(f'Job submitted : {job_id}')
    except:
        logging.warning('Warning: ret = subprocess.run has failed')
    
if __name__ == "__main__":
    query_job_numbers("OOD-250618_openlam_v2_fparam_test#200000#AQM-sol#OMol25")