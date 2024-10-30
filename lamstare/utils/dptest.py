import json
from pathlib import Path
from typing import Dict, Optional
import subprocess
import os
import yaml
from dotenv import load_dotenv # type: ignore
load_dotenv()

temp_file_path = os.environ.get("TEMP_FILE_DIR")
assert temp_file_path is not None, "TEMP_FILE_DIR is not set in .env"

# DP-test related functions
def run_single_head_dptest(exp_path:str, ckpt:int, head:str):
    dptest_res = {}
    run_id=exp_path.split("/")[-1] # Get basename as id
    temp_file_name = f"{run_id}#{ckpt}#{head}"
    try:
        script_path = os.path.join(os.path.dirname(__file__), "single_dptest.sh")
        print(f"Executing command: {script_path} {exp_path} {ckpt} {head}")
        subprocess.run([script_path, exp_path, str(ckpt), head, temp_file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        dptest_res = extract_info_from_dptest_txt(f"{head} ",temp_file_path+temp_file_name+".txt")
        os.remove(temp_file_path+temp_file_name+".txt")
        os.remove(temp_file_path+temp_file_name+".pth")
    except:
        print(f"Fail to test {temp_file_name}")
    return dptest_res


def extract_info_from_dptest_txt(dataset_name:str, filepath:Path|str) -> Dict[str,float]:
    """
    Parse dptest results to a dict

    Parameters:
    ----------
        dataset_name: str
            The name of the dataset being tested on.
        filepath: str
            The path to the dptest output logfile.

    """
    with open(filepath,"r") as f:
        content = f.readlines()

    metrics = {}
    for line in content[-11:-1]:
        line = line.split("deepmd.entrypoints.test")[-1].strip()
        metrics[f"{dataset_name} " + line.split(":")[0].strip()] = float(line.split(":")[-1].strip().split(" ")[0])
    return metrics



def extract_valid_path_from_input(input, output):
    """
    A helper function to prepare dp test files using multitask test set in `input`.

    The paths to the test sets will be in `output#{head}_valid.txt`, one system a line.
    """
    with open(input,"r") as f:
        dd = json.load(f)
    try:
        heads = list(dd['training']['model_prob'].keys())
    except KeyError:
        heads =[""] # single task

    for head in heads:
        if head:
            valid_paths = dd['training']['data_dict'][head]['validation_data']['systems']
        else: # single task
            valid_paths = dd['training']['validation_data']['systems']
        with open(f'{output}#{head}_valid.txt', 'w') as f:
            for path in valid_paths:
                f.write(f"{path}\n")

def extract_ood_test_pth_from_yml(input_file, output_path, overwrite:Optional[bool]=False):
    """
    A helper function to prepare dptest files using filepath provided in the yaml file.
    """
    if not Path(output_path).exists():
        Path(output_path).mkdir()
    with open(input_file,"r") as f:
        yaml_dd = yaml.safe_load(f)
    mapping = {k:v['filepath'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
    for ood_dataset, filepaths in mapping.items():
        tem_output_path = f'{output_path}/{ood_dataset}.txt'
        if Path(tem_output_path).exists() and not overwrite:
            print(f"File exists at {tem_output_path}. Skipping.")
            continue
        print(f"Writing to {tem_output_path}")
        with open(tem_output_path, 'w') as f:
            if filepaths is None:
                raise ValueError(f"Missing filepath for {ood_dataset}")
            for path in filepaths:
                for sys in Path(path).rglob("type_map.raw"):
                    f.write(f"{sys.parent}\n")

def get_head_weights(exp_path) -> Dict[str,float]:
    """
    Extract the model_prob dict from a multitask `input.json`.

    Parameters:
    -----------
        exp_path : str
            Absolute path to the experiment folder.

    """
    with open(exp_path+"/input.json","r") as f:
        dd = json.load(f)
    weights = dd['training']['model_prob']
    return weights
