import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from dotenv import load_dotenv

from lamstare.utils.dlc_submit import submit_job_to_dlc

load_dotenv()

# Deprecated
try:
    temp_file_path = os.environ["TEMP_FILE_DIR"]
    os.mkdir(temp_file_path)
except KeyError:
    raise EnvironmentError("TEMP_FILE_DIR is not set in .env")
except FileExistsError:
    pass


def run_dptest(
    ckpt: Path, # Path to the .pt model checkpoint FILE
    head: str,
    testfile: Path,
    ood_name: str = "",
):
    model:Path = ckpt
    temp_path = Path("/tmp")
    # We use /tmp as a local working directory, which is not shared across nodes and automatically cleaned up on job finishes.

    # I. Change bias for OOD dataset
    if ood_name:
        model = temp_path / f"change-bias-{ckpt.name}"
        command = (
            f"dp --pt change-bias {ckpt} -o {model} -f {testfile} --model-branch {head}"
        )
        logging.warning(command)
        ret = os.system(command)
        if ret != 0:
            raise RuntimeError(f"Failed to change bias for {model}")
    # II. Freeze model
    frozen_model = temp_path / model.with_suffix(".pth").name
    command = f"dp --pt freeze -c {model} -o {frozen_model} {f'--head {head}' if head else ''}"
    logging.warning(command)
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError(f"Failed to freeze model {model}")
    # III. Test model
    test_result = frozen_model.parent / f"{frozen_model.name}.txt"
    command = f"dp --pt test -m {frozen_model} -f {testfile} -l {test_result}"
    logging.warning(command)
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError(f"Failed to test model {frozen_model}")
    result = extract_info_from_dptest_txt(head, test_result)
    return result


## DEPRECATED

# def run_single_head_dptest(exp_path:str, ckpt:int, head:str, test_file:Optional[str], ood_name:str) -> Dict[str,float]:
#     logging.error("This function is deprecated. Please use run_dptest instead.")
#     dptest_res = {}
#     run_id=exp_path.split("/")[-1] # Get basename as id
#     temp_file_name = f"{run_id}#{ckpt}#{head}"
#     temp_file_dir=temp_file_path+run_id+ood_name+"/"

#     try:
#         result_file = temp_file_dir+temp_file_name+".txt"
#         print(result_file)
#         if not os.path.exists(result_file):
#             os.mkdir(temp_file_dir)
#             script_path = os.path.join(os.path.dirname(__file__), "single_dptest.sh")
#             args = [script_path, exp_path, str(ckpt), head, temp_file_dir, test_file]
#             command = (
#                 f'". /mnt/data_nas/public/.bashrc;'
#                 f"conda activate /mnt/data_nas/public/Miniconda/envs/{os.environ.get('CONDA_ENV','lamstare')};"
#                 f"cd /mnt/data_nas/cc/LAMstare/lamstare/release;"
#                 f"export TEMP_FILE_DIR={temp_file_dir};"
#             )
#             command += " ".join(args)
#             command += '"'
#             print("Executing command: ", command)
#             submit_job_to_dlc(f"OOD_TEST_{ood_name}_{run_id}", command)
#             return None
#             # print("Executing command: ", *args)
#             # subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#         else:
#             # check if "weighted average of errors" is in the file
#             with open(result_file, "r") as f:
#                 content = f.read()
#             if "weighted average of errors" in content:
#                 print(f"job {temp_file_dir} already finished")
#                 dptest_res = extract_info_from_dptest_txt(head, result_file)
#                 # shutil.rmtree(temp_file_dir)
#                 return dptest_res
#             else:
#                 print(f"job {temp_file_dir} ongoing")
#                 return None
#     except FileExistsError:
#         print(f"job {temp_file_dir} already submitted")
#         return None
#     except:
#         print(f"Fail to test {temp_file_dir}")


def extract_info_from_dptest_txt(dataset_name:str, filepath:Path|str, txt_type:str="standard") -> Dict[str,float]:
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
    
    if txt_type == "standard":
        metrics = {}
        for line in content[-11:-1]:
            line = line.split("deepmd.entrypoints.test")[-1].strip()
            metrics[f"{dataset_name} " + line.split(":")[0].strip()] = float(line.split(":")[-1].strip().split(" ")[0])
    elif txt_type == "property":
        metrics = {}
        metrics["PROPERTY MAE"] = float(content[-3].split(":")[-1].split("units")[0].strip())
        metrics["PROPERTY RMSE"] = float(content[-2].split(":")[-1].split("units")[0].strip())
    else:
        raise ValueError(f"Unknown dptest output type: {txt_type}")
    return metrics


def extract_valid_path_from_input(exp_path:str, head:str) -> Path:
    input=Path(exp_path) / "input.json"
    with open(input,"r") as f:
        dd = json.load(f)
    try:
        heads = list(dd['training']['model_prob'].keys())
        assert heads is not None, "Multitask model but head is empty."
        assert head in heads, f"Head {head} not found in model."
    except KeyError:
        heads =[""] # single task
        assert head == "", "Single task model but head is not empty."

    if head:
        valid_paths = dd['training']['data_dict'][head]['validation_data']['systems']
    else: # single task
        valid_paths = dd['training']['validation_data']['systems']

    output_file=Path(f"{exp_path}/test_valid/{head}_valid.txt")
    logging.debug(f"Writing to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for path in valid_paths:
            f.write(f"{path}\n")
    return output_file

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
            logging.debug(f"File exists at {tem_output_path}. Skipping.")
            continue
        logging.info(f"Writing to {tem_output_path}")
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
