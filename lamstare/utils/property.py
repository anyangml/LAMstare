import json
import yaml
import os
import shutil
import logging
from lamstare.utils.dptest import extract_info_from_dptest_txt
from pathlib import Path

def get_property_json(params: dict):
    finetune_config = {
        "_comment": "that's all",
        "model":{
            "type_map": params.get("type_map"),
            "descriptor": params.get("descriptor"),
            "fitting_net": {
            "type": "property",
            "intensive": params.get("intensive"),
            "bias_method": "no_bias",
            "task_dim": params.get("property_dim"),
            "neuron": [
                240,
                240,
                240
            ],
            "resnet_dt": True,
            "seed": 1,
            "_comment": " that's all"
            }
        },
        "loss": {
            "type": "property",
            "_comment": " that's all"
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-08,
            "_comment": "that's all"
        },
        "training": {
            "training_data":{
                "systems":params.get("train_path"),
                 "batch_size": "auto:512",
                "_comment": "that's all"
            },
            "validation_data":{
                "systems":params.get("test_path"),
                 "batch_size": 1,
                "_comment": "that's all"
            },
            "numb_steps": params.get("train_steps"),
            "warmup_steps": 0,
            "gradient_max_norm": 5.0,
            "max_ckpt_keep": 10,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": int(params.get("train_steps")//100),
            "save_freq": int(params.get("train_steps")//4),
            "_comment": "that's all",
        }
    }
    return finetune_config

def prepare_property_finetune_folder(pretrain_exp_path:str, task_name:str, step: int, property_yaml:str,output_path:str="finetune"):
    """
    This function prepares all the necessary files to finetune a pre-trained model on a property task.
    It should do the following:
        1. prepare property finetune input.json given a task name
        2. copy pretrain model model.ckpt-xxx.pt to finetune folder
    """
    run_id = pretrain_exp_path.split("/")[-1]
    finetune_path = os.path.join(output_path, f"{run_id}_{task_name}")
    try:
        os.makedirs(finetune_path,exist_ok=False)
    except FileExistsError:
        logging.warning(f"Finetune path {finetune_path} already exists. SKIP.")
        return finetune_path

    # I. prepare property finetune input.json
    with open(property_yaml, 'r') as f:
        tasks = yaml.load(f, Loader=yaml.FullLoader)["TASK_TO_HEAD_MAP"]
    
    with open(os.path.join(pretrain_exp_path,"input.json"), "r") as f:
        pretrain_config = json.load(f)
    
    params  = tasks[task_name]
    if "loss_dict" in pretrain_config: # multitask input
        params.update({
            "descriptor": pretrain_config["model"]["shared_dict"]["dpa2_descriptor"],
            "type_map": pretrain_config["model"]["shared_dict"]["type_map_all"]
        })
    else: #single task input
        raise NotImplementedError

    finetune_config = get_property_json(params)
    
    with open(os.path.join(finetune_path,"input.json"), "w") as f:
        json.dump(finetune_config, f, indent=4)

    # II. copy model.ckpt-xxx.pt to finetune folder
    shutil.copy(f"{pretrain_exp_path}/model.ckpt-{step}.pt", f"{finetune_path}/pretrain-{step}.pt")

    # III prepare test_valid folder
    with open(os.path.join(finetune_path,f"{task_name}_valid.txt"), "w") as f:
        for sys in Path(params.get("test_path")).rglob("type_map.raw"):
            f.write(f"{sys.parent}\n")
    return finetune_path


def run_property_train_test(finetune_path: str, task_name: str, step: int) -> dict:
    """This should be the shell interface that runs finetuning and testing, results extraction"""
    # I. change to finetune folder
    model = f"{task_name}_model.ckpt-{step}.pt"
    try:
        os.chdir(finetune_path)
    except Exception as e:
        print(f"Error changing directory: {e}")
    os.system("pwd")

    # II. Finetune model
    command = (
        f"dp --pt train input.json --finetune pretrain-{step}.pt --skip-neighbor-stat"
    )
    logging.warning(command)
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError(f"Failed to finetune for {model}")

    # III. Freeze model
    command = f"dp --pt freeze -o finetuned_{task_name}.pth"
    logging.warning(command)
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError(f"Failed to freeze model {model}")
    
    # IV. Test model
    test_result = f"{finetune_path.split('/')[-1]}.txt"
    command = f"dp --pt test -m {f'finetuned_{task_name}.pth'} -f {f'{task_name}_valid.txt'} -l {test_result}"
    logging.warning(command)
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError(f"Failed to test model {test_result}")
    
    # V. Extract results
    result = extract_info_from_dptest_txt(task_name, test_result,txt_type="property")
    return result
    
