import json
import yaml
import os

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
            "disp_freq": params.get("train_steps")/100,
            "save_freq": params.get("train_steps")//4,
            "_comment": "that's all",
        }
    }
    return finetune_config

def prepare_property_input(pretrain_exp_path:str,property_yaml:str,finetune_path:str="/tmp"):
    with open(property_yaml, 'r') as f:
        tasks = yaml.load(f, Loader=yaml.FullLoader)["TASK_TO_HEAD_MAP"]
    
    with open(os.path.join(pretrain_exp_path,"input.json"), "r") as f:
        pretrain_config = json.load(f)
    
    for task_name, paras in tasks.items():
        if "loss_dict" in pretrain_config: # multitask input
            paras.update({
                "descriptor": pretrain_config["model"]["shared_dict"]["dpa2_descriptor"],
                "type_map": pretrain_config["model"]["shared_dict"]["type_map_all"]
            })
        else: #single task input
            raise NotImplementedError

        finetune_config = get_property_json(paras)
        os.makedirs(os.path.join(finetune_path, task_name),exist_ok=True)
        with open(os.path.join(f"{finetune_path}/{task_name}","input.json"), "w") as f:
            json.dump(finetune_config, f, indent=4)

        break

def run_property_finetune
prepare_property_inputs("/mnt/data_nas/public/multitask/training_exps/1126_prod_shareft_120GUP_240by3_single_384_96_24", os.path.dirname(__file__) + "/../release/PROPERTY_TEST.yml")