import json
from typing import Dict, Optional
from pathlib import Path


# DP-test 

def extract_info_from_dptest_txt(dataset_name:str, filepath:str="dptest_res.txt") -> Dict[str,float]:
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



def extract_valid_pth_from_input(input_file, output_pth, overwrite:Optional[bool]=False):
    """
    A helper function to prepare dptest files using multitask validation set in `input.json`.
    """
    with open(input_file,"r") as f:
        dd = json.load(f)
    heads = list(dd['training']['model_prob'].keys())
    for head in heads:
        val_pth = dd['training']['data_dict'][head]['validation_data']['systems']
        tem_output_path = f'{output_pth}/{head}.txt'
        if Path(tem_output_path).exists() and not overwrite:
            print(f"File exists at {tem_output_path}. Skipping.")
            continue
        with open(tem_output_path, 'w') as f:
            for pth in val_pth:
                f.write(f"{pth}\n")


def extract_ood_test_pth_from_yml(input_file, output_pth, overwrite:Optional[bool]=False):
    """
    A helper function to prepare dptest files using filepath provided in the yaml file.
    """
    with open(mapping_path,"r") as f:
        yaml_dd =  yaml.safe_load(f)
    mapping = {k:v['filepath'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
    # for head in heads:
    #     val_pth = dd['training']['data_dict'][head]['validation_data']['systems']
    #     with open(f'{output_pth}/{head}.txt', 'w') as f:
    #         for pth in val_pth:
    #             f.write(f"{pth}\n")
    return mapping


def get_head_weights(run_pth) -> Dict[str,float]:
    """
    Extract the model_prob dict from a multitask `input.json`.

    Parameters:
    -----------
        run_pth : str
            Absolute path to the experiment folder.

    """
    with open(run_pth+"/input.json","r") as f:
        dd = json.load(f)
    weights = dd['training']['model_prob']
    return weights