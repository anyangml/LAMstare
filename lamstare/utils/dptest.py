import json
from typing import Dict, Optional
from pathlib import Path
import yaml

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



def extract_valid_pth_from_input(input_file, output_path, overwrite:Optional[bool]=False):
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


def extract_ood_test_pth_from_yml(input_file, output_path, overwrite:Optional[bool]=False):
    """
    A helper function to prepare dptest files using filepath provided in the yaml file.
    """
    if not Path(output_path).exists():
        Path(output_path).mkdir()
    with open(input_file,"r") as f:
        yaml_dd =  yaml.safe_load(f)
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


def get_head_weights(run_path) -> Dict[str,float]:
    """
    Extract the model_prob dict from a multitask `input.json`.

    Parameters:
    -----------
        run_path : str
            Absolute path to the experiment folder.

    """
    with open(run_path+"/input.json","r") as f:
        dd = json.load(f)
    weights = dd['training']['model_prob']
    return weights