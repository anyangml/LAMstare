import yaml
from typing import Optional, Tuple, Dict
from lamstare.utils.dptest import extract_ood_test_pth_from_yml, get_head_weights
from lamstare.infra.ood_database import OODRecord
import numpy as np

def get_ood_to_head_map(mapping_path: str = "OOD_DATASET.yml", output_path:str = "./testood", overwrite:Optional[bool]=False) -> Tuple[Dict[str, str], bool]:
    """
    Load the OOD test input yaml file, prepare dptest files, return a `OOD dataset -> task head` mapping.

    Args:
        mapping_path    : str
            Path to the YAML file containing the mapping {OOD: {head: HEAD_NAME, datapath: PATHs}}.
        output_path     : str
            Path to the directory where the dptest files will be saved.    

    Returns:
        mapping         : Dict[str, str]
            A dictionary mapping OOD names to head names.
        is_multitask    : bool
            Whether the model is a multitask model or not.
    """

    with open(mapping_path,"r") as f:
        yaml_dd =  yaml.safe_load(f)
    is_multitask =yaml_dd["IS_MULTI_TASK"]
    is_multitask = False
    mapping = {k:v['head'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}

    # check if all OOD datasets have a head
    if is_multitask:
        n_missing_heads = []
        for k, v in mapping.items():
            if v is None:
                n_missing_heads.append(k)
        assert len(n_missing_heads) == 0, f"Missing heads for OOD datasets: {n_missing_heads}"
    else:
        mapping = {k:"" for k in mapping.keys()}

    # prepare dptest files
    extract_ood_test_pth_from_yml(mapping_path,output_path, overwrite=overwrite)
    return mapping, is_multitask


def run_ood_test_single_task(exp_path: str, ckpt: Optional[int] = None, head: Optional[str] = None)->None:
    """
    This should be a reused function for running OOD test for a single task model or single head for a multitask model.
    """
    pass

def run_ood_test(exp_path: str, mapping:dict, model_version:str, ckpt: Optional[int] = None)->None:
    """
    This should be a loop through all heads and call run_ood_test_single_task.
    """
    run_id=exp_path.split("/")[-1] # Get basename as id
    for ood_dataset, head in mapping.items():
        temp_file_name = f"{run_id}#{ckpt}#{ood_dataset}#{head}"
        if len(OODRecord.query_by_name(run_name=temp_file_name)) == 0:
            head_dptest_res = run_ood_test_single_task(exp_path, ckpt, head)

            if np.isnan(head_dptest_res[f"{ood_dataset}  Virial MAE"]):
                        head_dptest_res[f"{ood_dataset}  Virial MAE"] = -1
                        head_dptest_res[f"{ood_dataset}  Virial RMSE"] = -1
                        head_dptest_res[f"{ood_dataset}  Virial MAE/Natoms"] = -1
                        head_dptest_res[f"{ood_dataset}  Virial RMSE/Natoms"] = -1

            OODRecord(
                run_id=run_id, 
                run_name=temp_file_name,
                model_version=model_version,
                ood_dataset=ood_dataset,
                step=ckpt, 
                head=head,
                energy_mae=head_dptest_res[f"{ood_dataset}  Energy MAE"],
                energy_rmse=head_dptest_res[f"{ood_dataset}  Energy RMSE"],
                energy_mae_natoms=head_dptest_res[f"{ood_dataset}  Energy MAE/Natoms"],
                energy_rmse_natoms=head_dptest_res[f"{ood_dataset}  Energy RMSE/Natoms"],
                force_mae=head_dptest_res[f"{ood_dataset}  Force  MAE"],
                force_rmse=head_dptest_res[f"{ood_dataset}  Force  RMSE"],
                virial_mae=head_dptest_res[f"{ood_dataset}  Virial MAE"],
                virial_rmse=head_dptest_res[f"{ood_dataset}  Virial RMSE"],
                virial_mae_natoms=head_dptest_res[f"{ood_dataset}  Virial MAE/Natoms"],
                virial_rmse_natoms=head_dptest_res[f"{ood_dataset}  Virial RMSE/Natoms"],
            ).insert()
        elif len(OODRecord.query_by_name(run_name=temp_file_name)) == 1:
            continue
        else:
            print(f"{temp_file_name} has multiple records, please check!!!")


def get_latest_ckpt(exp_path: str)->int:
    """
    Get the latest checkpoint number from the experiment folder.
    """
    with open(exp_path +"/checkpoint","r") as f: # may need update
        lastest_ckpt = int(f.readlines()[0].split("-")[1].split(".")[0])    
    return lastest_ckpt

def main(
        exp_path: str, 
        model_version: str,
        ckpt: Optional[int] = None, 
        is_multitask: bool = True, 
        mapping_path: str = "OOD_DATASET.yml", 
        output_path:str = "./testood", 
        overwrite:Optional[bool]=False
    )->None:

    # generate test file from input.json
    mapping, is_multitask = get_ood_to_head_map(mapping_path=mapping_path, output_path=output_path, overwrite=overwrite)
    
    if ckpt is None:
        ckpt = get_latest_ckpt(exp_path)
    else:
        ckpt = ckpt

    if is_multitask:
        avaliable_heads_in_model = set(get_head_weights(exp_path).keys())
        heads_needed = set(mapping.values())
        assert heads_needed.issubset(avaliable_heads_in_model), f"Missing heads in model: {heads_needed - avaliable_heads_in_model}"
        run_ood_test(exp_path, mapping, model_version, ckpt)
    else:
       run_ood_test(exp_path, mapping, model_version, ckpt)

if __name__ == "__main__":
    # main("", is_multitask=False)
    pass

