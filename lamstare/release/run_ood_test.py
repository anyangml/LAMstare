import yaml
from typing import Optional, Tuple, Dict
from lamstare.utils.dptest import extract_ood_test_pth_from_yml

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

def run_ood_test_multitask(exp_path: str, ckpt: Optional[int] = None)->None:
    """
    This should be a loop through all heads and call run_ood_test_single_task.
    """
    pass



def main(
        exp_path: str, 
        ckpt: Optional[int] = None, 
        is_multitask: bool = True, 
        mapping_path: str = "OOD_DATASET.yml", 
        output_path:str = "./testood", 
        overwrite:Optional[bool]=False
    )->None:

    # generate test file from input.json
    mapping, is_multitask = get_ood_to_head_map()
    
    if ckpt is None:
        pass #call get latest ckpt
    else:
        ckpt = ckpt

    if is_multitask:
        # assert all heads are available in ckpt 
        pass #call test multitask run_ood_test_multitask
    else:
        pass #call test single task run_ood_test_single_task

if __name__ == "__main__":
    # main("", is_multitask=False)
    pass

