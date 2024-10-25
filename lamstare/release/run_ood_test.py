import yaml
from typing import Optional, Tuple, Dict
from lamstare.utils.dptest import extract_ood_test_pth_from_yml

def get_ood_to_head_map(mapping_path: str = "OOD_DATASET.yml", overwrite:Optional[bool]=False) -> Tuple[Dict[str, str], bool]:
    """
    Load the OOD test input yaml file, prepare dptest files, return a `OOD dataset -> task head` mapping.

    Args:
        mapping_path    : str
            Path to the YAML file containing the mapping {OOD: {head: HEAD_NAME, datapath: PATHs}}.

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
    mapping = extract_ood_test_pth_from_yml(mapping_path, overwrite=overwrite)


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

def prepare_test_file(exp_path: str)->None:
    """
    This should be a function that generates the test file from the yml file.
    """
    pass



def main(exp_path: str, ckpt: Optional[int] = None, is_multitask: bool = True)->None:
    if ckpt is not None:
        pass #call get latest ckpt
    else:
        ckpt = ckpt

    # generate test file from input.json

    if is_multitask:
        pass #call test multitask run_ood_test_multitask
    else:
        pass #call test single task run_ood_test_single_task

if __name__ == "__main__":
    # main("", is_multitask=False)
    print(get_ood_to_head_map())

