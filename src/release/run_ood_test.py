import yaml
from typing import Optional

def get_ood_to_head_map(mapping_path: str = "OOD_DATASET.yml") -> dict:
    """
    Get the mapping between OOD and head names from a YAML file.

    Args:
        mapping_path (str): Path to the YAML file containing the mapping {OOD: {head: HEAD_NAME, datapath: PATH}}.

    Returns:
        dict: A dictionary mapping OOD names to head names.

    """
    with open(mapping_path,"r") as f:
        map = yaml.safe_load(f)["OOD_TO_HEAD_MAP"]
    return map

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
    map = 



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
    main("", is_multitask=False)