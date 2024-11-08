import logging
import os
from pathlib import Path
import sys
import yaml
from lamstare.infra.ood_database import OODRecord
from lamstare.release.run_ood_test import run_ood_test
from lamstare.utils.dlc_submit import query_job_numbers, submit_job_to_dlc
from lamstare.utils.dptest import (
    extract_ood_test_pth_from_yml,
    get_head_weights,
)
from typing import Optional, Tuple, Dict


def get_ood_to_head_map(
    mapping_path: str = "OOD_DATASET.yml",
    output_path: str = "./testood",
    overwrite: Optional[bool] = False,
) -> Tuple[Dict[str, str], bool]:
    """
    Load the OOD test input yaml file, prepare dptest files, return a `OOD dataset -> task head` mapping, and save the dptest input files as {dataset_name}.txt under `output_path`.

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

    with open(mapping_path, "r") as f:
        yaml_dd = yaml.safe_load(f)
    is_multitask = yaml_dd.get("IS_MULTI_TASK", False)
    mapping = {k: v["head"] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}

    # check if all OOD datasets have a head
    if is_multitask:
        n_missing_heads = []
        for k, v in mapping.items():
            if v is None:
                n_missing_heads.append(k)
        assert (
            len(n_missing_heads) == 0
        ), f"Missing heads for OOD datasets: {n_missing_heads}"
    else:
        mapping = {k: "" for k in mapping.keys()}

    # prepare dptest files
    extract_ood_test_pth_from_yml(mapping_path, output_path, overwrite=overwrite)
    return mapping, is_multitask


def get_latest_ckpt(exp_path: str) -> int:
    """
    Get the latest checkpoint number from the experiment folder.
    """
    with open(exp_path + "/checkpoint", "r") as f:
        lastest_ckpt = int(f.readlines()[0].split("-")[1].split(".")[0])
    return lastest_ckpt


def main(
    exp_path: str,
    model_version: str,
    step: Optional[int] = None,
    is_multitask: bool = True,
    mapping_path: str = "OOD_DATASET.yml",
    output_path: str = "./testood",
    overwrite: Optional[bool] = False,
) -> None:

    # generate test file from input.json
    mapping, is_multitask = get_ood_to_head_map(
        mapping_path=mapping_path, output_path=output_path, overwrite=overwrite
    )  # also generates the dp test input files
    if step is None:
        step = get_latest_ckpt(exp_path)
    is_multitask = False
    if is_multitask:
        avaliable_heads_in_model = set(get_head_weights(exp_path).keys())
        heads_needed = set(mapping.values())
        assert heads_needed.issubset(
            avaliable_heads_in_model
        ), f"Missing heads in model: {heads_needed - avaliable_heads_in_model}"

    USE_DLC = False
    USE_DLC = True
    for ood_dataset, head in reversed(mapping.items()):
        testfile = Path(output_path) / f"{ood_dataset}.txt"
        run_id = exp_path.split("/")[-1]  # Get basename as id
        run_name = f"{run_id}#{step}#{ood_dataset}#{head}"  # we don't change it since it is used as the key in DB
        record_count = len(OODRecord.query_by_name(run_name=run_name))
        if record_count == 0:
            if USE_DLC:
                command = (
                    ". /mnt/data_nas/public/.bashrc; "
                    "conda activate /mnt/data_nas/public/Miniconda/envs/lamstare;"
                    "export PYTHONPATH=/mnt/data_nas/cc/LAMstare_new;"
                    f"cd {Path(__file__).resolve().parent} ; "
                    f"python3 run_ood_test.py {exp_path} {ood_dataset} {head} {model_version} {step} {testfile} {run_name}"
                )
                logging.debug(f"Job command: \n{command}")
                # os.system(command) # debug
                job_name=f"TEST-{run_id}-{ood_dataset}"
                if query_job_numbers(job_name):
                    logging.warning(f"SKIPPED: {job_name} is already running.")
                else:
                    submit_job_to_dlc(job_name,command)
            else:
                run_ood_test(
                    exp_path,
                    ood_dataset,
                    head,
                    model_version,
                    step,
                    testfile,
                    run_name,
                )
        elif record_count == 1:
            logging.info(f"SKIPPED: {run_name} already exists.")
        else:
            logging.error(f"ERROR: {run_name} has multiple records, please check.")


if __name__ == "__main__":
    path = sys.argv[1]
    # e.g. "/mnt/data_nas/public/multitask/training_exps/1103_shallow_fitting_medium_l6_atton_37head_tanh_40GPU_bs_auto256"
    logging.basicConfig(level=logging.DEBUG)
    main(path, "b4_release", is_multitask=False)
