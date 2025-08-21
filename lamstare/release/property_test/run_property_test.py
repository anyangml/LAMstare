import logging
from pathlib import Path
import sys

from lamstare.infra.property_database import PropertyRecord
import numpy as np

from lamstare.utils.property import run_property_train_test
import yaml

def run_property_test(
    exp_path: str,
    task_name,
    model_version: str,
    step: int,
    finetune_path: Path,
    run_name: str,
) -> None:
    run_id = exp_path.split("/")[-1]  # Get basename as id
    head_dptest_res = run_property_train_test(finetune_path, task_name, step)

    print(head_dptest_res)

    PropertyRecord(
        run_id=run_id,
        run_name=run_name,
        model_version=model_version,
        task_name=task_name,
        step=step,
        property_rmse=head_dptest_res["PROPERTY RMSE"],
        property_mae=head_dptest_res["PROPERTY MAE"],
        
    ).insert()


def main():
    
    logging.basicConfig(level=logging.DEBUG)
    exp_path, task_name, model_version, step, finetune_path, run_name = sys.argv[1:]

    run_property_test(
        exp_path, task_name, model_version, int(step), finetune_path, run_name
    )


if __name__ == "__main__":
    main()
