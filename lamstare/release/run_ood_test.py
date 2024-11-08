import argparse
import logging
from pathlib import Path
import sys

from lamstare.infra.ood_database import OODRecord
import numpy as np

from lamstare.utils.dptest import run_dptest

def run_ood_test(
    exp_path: str,
    ood_dataset,
    head,
    model_version: str,
    step: int,
    testfile: Path,
    run_name: str,
) -> None:
    run_id = exp_path.split("/")[-1]  # Get basename as id
    checkpoint_path = Path(f"{exp_path}/model.ckpt-{step}.pt")
    head_dptest_res = run_dptest(checkpoint_path, head, testfile, ood_dataset)
    print(head_dptest_res)
    if np.isnan(head_dptest_res[f"{head} Virial MAE"]):
        head_dptest_res[f"{head} Virial MAE"] = -1
        head_dptest_res[f"{head} Virial RMSE"] = -1
        head_dptest_res[f"{head} Virial MAE/Natoms"] = -1
        head_dptest_res[f"{head} Virial RMSE/Natoms"] = -1
    OODRecord(
        run_id=run_id,
        run_name=run_name,
        model_version=model_version,
        ood_dataset=ood_dataset,
        step=step,
        head=head,
        energy_mae=head_dptest_res[f"{head} Energy MAE"],
        energy_rmse=head_dptest_res[f"{head} Energy RMSE"],
        energy_mae_natoms=head_dptest_res[f"{head} Energy MAE/Natoms"],
        energy_rmse_natoms=head_dptest_res[f"{head} Energy RMSE/Natoms"],
        force_mae=head_dptest_res[f"{head} Force  MAE"],
        force_rmse=head_dptest_res[f"{head} Force  RMSE"],
        virial_mae=head_dptest_res[f"{head} Virial MAE"],
        virial_rmse=head_dptest_res[f"{head} Virial RMSE"],
        virial_mae_natoms=head_dptest_res[f"{head} Virial MAE/Natoms"],
        virial_rmse_natoms=head_dptest_res[f"{head} Virial RMSE/Natoms"],
    ) .insert() # FIXME


def main():
    logging.basicConfig(level=logging.DEBUG)
    exp_path, ood_dataset, head, model_version, step, testfile, run_name = sys.argv[1:]
    run_ood_test(exp_path, ood_dataset, head, model_version, int(step), Path(testfile), run_name)

if __name__ == "__main__":
    main()
