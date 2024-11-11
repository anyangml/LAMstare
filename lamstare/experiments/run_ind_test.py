from pathlib import Path
import numpy as np

from lamstare.infra.trn_database import Record
from lamstare.utils.dptest import run_dptest


def run_ind_test(exp_path: str, head: str, step: int, testfile: Path, run_name: str):
    run_id = exp_path.split("/")[-1]  # Get basename as id
    checkpoint_path = Path(f"{exp_path}/model.ckpt-{step}.pt")
    head_dptest_res = run_dptest(checkpoint_path, head, testfile)
    print(head_dptest_res)
    if np.isnan(head_dptest_res[f"{head} Virial MAE"]):
        head_dptest_res[f"{head} Virial MAE"] = -1
        head_dptest_res[f"{head} Virial RMSE"] = -1
        head_dptest_res[f"{head} Virial MAE/Natoms"] = -1
        head_dptest_res[f"{head} Virial RMSE/Natoms"] = -1
    Record(
        run_id=run_id,
        run_name=run_name,
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
    ).insert()


if __name__ == "__main__":
    import sys

    run_ind_test(
        exp_path=sys.argv[1],
        head=sys.argv[2],
        step=int(sys.argv[3]),
        testfile=Path(sys.argv[4]),
        run_name=sys.argv[5],
    )
