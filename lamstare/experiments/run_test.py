from pathlib import Path
from lamstare.utils.dptest import (
    get_head_weights,
    extract_valid_path_from_input,
    run_single_head_dptest,
)
from lamstare.infra import Record
import numpy as np
from dotenv import load_dotenv # type: ignore
import os
import sys
load_dotenv()

temp_file_path = os.environ.get("TEMP_FILE_DIR")
assert temp_file_path is not None, "TEMP_FILE_DIR is not set in .env"

def dptest_one_cpkt_on_all_heads(exp_path:str, ckpt: int):
    run_id=exp_path.split("/")[-1] # Get basename as id
    try:
        heads = list(get_head_weights(exp_path).keys())
    except KeyError:
        heads = [""] # single task

    extract_valid_path_from_input(exp_path+"/input.json",
        f"{temp_file_path}{run_id}#{ckpt}") # may need a proper teardown

    for head in heads:
        temp_file_name = f"{run_id}#{ckpt}#{head}"
        if len(Record.query_by_name(run_name=temp_file_name)) == 0:
            head_dptest_res = run_single_head_dptest(exp_path, ckpt, head)
            print(head_dptest_res)

            if np.isnan(head_dptest_res[f"{head}  Virial MAE"]):
                head_dptest_res[f"{head}  Virial MAE"] = -1
                head_dptest_res[f"{head}  Virial RMSE"] = -1
                head_dptest_res[f"{head}  Virial MAE/Natoms"] = -1
                head_dptest_res[f"{head}  Virial RMSE/Natoms"] = -1
            Record(
                run_id=run_id,
                run_name=temp_file_name,
                step=ckpt,
                head=head,
                energy_mae=head_dptest_res[f"{head}  Energy MAE"],
                energy_rmse=head_dptest_res[f"{head}  Energy RMSE"],
                energy_mae_natoms=head_dptest_res[f"{head}  Energy MAE/Natoms"],
                energy_rmse_natoms=head_dptest_res[f"{head}  Energy RMSE/Natoms"],
                force_mae=head_dptest_res[f"{head}  Force  MAE"],
                force_rmse=head_dptest_res[f"{head}  Force  RMSE"],
                virial_mae=head_dptest_res[f"{head}  Virial MAE"],
                virial_rmse=head_dptest_res[f"{head}  Virial RMSE"],
                virial_mae_natoms=head_dptest_res[f"{head}  Virial MAE/Natoms"],
                virial_rmse_natoms=head_dptest_res[f"{head}  Virial RMSE/Natoms"],
            ).insert()
        elif len(Record.query_by_name(run_name=temp_file_name)) == 1:
            continue
        else:
            print(f"{temp_file_name} has multiple records, please check!!!")


def find_ckpt_to_test_cron(exp_path:str, freq:int):
    ckpt_pth = Path(exp_path)
    with open(ckpt_pth / "checkpoint","r") as f:
        latest_ckpt = int(f.readlines()[0].split("-")[1].split(".")[0])
    # Get exp_path abs path
    exp_path = str(ckpt_pth.resolve())
    # Get basename
    run_id = exp_path.split("/")[-1]
    if len(Record.query(run_id=run_id)) > 0:
        previous_tested_step = int(Record.query(run_id=run_id)[-1].step)
    else:
        previous_tested_step = 0
    print(f"Latest ckpt tested: {previous_tested_step}, Latest ckpt available: {latest_ckpt}\n")
    if latest_ckpt >= previous_tested_step + freq:
        need_to_test = previous_tested_step + freq
    else:
        need_to_test = None

    return need_to_test

def main(exp_path:str, freq:int=200000):
    need_to_test = find_ckpt_to_test_cron(exp_path, freq)
    if need_to_test is not None:
        print(f"Running DPTEST for {exp_path} on ckpt-{need_to_test}...\n")
        dptest_one_cpkt_on_all_heads(exp_path, need_to_test)
    else:
        print("No new ckpt to test.\n")

if __name__ == "__main__":
    main(sys.argv[1])
    # main("/mnt/workspace/cc/multitask/training_exps/1015_37head_multitask_1gpu_test") # multi task test data
    # main("/mnt/workspace/penganyang/experiments/1018_mptrj_l6_atton_b256_test") # single task test data
