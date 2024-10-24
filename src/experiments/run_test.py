from pathlib import Path
from src.utils.dptest import extract_info_from_dptest_txt, get_head_weights
from src.infra import Record
import subprocess
import os
import numpy as np


def run_single_head_dptest(exp_path:str, ckpt:int, head:str):
    dptest_res = {}
    run_id=exp_path.split("/")[-1] # Get basename as id
    temp_file_name = f"{run_id}#{ckpt}#{head}"
    try:
        script_path = os.path.join(os.path.dirname(__file__), "single_dptest.sh")
        print(f"Executing command: {script_path} {exp_path} {ckpt} {head}")
        process = subprocess.run([script_path, exp_path, str(ckpt), head], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        outdir="/mnt/workspace/public/multitask/eval_scripts/temp_frz_model/" # from single_dptest.sh
        dptest_res = extract_info_from_dptest_txt(f"{head} ",outdir+temp_file_name+".txt")
        os.remove(outdir+temp_file_name+".txt")
        os.remove(outdir+temp_file_name+".pth")
    except:
        print(f"Fail to test {temp_file_name}")
        # raise # DEBUG
    return dptest_res

def dptest_one_cpkt_on_all_heads(exp_path:str, ckpt: int):
    run_id=exp_path.split("/")[-1] # Get basename as id
    heads = list(get_head_weights(exp_path).keys())
    for head in heads:
        temp_file_name = f"{run_id}#{ckpt}#{head}"
        if len(Record.query_by_name(run_name=temp_file_name)) == 0:
            head_dptest_res = run_single_head_dptest(exp_path, ckpt, head)

            if np.isnan(head_dptest_res[f"{head}  Virial MAE"]):
                        head_dptest_res[f"{head}  Virial MAE"] = -1
                        head_dptest_res[f"{head}  Virial RMSE"] = -1
                        head_dptest_res[f"{head}  Virial MAE/Natoms"] = -1
                        head_dptest_res[f"{head}  Virial RMSE/Natoms"] = -1
            # print(f"{run_id=}, {head=}, {ckpt=}")
            # continue # DEBUG
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
    ckpt_pth = Path(exp_path) # may need update
    with open(ckpt_pth / "checkpoint","r") as f: # may need update
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

def main(exp_path:str, freq:int=510000):
    need_to_test = find_ckpt_to_test_cron(exp_path, freq)
    if need_to_test is not None:
        print(f"Running DPTEST for {exp_path} on ckpt-{need_to_test}...\n")
        dptest_one_cpkt_on_all_heads(exp_path, need_to_test)
    else:
        print("No new ckpt to test.\n")

if __name__ == "__main__":
    # main("1018_b4_medium_l6_atton_37head_linear_fitting_tanh")
    main("/mnt/workspace/cc/multitask/training_exps/1015_37head_multitask_1gpu_test")