from src.utils.dptest import extract_info_from_dptest_txt, get_head_weights
from database import Record
import subprocess
import os
import numpy as np
from pathlib import Path
ROOT_PATH = "/mnt/data_nas/public/multitask/" # may need update


def run_single_head_dptest(run_id:str, ckpt:int, head:str):
    dptest_res = {}
    temp_file_name = f"{run_id}#{ckpt}#{head}"
    try:
        process = subprocess.run([ROOT_PATH+"eval_scripts/single_dptest.sh", run_id, str(ckpt), head], check=True) # may need update
        dptest_res = extract_info_from_dptest_txt(f"{head} ", ROOT_PATH+"eval_scripts/temp_frz_model/"+temp_file_name+".txt") # may need update
        os.remove(ROOT_PATH+"eval_scripts/temp_frz_model/"+temp_file_name+".txt") # may need update
        os.remove(ROOT_PATH+"eval_scripts/temp_frz_model/"+temp_file_name+".pth") # may need update
    except:
        print(f"Fail to test {temp_file_name}")
    return dptest_res

def dptest_one_cpkt_on_all_heads(run_id:str, ckpt: int):
    
    heads = list(get_head_weights(ROOT_PATH +"training_exps/"+ run_id).keys()) # may need update
    for head in heads:
        temp_file_name = f"{run_id}#{ckpt}#{head}"
        if len(Record.query_by_name(run_name=temp_file_name)) == 0:
            head_dptest_res = run_single_head_dptest(run_id, ckpt, head)
            
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
        
        

def find_ckpt_to_test_cron(run_id:str, freq:int=100000):
    ckpt_pth = Path(ROOT_PATH +"training_exps/"+run_id) # may need update
    with open(ROOT_PATH +"training_exps/"+run_id +"/checkpoint","r") as f: # may need update
        lastest_ckpt = int(f.readlines()[0].split("-")[1].split(".")[0])
    
    if len(Record.query(run_id=run_id)) > 0:
        alread_test = int(Record.query(run_id=run_id)[-1].step)
    else:
        alread_test = 0
    print(f"Latest ckpt tested: {alread_test}, Latest ckpt available: {lastest_ckpt}\n")
    if lastest_ckpt >= alread_test + freq:
        need_to_test = alread_test + freq
    else:
        need_to_test = None
    
    return need_to_test
    
def main(run_id:str, freq:int=200000):
    need_to_test = find_ckpt_to_test_cron(run_id, freq)
    if need_to_test is not None:
        print(f"Running DPTEST for {run_id} on ckpt-{need_to_test}...\n")

        dptest_one_cpkt_on_all_heads(run_id, need_to_test)
    else:
        print("No new ckpt to test.\n")
        pass

if __name__ == "__main__":
    main("1018_b4_medium_l6_atton_37head_linear_fitting_tanh")
    
    
