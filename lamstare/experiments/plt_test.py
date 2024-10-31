from lamstare.infra import Record
from lamstare.utils.plot import sendimg
from lamstare.utils.dptest import get_head_weights
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import json

with open("/mnt/data_nas/public/multitask/eval_scripts/baseline_stat.json","r") as f:
    PREVIOUS_BASELINE = json.load(f)


def fetch_dptest_res(run_id:str):
    all_records = defaultdict(dict)

    for rec in Record.query_by_run(run_id):
        all_records[rec.head][rec.step] = {
            "energy_mae":rec.energy_mae,
            "energy_rmse":rec.energy_rmse,
            "energy_mae_natoms":rec.energy_mae_natoms,
            "energy_rmse_natoms":rec.energy_rmse_natoms,
            "force_mae":rec.force_mae,
            "force_rmse":rec.force_rmse,
            "virial_mae":rec.virial_mae if rec.virial_mae != -1 else np.nan,
            "virial_rmse":rec.virial_rmse if rec.virial_rmse != -1 else np.nan,
            "virial_mae_natoms":rec.virial_mae_natoms if rec.virial_mae_natoms != -1 else np.nan,
            "virial_rmse_natoms":rec.virial_rmse_natoms if rec.virial_rmse_natoms != -1 else np.nan,
        }
    return all_records

def parse_record_dict(all_records:dict) -> dict:
    heads = list(all_records.keys())
    parsed_records = {}
    for head in heads:
        temp = {}
        step_list = []
        energy_mae_natoms_list = []
        energy_rmse_natoms_list = []
        force_mae_list = []
        force_rmse_list = []
        virial_mae_natoms_list = []
        virial_rmse_natoms_list = []
        for step, dd in all_records[head].items():
            step_list.append(step)
            energy_mae_natoms_list.append(dd["energy_mae_natoms"])
            energy_rmse_natoms_list.append(dd["energy_rmse_natoms"])
            force_mae_list.append(dd["force_mae"])
            force_rmse_list.append(dd["force_rmse"])
            virial_mae_natoms_list.append(dd["virial_mae_natoms"])
            virial_rmse_natoms_list.append(dd["virial_rmse_natoms"])
        temp["step"] = step_list
        temp["ener_mae"] = energy_mae_natoms_list
        temp["ener_rmse"] = energy_rmse_natoms_list
        temp["force_mae"] = force_mae_list
        temp["force_rmse"] = force_rmse_list
        temp["virial_mae"] = virial_mae_natoms_list
        temp["virial_rmse"] = virial_rmse_natoms_list
        parsed_records[head] = temp
    return parsed_records


def main(exp_path:str):
    run_id=exp_path.split("/")[-1] # Get basename as id
    try:
        weights = get_head_weights(exp_path)
        heads = list(get_head_weights(exp_path).keys())
    except KeyError:
        heads = [""] # single task

    n_heads = len(heads)
    
    all_records = fetch_dptest_res(run_id)
    all_records = parse_record_dict(all_records)
    if n_heads == 1:
        fig, ax = plt.subplots(1, 3, figsize=(12,5),sharex=True)
        ax[0].loglog(all_records[""]["step"], all_records[""]["ener_mae"],"bo-",label="MAE")
        ax[0].loglog(all_records[""]["step"], all_records[""]["ener_rmse"],"ro-",label="RMSE")
        ax[0].set_ylabel(f"Energy")

        ax[1].loglog(all_records[""]["step"], all_records[""]["force_mae"],"bo-",label="MAE")
        ax[1].loglog(all_records[""]["step"], all_records[""]["force_rmse"],"ro-",label="RMSE")
        ax[1].set_ylabel(f"Force")

        ax[2].loglog(all_records[""]["step"], all_records[""]["virial_mae"],"bo-",label="MAE")
        ax[2].loglog(all_records[""]["step"], all_records[""]["virial_rmse"],"ro-",label="RMSE")
        ax[2].set_ylabel(f"Virial")
        fig.suptitle(run_id)
    else:
        fig, ax = plt.subplots(n_heads+1, 3, figsize=(12,2*n_heads+3),sharex=True)
        weighted_dptest = []
        for i, head in enumerate(heads):
            if head in all_records:
                ax[i][0].loglog(all_records[head]["step"], all_records[head]["ener_mae"],"bo-",label="MAE")
                ax[i][0].loglog(all_records[head]["step"], all_records[head]["ener_rmse"],"ro-",label="RMSE")
                ax[i][0].set_ylabel(f"Energy_{head}")

                ax[i][1].loglog(all_records[head]["step"], all_records[head]["force_mae"],"bo-",label="MAE")
                ax[i][1].loglog(all_records[head]["step"], all_records[head]["force_rmse"],"ro-",label="RMSE")
                ax[i][1].set_ylabel(f"Force_{head}")

                ax[i][2].loglog(all_records[head]["step"], all_records[head]["virial_mae"],"bo-",label="MAE")
                ax[i][2].loglog(all_records[head]["step"], all_records[head]["virial_rmse"],"ro-",label="RMSE")
                ax[i][2].set_ylabel(f"Virial_{head}")
                if head in PREVIOUS_BASELINE:
                    ax[i][0].axhline(PREVIOUS_BASELINE[head]["rmse_e"],color="red", linestyle="-.")
                    ax[i][1].axhline(PREVIOUS_BASELINE[head]["rmse_f"],color="red", linestyle="-.")
                    ax[i][2].axhline(PREVIOUS_BASELINE[head]["rmse_v"],color="red", linestyle="-.")
                    if "mae_e" in PREVIOUS_BASELINE[head]:
                        ax[i][0].axhline(PREVIOUS_BASELINE[head]["mae_e"],color="blue", linestyle="-.")
                        ax[i][1].axhline(PREVIOUS_BASELINE[head]["mae_f"],color="blue", linestyle="-.")
                        ax[i][2].axhline(PREVIOUS_BASELINE[head]["mae_v"],color="blue", linestyle="-.")

                weighted_dptest.append(np.array([
                    all_records[head]["ener_mae"],
                    all_records[head]["ener_rmse"],
                    all_records[head]["force_mae"],
                    all_records[head]["force_rmse"],
                    all_records[head]["virial_mae"],
                    all_records[head]["virial_rmse"]
                ])*weights[head])
        n_ckpt_to_weight = weighted_dptest[-1].shape[1]
        weighted_dptest = [rr[:,:n_ckpt_to_weight] for rr in weighted_dptest]
        weighted_dptest = np.sum(weighted_dptest, axis=0)/sum(weights.values())

        ax[-1][0].loglog(all_records[head]["step"][:n_ckpt_to_weight], weighted_dptest[0],"bo-",label="MAE")
        ax[-1][0].loglog(all_records[head]["step"][:n_ckpt_to_weight], weighted_dptest[1],"ro-",label="RMSE")
        ax[-1][0].set_ylabel(f"Energy_Weighted")

        ax[-1][1].loglog(all_records[head]["step"][:n_ckpt_to_weight], weighted_dptest[2],"bo-",label="MAE")
        ax[-1][1].loglog(all_records[head]["step"][:n_ckpt_to_weight], weighted_dptest[3],"ro-",label="RMSE")
        ax[-1][1].set_ylabel(f"Force_Weighted")

        ax[-1][2].loglog(all_records[head]["step"][:n_ckpt_to_weight], weighted_dptest[4],"bo-",label="MAE")
        ax[-1][2].loglog(all_records[head]["step"][:n_ckpt_to_weight], weighted_dptest[5],"ro-",label="RMSE")
        ax[-1][2].set_ylabel(f"Virial_Weighted")

    plt.tight_layout()
    fig.savefig("dptest.jpg")
    sendimg(["dptest.jpg"], run_id)


if __name__ == "__main__":
    for exp_path in ["/mnt/data_nas/penganyang/experiments/1029_omat_batch128_medium_test"]:
        main(exp_path)
    # main("/mnt/data_nas/public/multitask/training_exps/1018_b4_medium_l6_atton_37head_linear_fitting_tanh")
    # main("1015_37head_multitask_1gpu_test")
