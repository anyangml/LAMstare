from pathlib import Path
from lamstare.infra import Record
from matplotlib.lines import Line2D
from typing import Dict
from lamstare.utils.plot import fetch_lcurve, sendimg
from lamstare.utils.dptest import get_head_weights
from lamstare.experiments.plt_test import fetch_dptest_res, parse_record_dict
import matplotlib.pyplot as plt
import json
import numpy as np
import os

COLOR = ["red", "yellowgreen","dodgerblue","mediumpurple","orange", "lightsalmon", "hotpink", "tan", "cyan", "navy", "orchid"]


if Path(f"{os.path.dirname(__file__) }/baseline_stat_new.json").exists():
    with open(f"{os.path.dirname(__file__) }/baseline_stat_new.json","r") as f:
        PREVIOUS_BASELINE = json.load(f)
else:
    print("No baseline provided.")
    PREVIOUS_BASELINE = {}


def compare_dptest_mae(exps: Dict[str,str], exp_name:str):
    compare_dptest(exps, exp_name, key="mae")

def compare_dptest_rmse(exps: Dict[str,str], exp_name:str):
    compare_dptest(exps, exp_name, key="rmse")

def compare_dptest(exps: Dict[str,str], exp_name:str, key:str):
    assert_comparable(exps)

    try:
        weights = get_head_weights(list(exps.keys())[0])
        heads_to_plot = list(weights.keys())
    except KeyError:
        heads_to_plot = [""] # single task

    n_heads = len(heads_to_plot)

    if n_heads == 1:
        fig, ax = plt.subplots(1, 3, figsize=(12,5),sharex=True)
        for idx, exp_path in enumerate(list(exps.keys())):
            run_id = exp_path.split("/")[-1]
            all_records = fetch_dptest_res(run_id)
            all_records = parse_record_dict(all_records)
            ax[0].loglog(all_records[""]["step"], all_records[""][f"ener_{key}"],color=COLOR[idx],marker="o")
            ax[0].set_ylabel(f"Energy")

            ax[1].loglog(all_records[""]["step"], all_records[""][f"force_{key}"],color=COLOR[idx],marker="o")
            ax[1].set_ylabel(f"Force")

            ax[2].loglog(all_records[""]["step"], all_records[""][f"virial_{key}"],color=COLOR[idx],marker="o")
            ax[2].set_ylabel(f"Virial")

    else:
        fig, ax = plt.subplots(n_heads+1, 3, figsize=(12,2*n_heads+3),sharex=True)
        for idx, exp_path in enumerate(list(exps.keys())):
            weighted_dptest = []
            run_id = exp_path.split("/")[-1]
            all_records = fetch_dptest_res(run_id)
            all_records = parse_record_dict(all_records)
            for i, head in enumerate(heads_to_plot):
                if head in all_records:
                    ax[i][0].loglog(all_records[head]["step"], all_records[head][f"ener_{key}"],color=COLOR[idx],marker="o")
                    ax[i][0].set_ylabel(f"Energy_{head}")

                    ax[i][1].loglog(all_records[head]["step"], all_records[head][f"force_{key}"],color=COLOR[idx],marker="o")
                    ax[i][1].set_ylabel(f"Force_{head}")

                    ax[i][2].loglog(all_records[head]["step"], all_records[head][f"virial_{key}"],color=COLOR[idx],marker="o")
                    ax[i][2].set_ylabel(f"Virial_{head}")

                    if head in PREVIOUS_BASELINE:
                        ax[i][0].axhline(PREVIOUS_BASELINE[head][f"{key}_e"],color="green", linestyle="-.")
                        ax[i][1].axhline(PREVIOUS_BASELINE[head][f"{key}_f"],color="green", linestyle="-.")
                        ax[i][2].axhline(PREVIOUS_BASELINE[head][f"{key}_v"],color="green", linestyle="-.")

                    weighted_dptest.append(np.array([
                    all_records[head][f"ener_{key}"],
                    all_records[head][f"force_{key}"],
                    all_records[head][f"virial_{key}"],
                    ])*weights[head])
            n_ckpt_to_weight = weighted_dptest[-1].shape[1]
            weighted_dptest = [rr[:,:n_ckpt_to_weight] for rr in weighted_dptest]
            weighted_dptest = np.sum(weighted_dptest, axis=0)/sum(weights.values())
            ax[-1][0].loglog(all_records[head]["step"][:n_ckpt_to_weight], weighted_dptest[0],color=COLOR[idx],marker="o")
            ax[-1][0].set_ylabel(f"Energy_Weighted")

            ax[-1][1].loglog(all_records[head]["step"][:n_ckpt_to_weight], weighted_dptest[1],color=COLOR[idx],marker="o")
            ax[-1][1].set_ylabel(f"Force_Weighted")

            ax[-1][2].loglog(all_records[head]["step"][:n_ckpt_to_weight], weighted_dptest[2],color=COLOR[idx],marker="o")
            ax[-1][2].set_ylabel(f"Virial_Weighted")
    plt.tight_layout()
    fig.subplots_adjust(top=0.98)
    legend_elements = [Line2D([0], [0], color=COLOR[idx], lw=4, label=f"{exps[exp_path]}") for idx, exp_path in enumerate(list(exps.keys()))]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, frameon=False)
    fig.savefig("compare-test.jpg")
    sendimg(["compare-test.jpg"], f"{exp_name}_dptest_{key}")


def compare_lcurve(exps: Dict[str,str], exp_name:str, roll:int=50):
    assert_comparable(exps)

    try:
        heads_to_plot = list(get_head_weights(list(exps.keys())[0]).keys())
    except KeyError:
        heads_to_plot = [""] # single task

    n_heads = len(heads_to_plot)
    fig, ax = plt.subplots(n_heads, 3, figsize=(12,2*n_heads+1),sharex=True)

    if n_heads == 1:
        for idx, exp_path in enumerate(list(exps.keys())):
            mult_hist = fetch_lcurve(exp_path)
            ax[0].loglog(mult_hist["step"], mult_hist[f"rmse_e_trn"].rolling(roll).mean(), linestyle='-',color=COLOR[idx])
            ax[0].loglog(mult_hist["step"], mult_hist[f"rmse_e_val"].rolling(roll).mean(), linestyle='-.',color=COLOR[idx])
            ax[0].set_ylabel(f"rmse_e")

            ax[1].loglog(mult_hist["step"], mult_hist[f"rmse_f_trn"].rolling(roll).mean(), linestyle='-',color=COLOR[idx])
            ax[1].loglog(mult_hist["step"], mult_hist[f"rmse_f_val"].rolling(roll).mean(), linestyle='-.',color=COLOR[idx])
            ax[1].set_ylabel(f"rmse_f")

            ax[2].loglog(mult_hist["step"], mult_hist[f"rmse_v_trn"].rolling(roll).mean(), linestyle='-',color=COLOR[idx])
            ax[2].loglog(mult_hist["step"], mult_hist[f"rmse_v_val"].rolling(roll).mean(), linestyle='-.',color=COLOR[idx])
            ax[2].set_ylabel(f"rmse_v")

    else:
        for idx, exp_path in enumerate(list(exps.keys())):
            mult_hist = fetch_lcurve(exp_path)
            for i, head in enumerate(heads_to_plot):
                ax[i][0].loglog(mult_hist["step"], mult_hist[f"rmse_e_trn_{head}"].rolling(roll).mean(), linestyle='-',color=COLOR[idx])
                ax[i][0].loglog(mult_hist["step"], mult_hist[f"rmse_e_val_{head}"].rolling(roll).mean(), linestyle='-.',color=COLOR[idx])
                ax[i][0].set_ylabel(f"rmse_e_{head}")

                ax[i][1].loglog(mult_hist["step"], mult_hist[f"rmse_f_trn_{head}"].rolling(roll).mean(), linestyle='-',color=COLOR[idx])
                ax[i][1].loglog(mult_hist["step"], mult_hist[f"rmse_f_val_{head}"].rolling(roll).mean(), linestyle='-.',color=COLOR[idx])
                ax[i][1].set_ylabel(f"rmse_f_{head}")

                ax[i][2].loglog(mult_hist["step"], mult_hist[f"rmse_v_trn_{head}"].rolling(roll).mean(), linestyle='-',color=COLOR[idx])
                ax[i][2].loglog(mult_hist["step"], mult_hist[f"rmse_v_val_{head}"].rolling(roll).mean(), linestyle='-.',color=COLOR[idx])
                ax[i][2].set_ylabel(f"rmse_v_{head}")


                if head in PREVIOUS_BASELINE:
                    ax[i][0].axhline(PREVIOUS_BASELINE[head]["rmse_e"],color="green", linestyle="-.")
                    ax[i][0].axhline(PREVIOUS_BASELINE[head]["e_std"],color="purple", linestyle="-.")
                    ax[i][1].axhline(PREVIOUS_BASELINE[head]["rmse_f"],color="green", linestyle="-.")
                    ax[i][1].axhline(PREVIOUS_BASELINE[head]["f_std"],color="purple", linestyle="-.")
                    ax[i][2].axhline(PREVIOUS_BASELINE[head]["rmse_v"],color="green", linestyle="-.")
                    ax[i][2].axhline(PREVIOUS_BASELINE[head]["v_std"],color="purple", linestyle="-.")

    plt.tight_layout()
    fig.subplots_adjust(top=0.98)
    legend_elements = [Line2D([0], [0], color=COLOR[idx], lw=4, label=f"{exps[exp_path]}") for idx, exp_path in enumerate(list(exps.keys()))]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, frameon=False)
    fig.savefig("compare-lcurve.jpg")
    sendimg(["compare-lcurve.jpg"], exp_name)

def assert_comparable(exps: Dict[str,str]):
    unique_heads = set()
    assert len(exps) > 1, "No Experiments to compare, only get 1 experiment."
    for exp_path in exps.keys():
        try:
            heads = "@@".join(list(get_head_weights(exp_path).keys()))
        except KeyError:
            heads = "@@".join([""]) # single task

        unique_heads.add(heads)
    if len(unique_heads) > 1:
        raise ValueError("Please check the experiments provided are comparable.")

def main(exps: Dict[str,str]):
    compare_lcurve(exps)
    compare_dptest(exps)

if __name__ == "__main__":
    exps = {
        "/mnt/data_nas/public/multitask/training_exps/1103_linear_fitting_medium_l8_atton_37head_tanh_40GPU_bs_auto128":"layer8-bs128",
        "/mnt/data_nas/public/multitask/training_exps/1103_shallow_fitting_medium_l6_atton_37head_tanh_40GPU_bs_auto256":"layer6-bs256-fitting32",
        "/mnt/data_nas/public/multitask/training_exps/1103_linear_fitting_medium_l6_atton_37head_tanh_40GPU_bs_auto256": "bs256-baseline",
        "/mnt/data_nas/public/multitask/training_exps/1031_shareft_medium_l6_atton_37head_tanh_40GPU": "bs256-sharefitting",
        "/mnt/data_nas/public/multitask/training_exps/1018_b4_medium_l6_atton_37head_linear_fitting_tanh":"bs256-baseline-5M"
        # "/mnt/data_nas/penganyang/experiments/1029_omat_batch128_medium_test":""
    }
    compare_lcurve(exps, "Compare 37head")
    # compare_dptest_mae(exps, "Compare 37head")
    compare_dptest_rmse(exps, "Compare 37head-IDTest")