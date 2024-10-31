from pathlib import Path
from lamstare.utils.plot import fetch_lcurve, sendimg
from lamstare.utils.dptest import get_head_weights
import matplotlib.pyplot as plt
import json

BASELINE_MAP = {} # Dict[str, str] --> head-name : absolute_path_to_experiment_folder

if Path("baseline_stat.json").exists():
    with open("baseline_stat.json","r") as f:
        PREVIOUS_BASELINE = json.load(f)
else:
    PREVIOUS_BASELINE = {}



def main(exp_path:str, roll:int=50):
    run_id=exp_path.split("/")[-1] # Get basename as id
    try:
        weights = get_head_weights(exp_path)
        heads = list(get_head_weights(exp_path).keys())
    except KeyError:
        heads = [""] # single task

    n_heads = len(heads)
    mult_hist = fetch_lcurve(exp_path)
    fig, ax = plt.subplots(n_heads, 3, figsize=(12,2*n_heads+1),sharex=True)

    if n_heads == 1:
        ax[0].loglog(mult_hist["step"], mult_hist[f"rmse_e_trn"].rolling(roll).mean(), linestyle='-',color="blue")
        ax[0].loglog(mult_hist["step"], mult_hist[f"rmse_e_val"].rolling(roll).mean(), linestyle='-.',color="blue")
        ax[0].set_ylabel(f"rmse_e")

        ax[1].loglog(mult_hist["step"], mult_hist[f"rmse_f_trn"].rolling(roll).mean(), linestyle='-',color="blue")
        ax[1].loglog(mult_hist["step"], mult_hist[f"rmse_f_val"].rolling(roll).mean(), linestyle='-.',color="blue")
        ax[1].set_ylabel(f"rmse_f")

        ax[2].loglog(mult_hist["step"], mult_hist[f"rmse_v_trn"].rolling(roll).mean(), linestyle='-',color="blue")
        ax[2].loglog(mult_hist["step"], mult_hist[f"rmse_v_val"].rolling(roll).mean(), linestyle='-.',color="blue")
        ax[2].set_ylabel(f"rmse_v")
        fig.suptitle(run_id)
    else:
        for i, head in enumerate(heads):
            ax[i][0].loglog(mult_hist["step"], mult_hist[f"rmse_e_trn_{head}"].rolling(roll).mean(), linestyle='-',color="blue")
            ax[i][0].loglog(mult_hist["step"], mult_hist[f"rmse_e_val_{head}"].rolling(roll).mean(), linestyle='-.',color="blue")
            ax[i][0].set_ylabel(f"rmse_e_{head}")

            ax[i][1].loglog(mult_hist["step"], mult_hist[f"rmse_f_trn_{head}"].rolling(roll).mean(), linestyle='-',color="blue")
            ax[i][1].loglog(mult_hist["step"], mult_hist[f"rmse_f_val_{head}"].rolling(roll).mean(), linestyle='-.',color="blue")
            ax[i][1].set_ylabel(f"rmse_f_{head}")

            ax[i][2].loglog(mult_hist["step"], mult_hist[f"rmse_v_trn_{head}"].rolling(roll).mean(), linestyle='-',color="blue")
            ax[i][2].loglog(mult_hist["step"], mult_hist[f"rmse_v_val_{head}"].rolling(roll).mean(), linestyle='-.',color="blue")
            ax[i][2].set_ylabel(f"rmse_v_{head}")

            if head in BASELINE_MAP:
                baseline_hist = fetch_lcurve(BASELINE_MAP[head])
                STEP_NORMAL_PREF = sum(weights.values())/weights[head]*128/120 # need to adjust this value

                ax[i][0].loglog([s * STEP_NORMAL_PREF for s in baseline_hist["step"]], baseline_hist[f"rmse_e_trn"].rolling(1000).mean(), linestyle='-',color="red")
                ax[i][0].loglog([s * STEP_NORMAL_PREF for s in baseline_hist["step"]], baseline_hist[f"rmse_e_val"].rolling(1000).mean(), linestyle='-.',color="red")
                ax[i][1].loglog([s * STEP_NORMAL_PREF for s in baseline_hist["step"]], baseline_hist[f"rmse_f_trn"].rolling(1000).mean(), linestyle='-',color="red")
                ax[i][1].loglog([s * STEP_NORMAL_PREF for s in baseline_hist["step"]], baseline_hist[f"rmse_f_val"].rolling(1000).mean(), linestyle='-.',color="red")
                if "rmse_v_val" in baseline_hist:
                    ax[i][2].loglog([s * STEP_NORMAL_PREF for s in baseline_hist["step"]], baseline_hist[f"rmse_v_trn"].rolling(1000).mean(), linestyle='-',color="red")
                    ax[i][2].loglog([s * STEP_NORMAL_PREF for s in baseline_hist["step"]], baseline_hist[f"rmse_v_val"].rolling(1000).mean(), linestyle='-.',color="red")
            if head in PREVIOUS_BASELINE:
                ax[i][0].axhline(PREVIOUS_BASELINE[head]["rmse_e"],color="green", linestyle="-.")
                ax[i][0].axhline(PREVIOUS_BASELINE[head]["e_std"],color="purple", linestyle="-.")
                ax[i][1].axhline(PREVIOUS_BASELINE[head]["rmse_f"],color="green", linestyle="-.")
                ax[i][1].axhline(PREVIOUS_BASELINE[head]["f_std"],color="purple", linestyle="-.")
                ax[i][2].axhline(PREVIOUS_BASELINE[head]["rmse_v"],color="green", linestyle="-.")
                ax[i][2].axhline(PREVIOUS_BASELINE[head]["v_std"],color="purple", linestyle="-.")

    plt.tight_layout()
    fig.savefig("lcurve.jpg")
    sendimg(["lcurve.jpg"], run_id)

if __name__ == "__main__":
    for exp_path in ["/mnt/data_nas/penganyang/experiments/1029_omat_batch128_medium_test"]:
        main(exp_path)
    # main("/mnt/data_nas/public/multitask/training_exps/1018_b4_medium_l6_atton_37head_linear_fitting_tanh")