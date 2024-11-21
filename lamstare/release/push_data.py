from lamstare.utils.table import main
import yaml
import os

with open(os.path.dirname(__file__)+"/OOD_DATASET.yml","r") as f:
    yaml_dd =  yaml.safe_load(f)

WEIGHTS_E = {k:v['energy_weight'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
WEIGHTS_F = {k:v['force_weight'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
WEIGHTS_V = {k:v['virial_weight'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}


if __name__ == "__main__":
    exp_paths = [
        # "/mnt/workspace/penganyang/experiments/0911_gelu_2M_RMSE_lr1e-3_1e-5_pref0220_10020_l6_attoff_batch128_4GPU_v024mixu"
        # "/OOD-dataset-std",
        # "/mnt/data_nas/public/multitask/training_exps/1103_linear_fitting_medium_l8_atton_37head_tanh_40GPU_bs_auto128",
        # "/mnt/data_nas/public/multitask/training_exps/1103_shallow_fitting_medium_l6_atton_37head_tanh_40GPU_bs_auto256",
        # "/mnt/data_nas/public/multitask/training_exps/1103_linear_fitting_medium_l6_atton_37head_tanh_40GPU_bs_auto256",
        # "/mnt/data_nas/public/multitask/training_exps/1031_shareft_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/workspace/public/multitask/training_exps/1018_b4_medium_l6_atton_37head_linear_fitting_tanh",
        # "/mnt/data_nas/public/multitask/training_exps/1107_shareft_pref0021_1000100_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1107_shareft_pref0021_10001_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1107_shareft_240by6_medium_l6_atton_37head_tanh_40GPU"
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_240by6_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_pref0021_1000100_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_sharft_lr1e-3_1e-5_medium_l6_atton_37head_tanh_40GPU",
        # "/1114_BASELINE_MACE",
        # "/1114_BASELINE_7net",
        # "/1114_BASELINE_Orb",
        # "/1114_BASELINE_DP"
        # "/mnt/data_nas/public/multitask/training_exps/1113_shareft_lr1e-3_1e-5_pref0220_10020_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1113_shareft_960by3_lr1e-3_1e-5_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1116_shareft_960by3_lr1e-3_1e-5_medium_l6_atton_37head_tanh_120GPU",
        "/mnt/workspace/public/multitask/training_exps/0920_dpa2_large_28head_MPTraj_gpu8_10M",
        "/mnt/workspace/public/multitask/training_exps/0920_dpa2_medium_28head_MPTraj_gpu8_10M",

    ]

    main(exp_paths, WEIGHTS_E, WEIGHTS_F, WEIGHTS_V)