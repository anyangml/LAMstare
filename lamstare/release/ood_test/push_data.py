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
        
        # "/mnt/data_nas/public/multitask/training_exps/1107_shareft_pref0021_1000100_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_240by6_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_pref0021_1000100_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_sharft_lr1e-3_1e-5_medium_l6_atton_37head_tanh_40GPU",
        # "/1114_BASELINE_MACE",
        # "/1114_BASELINE_7net",
        # "/1114_BASELINE_Orb",
        # "/1114_BASELINE_DP",
        "/1114_BASELINE_Mattersim",
        # "/mnt/data_nas/public/multitask/training_exps/1113_shareft_lr1e-3_1e-5_pref0220_10020_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1113_shareft_960by3_lr1e-3_1e-5_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1116_shareft_960by3_lr1e-3_1e-5_medium_l6_atton_37head_tanh_8GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1116_shareft_960by3_lr1e-3_1e-5_medium_l6_atton_37head_tanh_120GPU",
        # "/mnt/workspace/public/multitask/training_exps/0920_dpa2_large_28head_MPTraj_gpu8_10M",
        # "/mnt/workspace/public/multitask/training_exps/0920_dpa2_medium_28head_MPTraj_gpu8_10M",
        # "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_960by3_baseline",
        # "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_480by3",
        # "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_240by6",
        # "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_240by3_large_descp",
        # "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_240by3",
        # "/mnt/data_nas/public/multitask/training_exps/1122_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_240by3_single_192_48_12",
        # "/mnt/data_nas/public/multitask/training_exps/1122_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_240by3_single_192_48_32",
        # "/mnt/data_nas/public/multitask/training_exps/1122_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_240by3_single_384_96_24",

    ]

    main(exp_paths, WEIGHTS_E, WEIGHTS_F, WEIGHTS_V)