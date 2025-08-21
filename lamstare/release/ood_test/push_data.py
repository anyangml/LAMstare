from lamstare.utils.table import main
import yaml
import os

with open(os.path.dirname(__file__)+"/OOD_DATASET_v2.yml","r") as f:
    yaml_dd =  yaml.safe_load(f)

WEIGHTS_E = {k:v['energy_weight'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
WEIGHTS_F = {k:v['force_weight'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
WEIGHTS_V = {k:v['virial_weight'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}


if __name__ == "__main__":
    exp_paths = [
        # "/OOD_STD",
        "/mnt/data_nas/public/multitask/training_exps/1126_prod_shareft_120GUP_240by3_single_384_96_24",
        "/mnt/data_nas/public/multitask/training_exps/1225_dpa3a_shareft_rc6_120_arc_4_30_l6_120GPU_240by3_384_96_32_comp1",
        "/mnt/data_nas/public/multitask/training_exps/0105_dpa3a_shareft_384_96_32_scp1_e1a_tanh_rc6_120_arc_4_30_l6_120GPU_240by3",
        "/250120_BASELINE_Orb",
        "/250120_BASELINE_Mattersim",
        "/250120_BASELINE_MACE",
        "/250120_BASELINE_7net",
        "/250120_BASELINE_EqV2",
        "/250120_BASELINE_DP",
        "/250120_BASELINE_DPA3"

    ]

    main(exp_paths, WEIGHTS_E, WEIGHTS_F, WEIGHTS_V)