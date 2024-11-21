import logging
from lamstare.release.submit_ood_test import main as main_ood
from lamstare.experiments.run_test import main as main_ind

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    freq = 200000
    step = None
    # export CONDA_ENV=openlam_db
    # os.environ["CONDA_ENV"] = "openlam_db"
    for exp_path in [
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_240by6_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_pref0021_1000100_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_sharft_lr1e-3_1e-5_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1113_shareft_960by3_lr1e-3_1e-5_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1113_shareft_lr1e-3_1e-5_pref0220_10020_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1116_shareft_960by3_lr1e-3_1e-5_medium_l6_atton_37head_tanh_8GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1116_shareft_960by3_lr1e-3_1e-5_medium_l6_atton_37head_tanh_120GPU"
        # "/mnt/workspace/public/multitask/training_exps/0920_dpa2_large_28head_MPTraj_gpu8_10M",
        # "/mnt/workspace/public/multitask/training_exps/0920_dpa2_medium_28head_MPTraj_gpu8_10M",
        "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_240by3",
        "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_240by3_large_descp",
        "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_240by6",
        "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_480by3",
        "/mnt/data_nas/public/multitask/training_exps/1119_shareft_lr1e-3_1e-5_pref0021_1000100_24GUP_960by3_baseline"
    ]:
        main_ind(exp_path, freq, step)
        main_ood(exp_path, freq, step)
