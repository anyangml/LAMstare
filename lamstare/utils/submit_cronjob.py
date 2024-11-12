import logging
from lamstare.release.submit_ood_test import main as main_ood
from lamstare.experiments.run_test import main as main_ind

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    freq = 200_000
    step = None  # 400_000
    for exp_path in [
        # export CONDA_ENV=openlam_db
        "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_240by6_medium_l6_atton_37head_tanh_40GPU",
        "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_pref0021_1000100_medium_l6_atton_37head_tanh_40GPU",
        "/mnt/data_nas/public/multitask/training_exps/1110_newdata_sharft_lr1e-3_1e-5_medium_l6_atton_37head_tanh_40GPU",
    ]:
        main_ind(exp_path, freq, step)
        main_ood(exp_path, freq, step)
