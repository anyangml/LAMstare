import logging
from lamstare.release.ood_test.submit_ood_test import main as main_ood
from lamstare.experiments.run_test import main as main_ind

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    freq = 400000
    step = 500000
    # export CONDA_ENV=openlam_db
    # os.environ["CONDA_ENV"] = "openlam_db"
    for exp_path, freq in {
        # "/mnt/data_nas/public/multitask/training_exps/1126_prod_shareft_120GUP_240by3_single_384_96_24": 400000,
        # "/mnt/data_nas/public/multitask/training_exps/0103_dpa3a_shareft_restart_rc6_120_arc_4_30_l6_120GPU_240by3_384_96_32_comp1":200000,
        # "/mnt/data_nas/public/multitask/training_exps/0105_dpa3a_shareft_384_96_32_scp1_e1a_tanh_rc6_120_arc_4_30_l6_120GPU_240by3":400000,
        # "/mnt/data_nas/public/multitask/training_exps/0115_dpa3a_shareft_128_64_32_scp1_e1a_tanh_rc6_120_arc_4_30_l6_64GPU_240by3_float32":400000,
        # "/mnt/data_nas/public/multitask/training_exps/1225_dpa3a_shareft_rc6_120_arc_4_30_l6_120GPU_240by3_384_96_32_comp1":400000,
        # "/mnt/workspace/public/multitask/training_exps/N0130_dpa3a_shareft_128_64_32_scp1_e1a_cdsilu10_rc6_120_arc_4_30_l6_64GPU_240by3_float32":200000,
        "/mnt/workspace/public/multitask/training_exps/0202_dpa3a_shareft_256_128_32_scp1_e1a_csilu10_rc6_120_arc_4_30_l9_104GPU_240by3":200000,
        

    }.items():
        main_ind(exp_path, freq, step)
        main_ood(exp_path, freq, step)
