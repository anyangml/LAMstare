import logging
from lamstare.release.ood_test.submit_ood_test import main as main_ood
from lamstare.experiments.run_test import main as main_ind

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    freq = 400000
    step = None
    # export CONDA_ENV=openlam_db
    # os.environ["CONDA_ENV"] = "openlam_db"
    for exp_path, freq in {
        "/aisi/public/multitask/training_exps/250618_openlam_v2_fparam_test":100000,
        "/aisi/public/multitask/training_exps/0415_h20_dpa3a_shareft_nosel_128_64_32_scp1_e1a_csilu3_rc6_arc_4_expsw_l16_128GPU_240by32": 200000,
    }.items():
        # main_ind(exp_path, freq, step)
        main_ood(exp_path, freq, step)
