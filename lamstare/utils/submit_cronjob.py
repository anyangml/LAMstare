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
        "/aisi/public/multitask/training_exps/250630_dpa3_openlam_v1_old_weight_8M_compare0415":100000,
        "/aisi/public/multitask/training_exps/250701_dpa3_openlam_v2_old_weight_8M_l16":100000,
        # "/aisi/public/multitask/training_exps/250701_dpa3_openlam_v2_new_weight_8M_l16":100000,
        # "/aisi/public/multitask/training_exps/250625_openlam_v2_fparam_test_old_weight":100000,
    }.items():
        # main_ind(exp_path, freq, step)
        main_ood(exp_path, freq, step)
