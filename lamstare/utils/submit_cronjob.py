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
        "/mnt/data_nas/public/multitask/training_exps/0415_DPA3.1_3M_new_dptest_drug_omat24":400000,
    }.items():
        # main_ind(exp_path, freq, step)
        main_ood(exp_path, freq, step)
