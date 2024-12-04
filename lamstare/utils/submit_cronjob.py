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
        
        "/mnt/data_nas/public/multitask/training_exps/1126_prod_shareft_120GUP_240by3_single_384_96_24"
    ]:
        main_ind(exp_path, freq, step)
        main_ood(exp_path, freq, step)
