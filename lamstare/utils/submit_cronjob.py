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
        # "/aisi/public/multitask/training_exps/250630_dpa3_openlam_v1_old_weight_8M_compare0415":100000,
        # "/aisi/public/multitask/training_exps/0415_h20_dpa3a_shareft_nosel_128_64_32_scp1_e1a_csilu3_rc6_arc_4_expsw_l16_128GPU_240by3":400000,
        # "/aisi/public/multitask/training_exps/250701_dpa3_openlam_v2_old_weight_8M_l16":100000,
        # "/aisi/public/multitask/training_exps/250701_dpa3_openlam_v2_new_weight_8M_l16":100000,
        # "/aisi/public/multitask/training_exps/250703_dpa3_openlam_v2_old_weight_8M_L16_only_change_omol":100000,
        # "/aisi/public/multitask/training_exps/250714_dpa3_openlam_v2_old_weight_22task":100000,
        # "/aisi/public/multitask/training_exps/250714_dpa3_openlam_v2_old_weight_8M_L16_only_change_omol":100000,
        "/aisi/public/multitask/training_exps/250721_dpa3_openlam_v2_old_weight_22task_deepcsp":100000,
        "/aisi/public/multitask/training_exps/250722_dpa3_openlam_v2_old_weight_22task_deepcsp_mpgen":100000,

        # "/aisi/public/multitask/training_exps/250710_dpa3_openlam_v2_old_weight_8M_L16_remove_mptrj": 100000
    }.items():
        # main_ind(exp_path, freq, step)
        main_ood(exp_path, freq, step)
