from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import dpdata
import glob
from pathlib import Path
from lamstare.infra.ood_database import OODRecord

def run_ase_dptest(
        calc,
        testpath: str,
        ood_name: str,
):
    adptor = AseAtomsAdaptor()

    energy_err = []
    energy_pre = []
    energy_lab = []
    atom_num = []
    energy_err_per_atom = []
    force_err = []
    virial_err = []
    virial_err_per_atom = []
    max_ele_num = 120

    systems = []
    for path in testpath: 
        systems.extend(glob.glob(f"{path}/*"))
    # check if the system is mixed type
    if len(glob.glob(systems[0] + '/**/real_atom_types.npy', recursive=True)) == 0:
        mix_type = False
    else:
        mix_type = True

    
    for filepth in systems:
        if mix_type:
            sys = dpdata.MultiSystems()
            sys.load_systems_from_file(filepth, fmt='deepmd/npy/mixed')
        else:
            sys = dpdata.LabeledSystem(filepth, fmt='deepmd/npy')
        
        for ls in sys:
            for frame in ls:
                try:
                    atoms = frame.to_ase_structure()[0]
                    atoms.calc = calc
                    ff = atoms.get_forces()

                    atomic_numbers = atoms.get_atomic_numbers()
                    atom_num.append(np.bincount(atomic_numbers, minlength=max_ele_num))
                    energy_predict = np.array(atoms.get_potential_energy())
                    energy_pre.append(energy_predict)
                    energy_lab.append(frame.data["energies"])
                    energy_err.append(energy_predict - frame.data["energies"])
                    print(energy_err)
                    force_err.append(frame.data["forces"].squeeze(0) - np.array(ff))
                    energy_err_per_atom.append(energy_err[-1]/force_err[-1].shape[0])
                    try:
                        stress = atoms.get_stress()
                        stress_tensor = - np.array(
                        [[stress[0],stress[5],stress[4]],
                        [stress[5],stress[1],stress[3]],
                        [stress[4],stress[3], stress[2]]]
                        )* atoms.get_volume()
                        virial_err.append(frame.data['virials'] - stress_tensor)
                        virial_err_per_atom.append(virial_err[-1]/force_err[-1].shape[0])
                    except:
                        pass
                except:
                    pass
        
    
    atom_num = np.array(atom_num)
    energy_err = np.array(energy_err)
    energy_pre = np.array(energy_pre)
    energy_lab = np.array(energy_lab)
    shift_bias, _, _, _ = np.linalg.lstsq(atom_num, energy_err, rcond=1e-10)
    unbiased_energy = energy_pre - (atom_num @ shift_bias.reshape(max_ele_num, -1)).reshape(-1) - energy_lab.squeeze()
    unbiased_energy_err_per_a = unbiased_energy / atom_num.sum(-1)
    
    
    res = {
    "Energy MAE": [np.mean(np.abs(np.stack(unbiased_energy)))],
    "Energy RMSE": [np.sqrt(np.mean(np.square(unbiased_energy)))],
    "Energy MAE/Natoms": [np.mean(np.abs(np.stack(unbiased_energy_err_per_a)))],
    "Energy RMSE/Natoms":  [np.sqrt(np.mean(np.square(unbiased_energy_err_per_a)))],
    "Force MAE": [np.mean(np.abs(np.concatenate(force_err)))],
    "Force RMSE": [np.sqrt(np.mean(np.square(np.concatenate(force_err))))],
    }
    if virial_err_per_atom != []:
        res.update(
            {
        "Virial MAE": [np.mean(np.abs(np.stack(virial_err)))],
        "Virial RMSE": [np.sqrt(np.mean(np.square(np.stack(virial_err))))],
        "Virial MAE/Natoms": [np.mean(np.abs(np.stack(virial_err_per_atom)))],
        "Virial RMSE/Natoms": [np.sqrt(np.mean(np.square(np.stack(virial_err_per_atom))))]}
        )
    print(res)
    return res

def main(model_name, testpath_mapping):
    if model_name == "MACE":
        from mace.calculators import mace_mp
        CALC = mace_mp(model="medium", device="cuda", default_dtype="float64")
    elif model_name == "DP":
        from deepmd.calculator import DP
        CALC = DP("/mnt/data_nas/penganyang/mptrj1113.pth")
    elif model_name == "Orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orbff = pretrained.orb_v2(device="cuda") # orb-v2-20241011.ckpt
        CALC = ORBCalculator(orbff, device="cuda")
    elif model_name == "7net":
        from sevenn.sevennet_calculator import SevenNetCalculator
        CALC = SevenNetCalculator("7net-0", device='cuda') 
    elif model_name == "EqV2":
        from fairchem.core import OCPCalculator
        CALC = OCPCalculator(
            model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
            local_cache="pretrained_models",
            cpu=False,
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    for ood_name, testpath in mapping.items():
        print(f"Processing {ood_name}, {testpath}")
        head_dptest_res = run_ase_dptest(CALC, testpath, ood_name)

        if "Virial MAE" not in head_dptest_res:
            head_dptest_res["Virial MAE"] = -1
            head_dptest_res["Virial RMSE"] = -1
            head_dptest_res["Virial MAE/Natoms"] = -1
            head_dptest_res["Virial RMSE/Natoms"] = -1
        OODRecord(
            run_id=f"1114_BASELINE_{model_name}",
            run_name=model_name+"#baseline#"+ood_name,
            model_version="MPtrj",
            ood_dataset=ood_name,
            step=1000000,
            head=model_name,
            energy_mae=head_dptest_res["Energy MAE"],
            energy_rmse=head_dptest_res["Energy RMSE"],
            energy_mae_natoms=head_dptest_res["Energy MAE/Natoms"],
            energy_rmse_natoms=head_dptest_res["Energy RMSE/Natoms"],
            force_mae=head_dptest_res["Force MAE"],
            force_rmse=head_dptest_res["Force RMSE"],
            virial_mae=head_dptest_res["Virial MAE"],
            virial_rmse=head_dptest_res["Virial RMSE"],
            virial_mae_natoms=head_dptest_res["Virial MAE/Natoms"],
            virial_rmse_natoms=head_dptest_res["Virial RMSE/Natoms"],
        ).insert()
        

if __name__ == "__main__":
    import yaml
    with open("/mnt/data_nas/public/multitask/LAMstare/lamstare/release/new_OOD_DATASET.yml","r") as f:
        yaml_dd =  yaml.safe_load(f)
    
    mapping = {k:v['filepath'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
    # main("DP", mapping)
    # main("Orb", mapping)
    # main("MACE", mapping)
    mapping= {"WBM": mapping["WBM"]}
    main("EqV2", mapping)

    

    
