from lamstare.utils.table import main
import yaml
import os

with open(f"{os.path.dirname(__file__)}/OOD_DATASET.yml","r") as f:
    yaml_dd =  yaml.safe_load(f)

WEIGHTS = {k:v['weights'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
WEIGHTS_V = {k:v['weights'] if v['test_virial'] else 0 for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}

if __name__ == "__main__":
    exp_paths = [

    ]

    main(exp_paths, WEIGHTS, WEIGHTS_V)