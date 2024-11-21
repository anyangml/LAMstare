from dotenv import load_dotenv  # type: ignore
from lamstare.utils.plot import get_tat_token
import requests
from datetime import datetime
import json
from lamstare.infra.ood_database import OODRecord
import numpy as np
from typing import List, Dict, Any
import os
import yaml

load_dotenv()

APP_TOKEN = os.environ.get("TABLE_APP_TOKEN")
TABLE_ID = os.environ.get("TABLE_ID")


def send2table(
    data: Dict[str, Any], run_id: str, record_id=None, method: str = "post"
) -> None:
    """
    This function sends data to the table in Feishu, either by creating a new record or updating an existing record.
    """
    record_url = (
        "https://open.feishu.cn/open-apis/bitable/v1/apps/%s/tables/%s/records"
        % (APP_TOKEN, TABLE_ID)
    )
    if method == "put":
        assert record_id is not None, "record_id is required for put method"
        record_url = f"{record_url}/{record_id}"

    headers = {
        "Authorization": f"Bearer {get_tat_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }

    data = {
        "fields": {
            "Experiment name": run_id,
            "Update Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **{k:v for k,v in data.items() if not np.isnan(v)},
        }
    }
    if method == "post":
        resp = requests.post(record_url, json=data, headers=headers)
    elif method == "put":
        resp = requests.put(record_url, json=data, headers=headers)
    res = json.loads(resp.text)


def fet_records_from_table():
    """
    This function fetches all records from the table in Feishu. returns a dictionary with the experiment name as key and the record_id as value.
    """
    get_record_url = (
        "https://open.feishu.cn/open-apis/bitable/v1/apps/%s/tables/%s/records"
        % (APP_TOKEN, TABLE_ID)
    )
    headers = {
        "Authorization": f"Bearer {get_tat_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }
    resp = requests.get(get_record_url, headers=headers)
    res = json.loads(resp.text)

    if res["data"]["total"] == 0:
        return {}
    return {
        item["fields"]["Experiment name"]: item["record_id"]
        for item in res["data"]["items"]
    }


def fetch_ood_res(exp_path: str, weights_e: dict, weights_f: dict, weights_v: dict) -> dict:
    run_id = exp_path.split("/")[-1]  # Get basename as id
    runs = OODRecord.query_best_by_run(run_id)

    if len(runs) < len(weights_e):
        print(f"Warning: missing dptest res for {run_id}")
    elif len(runs) > len(weights_e):
        print(f"Warning: more dptest res for {run_id}")

    data = {}
    rmse_e = {}
    rmse_f = {}
    rmse_v = {}
    for run in runs:
        data[f"{run.ood_dataset}_e_rmse"] = (
            run.energy_rmse_natoms if run.energy_rmse_natoms != -1 else np.nan
        )
        rmse_e[run.ood_dataset] = data[f"{run.ood_dataset}_e_rmse"]
        data[f"{run.ood_dataset}_f_rmse"] = (
            run.force_rmse if run.force_rmse != -1 else np.nan
        )
        rmse_f[run.ood_dataset] = data[f"{run.ood_dataset}_f_rmse"]
        data[f"{run.ood_dataset}_v_rmse"] = (
            run.virial_rmse_natoms if run.virial_rmse_natoms != -1 else np.nan
        )
        rmse_v[run.ood_dataset] = data[f"{run.ood_dataset}_v_rmse"]

    data["Weighted_e_rmse"] = cal_weighted_log_mean(rmse_e, weights_e)
    data["Weighted_f_rmse"] = cal_weighted_log_mean(rmse_f, weights_f)
    data["Weighted_v_rmse"] = cal_weighted_log_mean(rmse_v, weights_v)


    # path = "/mnt/workspace/public/multitask/LAMstare/ood_data_std_v.txt"
    # with open(path, "r") as f:
    #     contents = f.read()
    # data = {}
    # rmse_e = {}
    # rmse_f = {}
    # rmse_v = {}
    # for rr in contents.split("@")[1:]:
    #     seg = rr.split(",")
    #     ood = seg[0].split(":")[-1].strip()
    #     data[f"{ood}_e_rmse"] = float(seg[1].split(":")[-1].strip())
    #     rmse_e[ood] = data[f"{ood}_e_rmse"]
    #     data[f"{ood}_f_rmse"] = None if ood == "WBM" else float(seg[2].split(":")[-1].strip())
    #     rmse_f[ood] = data[f"{ood}_f_rmse"]
    #     data[f"{ood}_v_rmse"] = None if seg[3].split(":")[-1].strip() == "None" else float(seg[3].split(":")[-1].strip())
    #     rmse_v[ood] = data[f"{ood}_v_rmse"]
    # data["Weighted_e_rmse"] = cal_weighted_log_mean(rmse_e, weights_e)
    # data["Weighted_f_rmse"] = cal_weighted_log_mean(rmse_f, weights_f)
    # data["Weighted_v_rmse"] = cal_weighted_log_mean(rmse_v, weights_v)
    # print(data)
    return data


def cal_weighted_log_mean(rmses: dict, weights: dict):
    rmses = [rmses.get(k, np.nan) for k in weights]
    weights = [weights[k] for k in weights]
    rmses, weights = np.array(rmses, dtype=np.float64).reshape(
        -1,
    ), np.array(weights, dtype=np.float64).reshape(
        -1,
    )

    assert rmses.shape == weights.shape
    weighted_log_rmse = np.log(rmses) * weights
    weighted_log_rmse_mean = np.nanmean(weighted_log_rmse)
    weighted_rmse_mean = np.exp(weighted_log_rmse_mean)
    return weighted_rmse_mean


def main(exp_paths:List[str], weights_e:dict, weights_f:dict, weights_v:dict):
    run2record = fet_records_from_table()

    for exp_path in exp_paths:
        data = fetch_ood_res(exp_path, weights_e, weights_f, weights_v)
        run_id=exp_path.split("/")[-1]
        if run_id in run2record:
            method = "put"
        else:
            method = "post"
        send2table(data, run_id, record_id=run2record.get(run_id), method=method)


def push_weights():
    with open(os.path.dirname(__file__) + "/../release/OOD_DATASET.yml", "r") as f:
        yaml_dd =  yaml.safe_load(f)

    WEIGHTS_E = {k:v['weights_e'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
    WEIGHTS_F = {k:v['weights_f'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
    WEIGHTS_V = {k:v['weights_v'] for k, v in yaml_dd["OOD_TO_HEAD_MAP"].items()}
    data = {
        **{f"{k}_e_rmse":v for k, v in WEIGHTS_E.items()},
        **{f"{k}_f_rmse":v for k, v in WEIGHTS_F.items()},
        **{f"{k}_v_rmse":v for k, v in WEIGHTS_V.items()}
    }
    send2table(data, "Weight")

def delete_column():
    url  = (
        "https://open.feishu.cn/open-apis/bitable/v1/apps/%s/tables/%s/fields"
        % (APP_TOKEN, TABLE_ID)
    )
    headers = {
        "Authorization": f"Bearer {get_tat_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }
    resp = requests.get(url, headers=headers)
    res = json.loads(resp.text)
    for fid in res['data']['items']:
        url = (
            "https://open.feishu.cn/open-apis/bitable/v1/apps/%s/tables/%s/fields/%s"
            % (APP_TOKEN, TABLE_ID, fid['field_id'])
        )
        headers = {
            "Authorization": f"Bearer {get_tat_token()}",
            "Content-Type": "application/json; charset=utf-8",
        }
        resp = requests.delete(url, headers=headers)

def add_column():

    with open(os.path.dirname(__file__) + "/../release/OOD_DATASET.yml","r") as f:
        yaml_dd =  yaml.safe_load(f)

    url  = (
        "https://open.feishu.cn/open-apis/bitable/v1/apps/%s/tables/%s/fields"
        % (APP_TOKEN, TABLE_ID)
    )
    headers = {
        "Authorization": f"Bearer {get_tat_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }

    data = {
                "field_name": "Update Time",
                "type": 5,
            }
    resp = requests.post(url, json=data,headers=headers)


    for ood in yaml_dd["OOD_TO_HEAD_MAP"]:
        data = {
            "field_name": f"{ood}_e_rmse",
            "type": 2,
            "property": {
                "formatter": "0.0000"
            }
        }
        resp = requests.post(url, json=data,headers=headers)
        data = {
            "field_name": f"{ood}_f_rmse",
            "type": 2,
            "property": {
                "formatter": "0.0000"
            }
        }
        resp = requests.post(url, json=data,headers=headers)
        data = {
            "field_name": f"{ood}_v_rmse",
            "type": 2,
            "property": {
                "formatter": "0.0000"
            }
        }
        resp = requests.post(url, json=data, headers=headers)

    data = {
        "field_name": "Weighted_e_rmse",
        "type": 2,
        "property": {
            "formatter": "0.0000"
        }
    }
    resp = requests.post(url, json=data,headers=headers)
    data = {
        "field_name": "Weighted_f_rmse",
        "type": 2,
        "property": {
            "formatter": "0.0000"
        }
    }
    resp = requests.post(url, json=data,headers=headers)
    data = {
        "field_name": "Weighted_v_rmse",
        "type": 2,
        "property": {
            "formatter": "0.0000"
        }
    }
    resp = requests.post(url, json=data, headers=headers)


if __name__ == "__main__":
    # delete_column()
    # add_column()
    # push_weights()
    pass
