from dotenv import load_dotenv  # type: ignore
from lamstare.utils.plot import get_tat_token
import requests
from datetime import datetime
import json
from lamstare.infra.ood_database import OODRecord
import numpy as np
from typing import List, Dict, Any
import os

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
            "Update time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **data,
        }
    }
    if method == "post":
        resp = requests.post(record_url, json=data, headers=headers)
    elif method == "put":
        resp = requests.put(record_url, json=data, headers=headers)


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


def fetch_ood_res(exp_path: str, weights: dict, weights_v: dict) -> dict:
    run_id = exp_path.split("/")[-1]  # Get basename as id
    runs = OODRecord.query_by_run(run_id)
    # update query to return best among heads
    if len(runs) != len(weights):
        print(f"Warning: missing dptest res for {run_id}")

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

    data["weighted_e_rmse"] = cal_weighted_log_mean(rmse_e, weights)
    data["weighted_f_rmse"] = cal_weighted_log_mean(rmse_f, weights)
    data["weighted_v_rmse"] = cal_weighted_log_mean(rmse_v, weights_v)
    return data


def cal_weighted_log_mean(rmses: dict, weights: dict):
    rmses = [rmses.get(k, np.nan) for k in weights]
    weights = [weights[k] for k in weights]
    rmses, weights = np.array(rmses, dtype=np.float64).reshape(
        -1,
    ), np.array(weights, dtype=np.float64).reshape(
        -1,
    )

    mask = ~np.isnan(rmses)
    rmses, weights = rmses[mask], weights[mask]
    assert rmses.shape == weights.shape

    weighted_log_values = np.dot(weights, np.log(rmses)) / np.sum(weights)
    weighted_log_mean = np.exp(weighted_log_values)
    return weighted_log_mean

def main(exp_paths:List[str], weights:dict, weights_v:dict):
    run2record = fet_records_from_table()
    for exp_path in exp_paths:
        data = fetch_ood_res(exp_path, weights, weights_v)
        run_id=exp_path.split("/")[-1]
        if run_id in run2record:
            method = "put"
        else:
            method = "post"
        send2table(data, run_id, record_id=run2record.get(run_id), method=method)