from dotenv import load_dotenv  # type: ignore
from lamstare.utils.plot import get_tat_token
import requests
from datetime import datetime
from collections import defaultdict
import json
from lamstare.infra.property_database import PropertyRecord
import numpy as np
from typing import List, Dict, Any
import os
import yaml
from pathlib import Path
import warnings

load_dotenv()

APP_TOKEN = "PpyPbMuxval4histOfDcjSoxnhc"
TABLE_ID = "tbl1rU5Md3tZx9cF"


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
    print(res)


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

def merge_similar_keys(data):
    grouped = defaultdict(list)
    
    # Iterate through the keys in the dictionary
    for key in data.keys():
        # Extract the prefix (everything before '_fold' or '_mae')
        prefix = key.split('_fold')[0] if '_fold' in key else key.split('_mae')[0]
        grouped[f"{prefix}_mae"].append(data[key])
    
    return grouped

def fetch_ood_res(exp_path: str) -> dict:
    run_id = exp_path.split("/")[-1]  # Get basename as id
    runs = PropertyRecord.query_by_run(run_id)
    data = {}
    columns = fetch_column()
    columns = [dd['field_name'] for dd in columns ]
    for run in runs:
        data[f"{run.task_name}_mae"] = (
            run.property_mae if run.property_mae != -1 else np.nan
        )

    # process batch average
    avg_fold_data = merge_similar_keys({k:v for k, v in data.items()})
    avg_fold_data = {k: np.mean(v) for k, v in avg_fold_data.items()}

    return avg_fold_data



def main(exp_paths:List[str]):
    run2record = fet_records_from_table()
    for exp_path in exp_paths:
        data = fetch_ood_res(exp_path)
        
        run_id=exp_path.split("/")[-1]
        if run_id in run2record:
            method = "put"
        else:
            method = "post"
        send2table(data, run_id, record_id=run2record.get(run_id), method=method)



def fetch_column():
    url  = (
        "https://open.feishu.cn/open-apis/bitable/v1/apps/%s/tables/%s/fields"
        % (APP_TOKEN, TABLE_ID)
    )
    headers = {
        "Authorization": f"Bearer {get_tat_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }
    resp = requests.get(url, headers=headers)
    res = json.loads(resp.text)['data']['items']
    return res

def delete_column():
    res = fetch_column()
    for fid in res:
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
    cur_columns = [col['field_name'] for col in fetch_column()]
    with open(Path(os.path.dirname(__file__)).parent / "release/property_test/PROPERTY_TEST.yml","r") as f:
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
    if data['field_name'] not in cur_columns:
        resp = requests.post(url, json=data,headers=headers)


    for ood in yaml_dd["TASK_TO_HEAD_MAP"]:
        data = {
            "field_name": f"{ood}_mae",
            "type": 2,
            "property": {
                "formatter": "0.0000"
            }
        }
        if data['field_name'] not in cur_columns:
            resp = requests.post(url, json=data,headers=headers)


if __name__ == "__main__":
    # fetch_column()
    # delete_column()
    # add_column()
    # push_weights()
    main([
        "/0105_dpa3a_shareft_384_96_32_scp1_e1a_tanh_rc6_120_arc_4_30_l6_120GPU_240by3",
        # "/1126_prod_shareft_120GUP_240by3_single_384_96_24",
        # "/1225_dpa3a_shareft_rc6_120_arc_4_30_l6_120GPU_240by3_384_96_32_comp1",
        # "/prod_shareft_120GUP_240by3_single_384_96_24",
        # "/dpa3a_shareft_rc6_120_arc_4_30_l6_120GPU_240by3_384_96_32_comp1",
        # "/dpa3a_shareft_384_96_32_scp1_e1a_tanh_rc6_120_arc_4_30_l6_120GPU_240by3"
        ])
    pass
