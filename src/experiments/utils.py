
import json
from pathlib import Path
import pandas as pd
import numpy as np
import os
from typing import List, Dict
import requests
from requests_toolbelt import MultipartEncoder
from dotenv import load_dotenv

load_dotenv()
app_id = os.environ.get("FEISHU_APP_ID")
app_secret = os.environ.get("FEISHU_APP_SECRET")
bot_url = os.environ.get("FEISHU_BOT_URL")

# Plotting
def fetch_lcurve(run_pth):
    """
    Given the absolute path to the experiment folder, return the dataframe containing learning curve information. 
    """

    logfile = Path(os.path.join(run_pth, "lcurve.out"))
    assert Path(logfile).exists(), f"No Such Experiment: {run_pth}."

    df = pd.read_csv(logfile, sep="\s+", header=0,skiprows=[1])
    df.columns = list(df.columns[1:]) + list(df.columns[0])
    df = df.iloc[:,:-1]
    return df


def get_tat_token():
    """
    Get TenantAccessToken from Feishu.
    """
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    data = {
        "app_id": app_id, 
        "app_secret": app_secret,
    }
    response = requests.request("POST", url, headers=headers, json=data)
    response = json.loads(response.text)
    return response["tenant_access_token"]


def uploadImage(token, file):
    """
    Upload an image to Feishu App to generate image_key for POST.
    """
    url = "https://open.feishu.cn/open-apis/im/v1/images"
    form = {"image_type": "message", "image": (open(file, "rb"))}  
    multi_form = MultipartEncoder(form)
    headers = {
        "Authorization": f"Bearer {token}", 
    }
    headers["Content-Type"] = multi_form.content_type
    response = requests.request("POST", url, headers=headers, data=multi_form)
    response = json.loads(response.text)
    return response["data"]["image_key"]


def send2bot(img_keys):
    """
    Send POST request to Feishu bot with all the image keys.
    """
    content = []
    for key in img_keys:
        node = {"tag": "img", "image_key": key}
        content.append(node)
    url = bot_url
    headers = {"Content-Type": "application/json"}
    data = {
        "msg_type": "post",
        "content": {
            "post": {"zh_cn": {"title": "Experiment Tracking", "content": [content]}}
        },
    }
    response = requests.request("POST", url, headers=headers, json=data)


def sendimg(files: List[str]):
    """
    Given a list of file path, send these images to Feishu bot.
    """
    img_keys = []
    token = get_tat_token()
    for file in files:
        img_keys.append(uploadImage(token, file))
    send2bot(img_keys)


# DP-test 

def extract_info_from_dptest_txt(dataset_name:str, filepath:str="dptest_res.txt") -> Dict[str,float]:
    """
    Parse dptest results to a dict

    Parameters:
    ----------
        dataset_name: str
            The name of the dataset being tested on.
        
        filepath: str
            The path to the dptest output logfile.

    """
    with open(filepath,"r") as f:
        content = f.readlines()

    metrics = {}
    for line in content[-11:-1]:
        line = line.split("deepmd.entrypoints.test")[-1].strip()
        metrics[f"{dataset_name} " + line.split(":")[0].strip()] = float(line.split(":")[-1].strip().split(" ")[0])
    return metrics



def extract_valid_pth_from_input(input_file, output_pth):
    """
    A helper function to prepare dptest files using multitask validation set in `input.json`.
    """
    with open(input_file,"r") as f:
        dd = json.load(f)
    heads = list(dd['training']['model_prob'].keys())
    for head in heads:
        val_pth = dd['training']['data_dict'][head]['validation_data']['systems']
        with open(f'{output_pth}/{head}.txt', 'w') as f:
            for pth in val_pth:
                f.write(f"{pth}\n")


def get_head_weights(run_pth) -> Dict[str,float]:
    """
    Extract the model_prob dict from a multitask `input.json`.

    Parameters:
    -----------
        run_pth : str
            Absolute path to the experiment folder.

    """
    with open(run_pth+"/input.json","r") as f:
        dd = json.load(f)
    weights = dd['training']['model_prob']
    return weights