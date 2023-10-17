import os
import requests
import fire
import subprocess
import json
from tqdm import tqdm
from generate import inference
from urllib.parse import unquote
import zipfile
# Function to download a file from Firebase Storage
def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    print("downloading train dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    print("download finished")
    return local_filename

# base inference routine
def inference_model(
    base_model: str = "",
    lora_weights_urls,
    ):
    local_filename = 'lora_weights.zip'
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    download_file(url, local_filename)
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        # Extract all the contents of the ZIP file into the specified directory
        zip_ref.extractall("lora_weights")
    print("starting infernce for model")
    inference(base_model=base_model, lora_weights="lora_weights")
