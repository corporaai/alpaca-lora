import os
import requests
import fire
import subprocess
import json
from tqdm import tqdm
from generate import inference
from urllib.parse import unquote

# Function to download a file from Firebase Storage
def download_file(url, download_folder):
    # Extract the original filename from the URL and decode it
    filename = unquote(url.split('%24')[-1].split('?')[0])
    filepath = os.path.join(download_folder, filename)

    # Send a GET request to download the file
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the file to the specified folder
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {filename}")

# base inference routine
def inference_model(
    base_model: str = "",
    lora_weights_urls,
    ):
    download_folder = 'lora_weights'
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    for url in lora_weights_urls:
        download_file(url, download_folder)
    print("starting infernce for model")
    inference(base_model=base_model, lora_weights="lora_weights")
