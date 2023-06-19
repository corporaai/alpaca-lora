from finetune import about_model, train
import request
import fire
import subprocess
def upload_model(model_chkpt):
    print("uploading finetuned model")
    return f"model {model_chkpt} uploaded"
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

def train_model( 
    model_name:str="",
    user_id:str=None,
    dataset_url:str=None
    ):
    if(user_id):
        output_dir = model_name if model_name else "./alpaca-lora-finetuned"
        print("installing dependencies...")
        subprocess.run(["pip", "install", "-r", "requirements.txt"])
        download_file(dataset_url, "dataset.json")
        # train(base_model="decapoda-research/llama-7b-hf", data_path=data_path, output_dir=f"{output_dir}-{user_id}")
        # print("uploading finetuned model to scrol hub")
        # res = uplaod(output_dir)
        # print(res)
    else:
        print("please provide a valid user token")

if __name__ == "__main__":
    fire.Fire(train_model)