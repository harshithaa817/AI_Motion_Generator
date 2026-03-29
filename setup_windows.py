import os
import zipfile
import gdown

def download_and_extract(url, zip_name, extract_to, output_name=None):
    if not os.path.exists(zip_name):
        print(f"Downloading {zip_name} from {url}...")
        gdown.download(url, zip_name, fuzzy=True)
    else:
        print(f"{zip_name} already exists. Skipping download.")
        
    print(f"Extracting {zip_name} to {extract_to}...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    if output_name:
        print(f"Finished extracting for {output_name}.")

def main():
    # 1. Download SMPL files (download_smpl_files.sh)
    os.makedirs('body_models', exist_ok=True)
    os.chdir('body_models')
    download_and_extract("https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2", "smpl.zip", ".", "smpl")
    os.chdir('..')

    # 1.5 Download GloVe (prepare/download_glove.sh)
    print("Downloading GloVe...")
    download_and_extract("https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view", "glove.zip", ".", "glove")

    # 2. Download T2M evaluators (download_t2m_evaluators.sh)
    # The script downloaded t2m and kit evaluators
    download_and_extract("https://drive.google.com/file/d/1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb/view", "t2m.zip", ".", "t2m")
    download_and_extract("https://drive.google.com/file/d/12liZW5iyvoybXD8eOw4VanTgsMtynCuU/view", "kit.zip", ".", "kit")

    # Ensure dataset directories exist and have Mean/Std
    os.makedirs('dataset/HumanML3D', exist_ok=True)
    # Copy Mean/Std if they exist in t2m/ kit/ or dataset/
    # In this repo, they seem to be in dataset/t2m_mean.npy etc.
    # We might need to symlink or copy them to where HumanML3D expects them.

    # 3. Download pretrained model for text-to-motion
    # [NEW!] humanml_trans_enc_512_bert-50steps: 1z5IW5Qa9u9UdkckKylkcSXCwIYgLPhIC
    # I'll download this fast 50-step model to save time during inference and eval.
    os.makedirs('save/my_humanml_trans_enc_512_bert-50steps', exist_ok=True)
    os.chdir('save/my_humanml_trans_enc_512_bert-50steps')
    print("Downloading 50-steps MDM model...")
    # wait, the google drive link from README is: 1z5IW5Qa9u9UdkckKylkcSXCwIYgLPhIC
    # But let's download the original one as fallback just in case: 1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821
    # We will use the smaller 50steps one to save space and time.
    download_url = "https://drive.google.com/file/d/1z5IW5Qa9u9UdkckKylkcSXCwIYgLPhIC/view"
    zip_name = "humanml_trans_enc_512_bert-50steps.zip"
    if not os.path.exists(zip_name):
        gdown.download(download_url, zip_name, fuzzy=True)
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(".")
    os.chdir('../..')

    print("Setup done!")

if __name__ == '__main__':
    main()
