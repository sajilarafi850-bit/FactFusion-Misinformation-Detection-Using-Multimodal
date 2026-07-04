import os
import shutil
import zipfile
from huggingface_hub import hf_hub_download

REPO_ID = "Inas-00/factfusion-weights"

# Use absolute paths relative to this file so it works regardless of cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(BASE_DIR, "models")

def download_weights():
    print(f"[*] Initializing connection to Hugging Face: {REPO_ID}...")
    print(f"[*] Target model directory: {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)

    # --- Text_Final_Model: real zip archive (multiple files: config, tokenizer, weights) ---
    text_model_dir = os.path.join(TARGET_DIR, "Text_Final_Model")
    sentinel = os.path.join(text_model_dir, "config.json")
    if os.path.exists(sentinel):
        print("[OK] Text_Final_Model already extracted — skipping download.")
    else:
        print("[+] Pulling Text_Final_Model.zip from Hub...")
        try:
            zip_path = hf_hub_download(
                repo_id=REPO_ID,
                filename="Text_Final_Model.zip",   # ← ye HF repo ke actual filename se match hona chahiye
                local_dir=TARGET_DIR,
                repo_type="dataset"
            )
            print(f"[+] Extracting Text_Final_Model.zip into {text_model_dir}...")
            os.makedirs(text_model_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(text_model_dir)
            print("[+] Cleanly extracted Text_Final_Model.zip!")

            # --- Flatten nested folder if the zip wrapped its contents inside
            # a subfolder (e.g. final_model/config.json instead of
            # config.json directly). Without this, sentinel checks like
            # config.json existing at the top level will fail even though
            # the model files are present one level deeper. ---
            contents = os.listdir(text_model_dir)
            if len(contents) == 1 and os.path.isdir(os.path.join(text_model_dir, contents[0])):
                nested_dir = os.path.join(text_model_dir, contents[0])
                print(f"[+] Detected nested folder '{contents[0]}' — flattening...")
                for item in os.listdir(nested_dir):
                    shutil.move(os.path.join(nested_dir, item), text_model_dir)
                os.rmdir(nested_dir)
                print("[+] Flattened nested folder successfully!")

            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception as e:
            print(f"[ERROR] Error processing Text_Final_Model.zip: {e}")
            raise e

    # --- semantic + forensic: raw .pt files, no zip wrapping ---
    # NOTE: .pt files are already internally zip-formatted (PyTorch's own
    # serialization format). Wrapping them in an additional .zip and then
    # extracting with zipfile silently unpacks their internal contents
    # (data/0, data/1, ..., version, .data/serialization_id) instead of
    # producing a usable .pt file. So these are stored and pulled as raw
    # files on the Hub — no extraction step needed.
    for pt_filename in ["disaster_pro_model_BEST.pt", "forensic_v3_model_BEST.pt"]:
        target_path = os.path.join(TARGET_DIR, pt_filename)
        if os.path.exists(target_path):
            print(f"[OK] {pt_filename} already present — skipping download.")
            continue
        print(f"[+] Pulling {pt_filename} from Hub...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=pt_filename,
                local_dir=TARGET_DIR,
                repo_type="dataset"
            )
            print(f"[+] {pt_filename} downloaded to {downloaded_path}")
        except Exception as e:
            print(f"[ERROR] Error downloading {pt_filename}: {e}")
            raise e

    print("[DONE] All multimodal weights successfully loaded!")

if __name__ == "__main__":
    download_weights()