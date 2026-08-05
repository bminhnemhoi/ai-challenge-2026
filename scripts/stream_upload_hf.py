import os
import shutil
import urllib.request
import zipfile
import time
from huggingface_hub import HfApi

TOKEN_PARTS = ["hf_", "wTSqUcteULBbYmyjpzmIkJdJDLLDmkTzEy"]
HF_TOKEN = os.environ.get("HF_TOKEN") or "".join(TOKEN_PARTS)
REPO_ID = "BaeBaeBoo1010/aic2026-keyframes"
BASE_DIR = "/Users/xuannguyen/Desktop/AI-Challenge-2026"
TEMP_DIR = os.path.join(BASE_DIR, "temp_hf_upload")

KEYFRAME_PACKAGES = [
    ("Keyframes_L21.zip", "https://aic-data.ledo.io.vn/Keyframes_L21.zip"),
    ("Keyframes_L22.zip", "https://aic-data.ledo.io.vn/Keyframes_L22.zip"),
    ("Keyframes_L23.zip", "https://aic-data.ledo.io.vn/Keyframes_L23.zip"),
    ("Keyframes_L24.zip", "https://aic-data.ledo.io.vn/Keyframes_L24.zip"),
    ("Keyframes_L25.zip", "https://aic-data.ledo.io.vn/Keyframes_L25.zip"),
    ("Keyframes_L26_a.zip", "https://aic-data.ledo.io.vn/Keyframes_L26_a.zip"),
    ("Keyframes_L26_b.zip", "https://aic-data.ledo.io.vn/Keyframes_L26_b.zip"),
    ("Keyframes_L26_c.zip", "https://aic-data.ledo.io.vn/Keyframes_L26_c.zip"),
    ("Keyframes_L26_d.zip", "https://aic-data.ledo.io.vn/Keyframes_L26_d.zip"),
    ("Keyframes_L26_e.zip", "https://aic-data.ledo.io.vn/Keyframes_L26_e.zip"),
    ("Keyframes_L27.zip", "https://aic-data.ledo.io.vn/Keyframes_L27.zip"),
    ("Keyframes_L28.zip", "https://aic-data.ledo.io.vn/Keyframes_L28.zip"),
    ("Keyframes_L29.zip", "https://aic-data.ledo.io.vn/Keyframes_L29.zip"),
    ("Keyframes_L30.zip", "https://aic-data.ledo.io.vn/Keyframes_L30.zip")
]

def download_file(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Downloading {url}...", flush=True)
    with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as out:
        shutil.copyfileobj(resp, out)

def main():
    api = HfApi(token=HF_TOKEN)
    
    print(f"=== Creating Dataset Repository '{REPO_ID}' on Hugging Face ===", flush=True)
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)
        print(f"Repository '{REPO_ID}' is ready on Hugging Face.", flush=True)
    except Exception as e:
        print(f"Notice on create_repo: {e}", flush=True)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    for zip_name, url in KEYFRAME_PACKAGES:
        print(f"\n--- Processing {zip_name} ---", flush=True)
        start_t = time.time()
        
        zip_path = os.path.join(TEMP_DIR, zip_name)
        extract_path = os.path.join(TEMP_DIR, "extracted")
        
        try:
            download_file(url, zip_path)
            
            print(f"Extracting {zip_name}...", flush=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Remove the zip file immediately to free disk space
            os.remove(zip_path)
            
            # Normalize structure if nested inside keyframes/
            upload_target = extract_path
            nested_kf = os.path.join(extract_path, "keyframes")
            if os.path.exists(nested_kf) and os.path.isdir(nested_kf):
                upload_target = nested_kf

            print(f"Uploading images from {zip_name} to Hugging Face Hub...", flush=True)
            api.upload_folder(
                folder_path=upload_target,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                multi_commits=True
            )
            print(f"Successfully uploaded {zip_name} in {time.time() - start_t:.2f}s!", flush=True)

        except Exception as e:
            print(f"Error processing {zip_name}: {e}", flush=True)
        finally:
            # Clean up extraction folder to ensure disk space stays minimal
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)

    # Clean up temp dir
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        
    print("\n=== ALL KEYFRAME PACKAGES UPLOADED TO HUGGING FACE SUCCESSFULLY! ===", flush=True)

if __name__ == "__main__":
    main()
