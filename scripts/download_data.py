import os
import sys
import zipfile
import urllib.request
import time

URLS = {
    "clip_features": "https://aic-data.ledo.io.vn/clip-features-32-aic25-b1.zip",
    "map_keyframes": "https://aic-data.ledo.io.vn/map-keyframes-aic25-b1.zip",
    "media_info": "https://aic-data.ledo.io.vn/media-info-aic25-b1.zip",
    "objects": "https://aic-data.ledo.io.vn/objects-aic25-b1.zip",
    "keyframes_l21": "https://aic-data.ledo.io.vn/Keyframes_L21.zip"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
DATA_DIR = os.path.join(BASE_DIR, "data")
KEYFRAMES_DIR = os.path.join(BASE_DIR, "keyframes")

def download_file(url: str, dest_path: str):
    print(f"\nDownloading: {url}", flush=True)
    print(f"Destination: {dest_path}", flush=True)
    start_time = time.time()

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        block_size = 1024 * 64

        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)

            percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
            sys.stdout.write(f"\r  Progress: {percent}% [{mb_downloaded:.1f}/{mb_total:.1f} MB]")
            sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\nDownloaded successfully in {elapsed:.2f}s.", flush=True)

def extract_zip(zip_path: str, extract_to: str):
    print(f"Extracting {os.path.basename(zip_path)} -> {extract_to}...", flush=True)
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted successfully. Removing zip file to save space...", flush=True)
    os.remove(zip_path)

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(KEYFRAMES_DIR, exist_ok=True)

    print("=== AIC 2026 Core Dataset Downloader & Extractor ===", flush=True)

    for key, url in URLS.items():
        filename = os.path.basename(url)
        dest_zip = os.path.join(DOWNLOAD_DIR, filename)

        if key in ["clip_features", "map_keyframes"]:
            target_folder = DATA_DIR
        elif key == "media_info":
            target_folder = os.path.join(DATA_DIR, "media_info")
        elif key == "objects":
            target_folder = os.path.join(DATA_DIR, "objects")
        elif key.startswith("keyframes"):
            target_folder = KEYFRAMES_DIR
        else:
            target_folder = DATA_DIR

        try:
            download_file(url, dest_zip)
            extract_zip(dest_zip, target_folder)
        except Exception as e:
            print(f"\nError processing {filename}: {e}", flush=True)

    print("\nAll essential dataset files downloaded and extracted successfully!", flush=True)

if __name__ == "__main__":
    main()
