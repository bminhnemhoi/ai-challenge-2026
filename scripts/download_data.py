"""
Unified Master Data Downloader & Indexer for AI Challenge 2026 (AIC 2026).
Downloads all essential local metadata for Task 1 (KIS), Task 2 (VQA), and Task 3 (TRAKE).
"""

import os
import sys
import shutil
import urllib.request
import zipfile
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

METADATA_FILES = [
    ("clip-features-32-aic25-b1.zip", "https://aic-data.ledo.io.vn/clip-features-32-aic25-b1.zip", "clip-features-32"),
    ("map-keyframes-aic25-b1.zip", "https://aic-data.ledo.io.vn/map-keyframes-aic25-b1.zip", "map-keyframes"),
    ("objects-aic25-b1.zip", "https://aic-data.ledo.io.vn/objects-aic25-b1.zip", "objects"),
    ("media-info-aic25-b1.zip", "https://aic-data.ledo.io.vn/media-info-aic25-b1.zip", "media-info")
]

def download_and_extract_all():
    print("=== 🚀 AIC 2026 MASTER DATA DOWNLOADER (ALL 3 TASKS) ===", flush=True)
    
    for zip_name, url, target_folder in METADATA_FILES:
        target_path = os.path.join(DATA_DIR, target_folder)
        zip_local = os.path.join(DATA_DIR, zip_name)
        
        # Check if already present and extracted
        if os.path.exists(target_path) and len(os.listdir(target_path)) > 0:
            print(f"✅ [{target_folder}] Dữ liệu đã có sẵn ({len(os.listdir(target_path))} tệp). Bỏ qua tải lại!", flush=True)
            continue

        print(f"\n⚡ Đang tải {zip_name}...", flush=True)
        start_t = time.time()
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        with urllib.request.urlopen(req) as resp, open(zip_local, 'wb') as out:
            total_size = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            block_size = 1024 * 512 # 512KB
            
            while True:
                buffer = resp.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out.write(buffer)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb_dl = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    print(f"\r📦 Tiến độ {target_folder}: {mb_dl:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)", end="", flush=True)

        print(f"\n📦 Đang giải nén {zip_name} vào data/{target_folder}...", flush=True)
        with zipfile.ZipFile(zip_local, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
        if os.path.exists(zip_local):
            os.remove(zip_local)

        # Un-nest directory if extracted into subfolder (e.g. data/objects/objects/*.json)
        nested_dir = os.path.join(target_path, target_folder)
        if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
            for item in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, item), os.path.join(target_path, item))
            os.rmdir(nested_dir)

        print(f"✅ Hoàn thành {zip_name} trong {time.time() - start_t:.1f} giây!", flush=True)

    print("\n=== 🔨 TỰ ĐỘNG XÂY DỰNG BỘ CHỈ MỤC VECTOR SYSTEM (embeddings.npy & metadata.json) ===", flush=True)
    try:
        from src.core.btc_index_builder import build_official_btc_index
        build_official_btc_index()
    except Exception as e:
        print(f"Lỗi khởi tạo index: {e}")

    print("\n🎉 HOÀN THÀNH 100%! TẤT CẢ DỮ LIỆU LOCAL VÀ CHỈ MỤC DÙNG CHO CẢ 3 TASK ĐÃ SẴN SÀNG!")

if __name__ == "__main__":
    download_and_extract_all()
