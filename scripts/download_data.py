"""
Unified Master Data Downloader & Indexer for AI Challenge 2026 (AIC 2026).
Downloads all essential local metadata and SigLIP 2 / CLIP vector indices for Task 1 (KIS), Task 2 (VQA), and Task 3 (TRAKE).
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

HF_CDN = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"

# Direct essential files (Hugging Face CDN)
HF_DIRECT_FILES = [
    ("metadata.json", f"{HF_CDN}/metadata.json"),
    ("blank_frame_indices.json", f"{HF_CDN}/blank_frame_indices.json"),
    ("embeddings_siglip2_384.npy", f"{HF_CDN}/embeddings_siglip2_384.npy"),
]


# Zipped packages from BTC / HF
ZIP_PACKAGES = [
    ("media-info.zip", f"{HF_CDN}/media-info.zip", "media-info"),
    ("map-keyframes-aic25-b1.zip", "https://aic-data.ledo.io.vn/map-keyframes-aic25-b1.zip", "map-keyframes"),
    ("objects-aic25-b1.zip", "https://aic-data.ledo.io.vn/objects-aic25-b1.zip", "objects"),
]

def download_file_with_progress(url, dest_path, desc):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AI-Challenge-2026)"})
    with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as out:
        total_size = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 1024 # 1MB
        
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
                print(f"\r📦 Tiến độ {desc}: {mb_dl:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)", end="", flush=True)
    print("", flush=True)

def download_atomically(url, dest_path, desc):
    """Download to a .part file and rename only on success.

    Writing straight to the final name means a Ctrl-C or a dropped connection
    leaves a truncated file that the "already present" check below happily
    accepts, because it only tests size > 0.  The next run skips it, and
    kis_engine's np.load(..., mmap_mode="r") then fails on Windows with
    "Not enough memory resources are available" — which sends the operator
    hunting for a RAM problem that does not exist.
    """
    part = dest_path + ".part"
    try:
        download_file_with_progress(url, part, desc)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.replace(part, dest_path)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise


def download_and_extract_all():
    print("=== 🚀 AIC 2026 MASTER DATA DOWNLOADER (ALL 3 TASKS) ===", flush=True)
    failures = []

    # 1. Download direct Hugging Face index files
    print("\n--- ⚡ 1. TẢI BỘ CHỈ MỤC SIGLIP 2 & METADATA (TASK 1 KIS) ---", flush=True)
    for fname, url in HF_DIRECT_FILES:
        dest_p = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest_p) and os.path.getsize(dest_p) > 0:
            sz_mb = os.path.getsize(dest_p) / (1024 * 1024)
            print(f"✅ [{fname}] Đã có sẵn trên máy ({sz_mb:.1f} MB). Bỏ qua tải lại!", flush=True)
            continue

        print(f"⚡ Đang tải {fname} từ Hugging Face CDN...", flush=True)
        start_t = time.time()
        try:
            download_atomically(url, dest_p, fname)
            print(f"✅ Hoàn thành tải {fname} trong {time.time() - start_t:.1f}s!", flush=True)
        except Exception as e:
            print(f"❌ Lỗi tải {fname}: {e}", flush=True)
            failures.append(fname)

    # 2. Download zipped packages
    print("\n--- 📦 2. TẢI DỮ LIỆU BỔ TRỢ (MEDIA INFO, OBJECTS, KEYFRAME MAPS) ---", flush=True)
    for zip_name, url, target_folder in ZIP_PACKAGES:
        target_path = os.path.join(DATA_DIR, target_folder)
        zip_local = os.path.join(DATA_DIR, zip_name)
        
        # Check if already present and extracted
        if os.path.exists(target_path) and len(os.listdir(target_path)) > 0:
            print(f"✅ [{target_folder}] Dữ liệu đã có sẵn ({len(os.listdir(target_path))} tệp). Bỏ qua tải lại!", flush=True)
            continue

        print(f"\n⚡ Đang tải {zip_name}...", flush=True)
        start_t = time.time()
        try:
            download_atomically(url, zip_local, target_folder)
            print(f"📦 Đang giải nén {zip_name} vào data/{target_folder}...", flush=True)
            with zipfile.ZipFile(zip_local, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)

            # The archive is KEPT. build_review_page.py reads object detections
            # straight out of objects-aic25-b1.zip, so deleting it here made the
            # review page lose every object caption on a fresh clone — with no
            # error, just silently emptier thumbnails.

            # Un-nest directory if extracted into subfolder (e.g. data/objects/objects/*.json)
            nested_dir = os.path.join(target_path, target_folder)
            if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
                for item in os.listdir(nested_dir):
                    shutil.move(os.path.join(nested_dir, item), os.path.join(target_path, item))
                os.rmdir(nested_dir)

            print(f"✅ Hoàn thành {zip_name} trong {time.time() - start_t:.1f} giây!", flush=True)
        except Exception as e:
            print(f"❌ Lỗi xử lý {zip_name}: {e}", flush=True)
            failures.append(zip_name)

    # Claiming success unconditionally is worse than useless the day before a
    # contest: a teammate whose downloads all failed reads "HOÀN THÀNH 100%",
    # closes the terminal, and finds out during the round.
    if failures:
        print(f"\n❌ CHƯA XONG — {len(failures)} mục tải lỗi: {', '.join(failures)}")
        print("   Chạy lại lệnh này; các file đã tải xong sẽ được bỏ qua.")
        return 1
    print("\n🎉 HOÀN THÀNH 100%! TẤT CẢ DỮ LIỆU LOCAL VÀ CHỈ MỤC DÙNG CHO CẢ 3 TASK ĐÃ SẴN SÀNG!")
    return 0

if __name__ == "__main__":
    raise SystemExit(download_and_extract_all())

