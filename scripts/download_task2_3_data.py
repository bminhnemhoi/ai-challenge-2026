import os
import shutil
import urllib.request
import zipfile
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = [
    ("objects-aic25-b1.zip", "https://aic-data.ledo.io.vn/objects-aic25-b1.zip", "objects"),
    ("media-info-aic25-b1.zip", "https://aic-data.ledo.io.vn/media-info-aic25-b1.zip", "media-info")
]

def download_and_extract():
    print("=== 🚀 DOWNLOAD & EXTRACT METADATA FOR TASK 2 (VQA) & TASK 3 (TRAKE) ===", flush=True)
    
    for zip_name, url, target_folder in FILES:
        target_path = os.path.join(DATA_DIR, target_folder)
        zip_local = os.path.join(DATA_DIR, zip_name)
        
        if os.path.exists(target_path) and len(os.listdir(target_path)) > 0:
            print(f"✅ Thư mục '{target_folder}' đã tồn tại đầy đủ ({len(os.listdir(target_path))} tệp). Bỏ qua!", flush=True)
            continue

        print(f"\n⚡ Đang tải {zip_name} từ {url}...", flush=True)
        start_t = time.time()
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(zip_local, 'wb') as out:
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
                    print(f"\r📦 Tiến độ: {mb_dl:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)", end="", flush=True)

        print(f"\n📦 Đang giải nén {zip_name} vào {target_path}...", flush=True)
        with zipfile.ZipFile(zip_local, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
        if os.path.exists(zip_local):
            os.remove(zip_local)
            
        print(f"✅ Hoàn thành {zip_name} trong {time.time() - start_t:.1f} giây!", flush=True)

    print("\n🎉 TOÀN BỘ METADATA CẦN THIẾT CHO CẢ 3 DẠNG BÀI ĐÃ ĐƯỢC TẢI & GIẢI NÉN THÀNH CÔNG!")

if __name__ == "__main__":
    download_and_extract()
