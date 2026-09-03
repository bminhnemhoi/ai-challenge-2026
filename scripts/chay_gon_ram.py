"""Chạy một script đo TRAKE với dấu chân RAM nhỏ hơn — kết quả Y HỆT.

Máy chỉ còn ~3GB trống khi lane pe-core đang encode (giữ 3GB). Checkpoint
SigLIP-2 SO400M là 4,5GB fp32; đường nạp mặc định của transformers cần đỉnh
~2× cỡ model nên tiến trình chết segfault khi nạp trọng số.

Wrapper này KHÔNG đổi con số nào của phép đo:

  * `low_cpu_mem_usage=True` — chỉ đổi CÁCH nạp (tensor-by-tensor từ mmap),
    trọng số fp32 ra giống hệt từng bit;
  * xoá tháp thị giác sau khi nạp — mọi phép đo TRAKE chỉ dùng
    `get_text_features`, không đụng tháp thị giác; xoá nó trả lại ~1,7GB.

    python -u scripts/chay_gon_ram.py scripts/do_soft_order_test.py --gt ...
"""

from __future__ import annotations

import gc
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.kis_engine import KISEngine  # noqa: E402


def _load_model_gon(self) -> None:
    import torch
    from transformers import AutoModel, AutoProcessor

    self._torch = torch
    dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
    self.device = dev
    self.processor = AutoProcessor.from_pretrained(self.model_name)
    self.model = AutoModel.from_pretrained(
        self.model_name, dtype=torch.float32, low_cpu_mem_usage=True
    ).to(dev).eval()
    # cac phep do TRAKE chi encode VAN BAN — thap thi giac chi ton RAM
    if hasattr(self.model, "vision_model"):
        del self.model.vision_model
        gc.collect()
    print("[chay_gon_ram] da nap model kieu tiet kiem, bo thap thi giac",
          flush=True)


KISEngine._load_model = _load_model_gon

if len(sys.argv) < 2:
    raise SystemExit("dung: python scripts/chay_gon_ram.py <script> [args...]")

target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
