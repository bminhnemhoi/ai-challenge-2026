"""Quét chuỗi bí mật — kể cả khi nó bị cắt nhỏ rồi nối lại.

Bộ chốt chặn đầu tiên của CI chỉ tìm `AIza[0-9A-Za-z_-]{30,}` bằng `git grep`. Nó
bỏ lọt một token HuggingFace nằm ngay trong repo suốt nhiều tuần, vì token ấy được
viết thế này:

    TOKEN_PARTS = ["hf_", "wTSqUcte...DmkTzEy"]
    HF_TOKEN = os.environ.get("HF_TOKEN") or "".join(TOKEN_PARTS)

Tiền tố `hf_` và phần thân nằm ở hai chuỗi khác nhau, nên mọi biểu thức chính quy
tìm `hf_<40 ký tự>` đều trượt. Cách chữa không phải thêm biểu thức dài hơn mà là
**bỏ dấu nháy và dấu nối đi trước khi so khớp**: sau khi xoá `", "` thì hai mảnh
dính lại thành đúng token và mẫu khớp ngay.

Quét hai vòng:

  1. **Đã dán lại** — bỏ mọi dấu nháy, dấu phẩy, dấu cộng, ngoặc vuông và khoảng
     trắng, rồi tìm các tiền tố nhà cung cấp đã biết. Bắt được cả token viết liền
     lẫn token bị cắt nhỏ.
  2. **Nghi ngờ theo ngữ cảnh** — một chuỗi dài, ngẫu nhiên, nằm cạnh một cái tên
     có chữ TOKEN/KEY/SECRET/PASSWORD. Bắt được token của nhà cung cấp mà vòng 1
     chưa biết mặt.

    python scripts/quet_bi_mat.py            # cây làm việc
    python scripts/quet_bi_mat.py --lich-su  # thêm toàn bộ lịch sử git

Trả mã thoát 1 nếu tìm thấy — CI dựa vào đó.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

#: tiền tố của các nhà cung cấp mà dự án này có đụng tới, cộng vài cái phổ biến
TIEN_TO = {
    "Google API": r"AIza[0-9A-Za-z_-]{30,}",
    "HuggingFace": r"hf_[A-Za-z0-9]{30,}",
    "GitHub cũ": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "GitHub mới": r"github_pat_[A-Za-z0-9_]{50,}",
    "OpenAI": r"sk-[A-Za-z0-9_-]{30,}",
    "Slack": r"xox[baprs]-[A-Za-z0-9-]{20,}",
    "AWS": r"AKIA[0-9A-Z]{16}",
}

#: bỏ hết những gì có thể chen vào GIỮA hai mảnh của một token bị cắt
NHIEU = re.compile(r"""['"`,\+\[\]\(\)\s\\]+""")

DA_CHET: set[str] = set()

TEN_NHAY_CAM = re.compile(r"(token|key|secret|password|passwd|credential|api[_-]?key)", re.I)
CHUOI_DAI = re.compile(r"""['"]([A-Za-z0-9_\-]{28,})['"]""")

BO_QUA_DUONG_DAN = re.compile(
    r"(^|/)(\.git/|data/|node_modules/|__pycache__/|\.venv/|dist/)"
    r"|\.(npy|npz|zip|jpg|jpeg|png|pdf|bin|safetensors|lock)$"
)

#: những chuỗi dài vô hại hay gặp, đừng báo động giả
CHO_PHEP = re.compile(
    r"^(sha256|abcdef|0123456789|BaeBaeBoo|huggingface|https?)"
    r"|^[0-9a-f]{40}$"          # sha1 của git
    r"|^[A-Za-z0-9_]*(test|example|placeholder|your|xxx|dummy|sample)",
    re.I,
)


DANH_SACH_DA_CHET = ROOT / ".secrets-revoked.txt"


def da_thu_hoi() -> set[str]:
    """Vân tay của những khoá đã bị thu hồi, nên không cần báo động nữa.

    Lịch sử git không xoá được nếu không viết lại lịch sử — mà việc đó buộc cả đội
    clone lại. Với một khoá ĐÃ THU HỒI thì nó chỉ còn là rác vô hại, và để CI đỏ mãi
    vì nó thì chẳng bao lâu sẽ không ai nhìn CI nữa. Ghi vân tay băm chứ không ghi
    khoá, để chính danh sách này không thành chỗ rò rỉ.
    """
    if not DANH_SACH_DA_CHET.is_file():
        return set()
    return {
        d.split("#", 1)[0].strip()
        for d in DANH_SACH_DA_CHET.read_text(encoding="utf-8").splitlines()
        if d.split("#", 1)[0].strip()
    }


def van_tay(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def do_hon_loan(s: str) -> float:
    """Entropy Shannon trên từng ký tự. Token thật thường > 3,5; chữ tiếng Anh ~2,5."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def quet_van_ban(ten: str, noi_dung: str) -> list[str]:
    ket: list[str] = []

    # vòng 1 — dán lại rồi mới so khớp
    dan_lai = NHIEU.sub("", noi_dung)
    for nha, mau in TIEN_TO.items():
        for m in re.finditer(mau, dan_lai):
            lo = m.group(0)
            if van_tay(lo) in DA_CHET:
                continue
            ket.append(f"{ten}: [{nha}] {lo[:12]}… ({len(lo)} ký tự)"
                       + ("  — BỊ CẮT NHỎ rồi nối lại, grep thường không thấy"
                          if not re.search(mau, noi_dung) else "")
                       + f"\n      vân tay: {van_tay(lo)}")

    # vòng 2 — chuỗi dài hỗn loạn nằm cạnh một cái tên nhạy cảm
    for dong_so, dong in enumerate(noi_dung.splitlines(), 1):
        if not TEN_NHAY_CAM.search(dong):
            continue
        for m in CHUOI_DAI.finditer(dong):
            gia_tri = m.group(1)
            if (CHO_PHEP.search(gia_tri) or do_hon_loan(gia_tri) < 3.5
                    or van_tay(gia_tri) in DA_CHET):
                continue
            ket.append(f"{ten}:{dong_so}: [nghi ngờ] chuỗi {len(gia_tri)} ký tự, "
                       f"entropy {do_hon_loan(gia_tri):.1f}, cạnh một tên có 'token/key/secret'"
                       f"\n      vân tay: {van_tay(gia_tri)}")
    return ket


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lich-su", action="store_true", help="quét cả lịch sử git, không chỉ cây hiện tại")
    args = ap.parse_args()

    global DA_CHET
    DA_CHET = da_thu_hoi()
    if DA_CHET:
        print(f"bỏ qua {len(DA_CHET)} khoá đã thu hồi (xem .secrets-revoked.txt)")

    loi: list[str] = []

    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           cwd=ROOT).stdout.splitlines()
    n = 0
    for rel in files:
        if BO_QUA_DUONG_DAN.search(rel):
            continue
        p = ROOT / rel
        try:
            noi_dung = p.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        n += 1
        loi += quet_van_ban(rel, noi_dung)
    print(f"đã quét {n} file trong cây làm việc")

    if args.lich_su:
        diff = subprocess.run(["git", "log", "--all", "-p", "--no-color"],
                              capture_output=True, text=True, cwd=ROOT,
                              errors="ignore").stdout
        them = "\n".join(l[1:] for l in diff.splitlines() if l.startswith("+"))
        print(f"đã quét {len(them.splitlines())} dòng được thêm trong toàn bộ lịch sử")
        loi += quet_van_ban("<lịch sử git>", them)

    if not loi:
        print("Sạch: không thấy chuỗi bí mật nào.")
        return 0

    print(f"\nTÌM THẤY {len(loi)} chỗ nghi vấn:")
    for x in dict.fromkeys(loi):
        print("  " + x)
    print("\nViệc đầu tiên luôn là THU HỒI/ĐỔI khoá. Xoá khỏi mã là việc thứ hai —"
          "\nlịch sử git vẫn đọc được, nên xoá thôi không cắt được thiệt hại.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
