"""Thu hoạch ground truth từ ĐỀ THẬT (vòng luyện tập + sơ tuyển 1 + sơ tuyển 2).

Vì sao cần: ngưỡng 2σ trên 30 câu TEST hiện tại đòi chênh rất lớn mới kết luận
được, nên phần lớn cải tiến ăn 1–3 câu không bao giờ qua nổi cổng. Bộ đo lớn hơn
hạ ngưỡng. Nguyên liệu duy nhất chắc chắn khớp phân bố ra đề của BTC là 79 câu
đề thật đã chạy qua tay người trong ba vòng.

Vì sao KHÔNG được chép thẳng các file picks vào ground truth:

* bài nộp vòng luyện tập được **8,6/24**, vòng sơ tuyển 2 được **10,0/30** —
  nghĩa là xét theo tổng thể, đa số pick trong các file đó SAI. Chép cả vào bộ
  đo là bơm nhiễu nhãn 50–60%, tệ hơn nhiều so với không có bộ đo;
* nhiều pick là "engine xếp hạng 1, VLM gật đầu". Chấm đường truy xuất trên
  chính thứ đường truy xuất đẻ ra là tự chấm bài mình — nền bị thổi lên, mọi
  cải tiến bị nén vào trần;
* một số câu bị đổi qua đổi lại giữa các lượt (p2-7, p2-8, p2-18, p2-24, p2-27,
  p2-2). Ít nhất một trong hai lựa chọn phải sai.

Nên script này làm ba việc, theo đúng thứ tự đó:

1. **Đọc máy** mọi file picks (round_p1, round1, round2), dựng bảng
   {(vòng, mã câu) -> các lượt chọn}, tự phát hiện mâu thuẫn và câu trùng đề.
2. **Đối chiếu tự động** với ba nguồn có sẵn trong repo:
   `data/metadata.json` (khung hình có tồn tại không, keyframe gần nhất),
   `data/media-info-aic25-b1.zip` (tiêu đề 873/873 video — kênh chốt video
   mạnh nhất và rẻ nhất), `data/captions/` (lời thoại **217/873** video —
   dùng để kiểm cụm từ hiếm có thật xuất hiện đúng video đó không).
3. **Xếp độ tin bằng LUẬT** (hàm ``xep_do_tin``), dựa trên bảng nhãn tay
   ``NHAN`` — trong đó trường ``kiem_lai_lane`` ghi kết quả MỞ THẬT khung hình
   1280px của lane này (xem docs/HARNESS_DE_THAT.md, mục "Đã mở những khung nào").

Chạy:

    python scripts/thu_hoach_de_that.py                 # sinh 2 file json + báo cáo
    python scripts/thu_hoach_de_that.py --in-bang       # thêm bảng từng mục
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import (  # noqa: E402
    detect_task,
    read_en_override,
    read_query_text,
    split_events,
    split_qa,
)

CDN = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"

#: Thư mục đề + các file picks của từng vòng, theo thứ tự thời gian trong vòng
#: (lượt sau đè lượt trước — đó là quyết định cuối cùng của người soát).
VONG = {
    "round_p1": {
        "nhan": "luyện tập (24 câu, điểm công bố 8,6/24)",
        "queries": "round_p1/queries",
        "picks": ["round_p1/picks_verified.txt"],
    },
    "round1": {
        "nhan": "sơ tuyển 1 (25 câu, không có điểm từng câu)",
        "queries": "round1/queries",
        "picks": ["round1/picks_verified.txt"],
    },
    "round2": {
        "nhan": "sơ tuyển 2 (30 câu, điểm công bố 10,0/30)",
        "queries": "round2/queries",
        "picks": [
            "round2/picks_luot1.txt",
            "round2/picks_luot2.txt",
            "round2/picks_luot3.txt",
            "round2/picks_final.txt",
            "round2/picks_final2.txt",
            "round2/picks_fix.txt",
            "round2/picks_p22.txt",
            "round2/picks_kis_final.txt",
            "round2/picks_p226.txt",
        ],
    },
}

# ---------------------------------------------------------------------------
# Bảng nhãn tay — phần DUY NHẤT của file này là phán đoán của người, không phải
# suy ra được từ dữ liệu. Mỗi trường có nghĩa hẹp, cố tình:
#
#   kenh_video  kênh đã chốt VIDEO, phải ĐỘC LẬP với SigLIP thì mới tính:
#               "tieu_de"        tiêu đề/mô tả video của BTC (đối chiếu được bằng máy)
#               "loi_thoai"      cụm từ hiếm trong transcript (đối chiếu được nếu có caption)
#               "chu_tren_hinh"  chữ đọc được ngay trên khung hình
#               "loai_tru"       loại trừ thủ công trên cả một nhóm video
#               "engine"         engine xếp hạng 1 / VLM gật đầu — KHÔNG độc lập
#   kenh_khung  kênh đã chốt KHUNG HÌNH:
#               "chu_tren_hinh"  chữ/nội dung chỉ có ở đúng khung đó
#               "moc_loi_thoai"  suy từ mốc thời gian lời thoại
#               "vlm_cham"       VLM chấm điểm các keyframe
#               "engine"         thang frame của engine
#               "khong"          pick không ghi frame nào
#   kiem_lai_lane  lane harness có MỞ khung hình 1280px ra xem không, và thấy gì:
#               "khop" | "khong_khop" | "chua_kiem"
#
# Không có mục nào ở đây được phép "đoán đáp án rồi ghi như thật": mọi đáp án
# đều là chuỗi người soát đã ghi trong file picks, và trường do_tin_dap_an nói
# rõ nó đã được đối chiếu lại hay chưa.
# ---------------------------------------------------------------------------

NHAN = {
    # ---------------------------------------------------------- vòng luyện tập
    ("round_p1", "query-p1-15-qa"): dict(
        kenh_video="chu_tren_hinh", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        dap_an_kiem="khop",
        bang_chung="Băng rôn trong khung: 'FANA CÙNG EM ĐẾN TRƯỜNG / Xã Giang Ly, "
                   "huyện Khánh Vĩnh, tỉnh Khánh Hoà 04.03.2024'. Mô tả video của BTC "
                   "nêu đúng CLB FANA.",
    ),
    ("round_p1", "query-p1-19-qa"): dict(
        kenh_video="loi_thoai", kenh_khung="chu_tren_hinh", kiem_lai_lane="khong_khop",
        dap_an_kiem="chua_kiem",
        bang_chung="Tiêu đề video = tập quay ở Rạch Giá (Kiên Giang) nên VIDEO hợp lý.",
        ghi_chu="LANE MỞ KHUNG: khung này là nội thất đình, KHÔNG có tấm bia ghi tên "
                "anh hùng và KHÔNG có hai câu thơ như bằng chứng đã ghi. Bằng chứng "
                "'VLM đọc được ... tin cậy 95%' không đứng vững ở khung này. Video "
                "có thể vẫn đúng; frame thì phải tìm lại.",
    ),
    ("round_p1", "query-p1-22-qa"): dict(
        kenh_video="tieu_de", kenh_khung="chu_tren_hinh", kiem_lai_lane="khong_khop",
        dap_an_kiem="chua_kiem",
        bang_chung="Tiêu đề video: 'Lớp học 0 đồng cho người yêu bếp' — khớp 'phụ nữ "
                   "dạy nấu ăn cho những người khác'.",
        ghi_chu="LANE MỞ KHUNG: khung này là cảnh bóp bột vào khuôn giấy làm bánh ngọt, "
                "KHÔNG có tờ công thức nào. Bằng chứng 'đọc tờ công thức, tin cậy 100%' "
                "không đứng vững ở khung này.",
    ),
    ("round_p1", "query-p1-4-trake"): dict(
        kenh_video="tieu_de", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Tiêu đề: 'MĂNG TÂY CHIÊN BIA XỐT CÁ NGỪ' — đề tả tẩm bột rồi CHIÊN "
                   "ngập dầu; bài đang nộp lúc đó là món XÀO.",
        ghi_chu="Bốn mốc TRAKE lấy từ mẫu 16 khung nửa sau video, chưa ai xem lại.",
    ),
    ("round_p1", "query-p1-17-kis"): dict(
        kenh_video="tieu_de", kenh_khung="khong", kiem_lai_lane="chua_kiem",
        bang_chung="Tiêu đề: 'Ấm áp phiên chợ 0 đồng ở bệnh viện' — khớp bối cảnh bệnh viện.",
    ),
    ("round_p1", "query-p1-24-kis"): dict(
        kenh_video="tieu_de", kenh_khung="khong", kiem_lai_lane="chua_kiem",
        bang_chung="Tiêu đề: '... CHIẾN THẮNG ĐỒNG ĐỘI TÍNH GIỜ' — đúng nội dung 3 tay "
                   "đua cùng đội đạp thành một hàng.",
    ),
    ("round_p1", "query-p1-21-kis"): dict(
        kenh_video="engine", kenh_khung="khong", kiem_lai_lane="chua_kiem",
        bang_chung="VLM chọn, người soát đồng ý. Không có kênh nào độc lập với engine.",
    ),
    ("round_p1", "query-p1-23-kis"): dict(
        kenh_video="engine", kenh_khung="khong", kiem_lai_lane="chua_kiem",
        bang_chung="Hai model cùng chọn. Không có kênh độc lập.",
    ),
    ("round_p1", "query-p1-18-trake"): dict(
        kenh_video="engine", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="VLM 0,80 — cao nhất trong 6 chuỗi.",
        ghi_chu="docs/SUA_VONG_1.md nêu một ứng viên KHÁC mạnh hơn (video có 'củ năng' "
                "trong lời thoại). Mâu thuẫn chưa đóng.",
    ),

    # ------------------------------------------------------------ sơ tuyển 1
    ("round1", "query-p1-9-qa"): dict(
        kenh_video="loi_thoai", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        dap_an_kiem="khop",
        loi_thoai_can=r"amsterdam",
        bang_chung="LANE MỞ KHUNG: kênh đào Amsterdam, đoàn xe lội nước (vàng/đỏ/đen) "
                   "nối đuôi chui qua cầu; vòm TRÁI có biển ghi '2,15', vòm phải ghi '6'. "
                   "Transcript nhắc 'kênh đào Amsterdam' quanh mốc này.",
        ghi_chu="Đáp án ghi '2.15' (dấu chấm) vì sanitise_field đổi dấu phẩy thành "
                "khoảng trắng, sẽ thành '2 15'.",
    ),
    ("round1", "query-p1-12-kis"): dict(
        kenh_video="chu_tren_hinh", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        bang_chung="LANE MỞ KHUNG: trạm xăng, nhiều tài xế xe ôm công nghệ áo xanh, "
                   "một người chạy xe từ trái sang phải; chữ chạy dưới màn hình đọc rõ "
                   "'dầu mazut còn 15.562 đồng/kg'. Đề đòi CẢ HAI thứ trong một khung.",
    ),
    ("round1", "query-p1-15-qa"): dict(
        kenh_video="chu_tren_hinh", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        dap_an_kiem="chua_kiem",
        bang_chung="LANE MỞ KHUNG: đúng bản đồ phân bố động đất, bảng chú giải nằm bên "
                   "TRÁI với các khoảng độ lớn (<4.0, 4.0-4.5, ...). Bản đồ chỉ xuất hiện "
                   "ở khung này.",
        ghi_chu="ĐÁP ÁN KHÔNG ĐÁNG TIN: chính file picks ghi 'độ tin thấp' — 'cấp độ 4' "
                "không khớp một lớp chú giải nào, hai model đếm ra 12 và 24 trên cùng ảnh. "
                "Nhãn TRUY XUẤT dùng được; nhãn ĐÁP ÁN thì không.",
    ),
    ("round1", "query-p1-17-qa"): dict(
        kenh_video="loi_thoai", kenh_khung="moc_loi_thoai", kiem_lai_lane="khop",
        dap_an_kiem="khop",
        loi_thoai_can=r"tằng quái|tăng quái",
        bang_chung="Cụm 'đèo Tằng quái' xuất hiện ĐÚNG MỘT LẦN trong 217 transcript có "
                   "trong repo, @381,8 s của chính video này; keyframe được chốt ở "
                   "pts 382,8 s — lệch 1 giây. LANE MỞ KHUNG: đúng cảnh đất đá/cây đổ "
                   "tràn kín mặt đường đèo.",
    ),
    ("round1", "query-p1-19-kis"): dict(
        kenh_video="tieu_de", kenh_khung="vlm_cham", kiem_lai_lane="khop",
        bang_chung="Tiêu đề: 'Nam Sư Du Hí Ăn Bông Bí'. LANE MỞ KHUNG: lân VÀNG trên trụ, "
                   "quả bí đỏ kèm bông hoa vàng ngay trước miệng — đúng khoảnh khắc đề tả.",
    ),
    ("round1", "query-p1-22-kis"): dict(
        kenh_video="loi_thoai", kenh_khung="moc_loi_thoai", kiem_lai_lane="khop",
        loi_thoai_can=r"remember",
        bang_chung="'remember' chỉ xuất hiện ở video này trong 217 transcript; tiêu đề "
                   "'Chuyên đề 6 — Động từ thêm ing và động từ có to'. LANE MỞ KHUNG: "
                   "bảng ghi 'Verbs followed by both gerund and to-infinitive with "
                   "different meanings', nhánh 'remember → V-ing', cô giáo áo dài HỒNG "
                   "đeo kính — khớp từng chi tiết của đề.",
    ),
    ("round1", "query-p1-4-kis"): dict(
        kenh_video="loi_thoai", kenh_khung="vlm_cham", kiem_lai_lane="khong_khop",
        loi_thoai_can=r"cân đo",
        bang_chung="'cân đo sức khoẻ' (vườn thú London) chỉ có ở video này trong 217 "
                   "transcript, @364 s và @645 s. VIDEO chắc.",
        ghi_chu="LANE MỞ KHUNG: khung được chốt (pts 667 s) là cảnh PHỎNG VẤN nhân viên "
                "áo xanh bên hồ, KHÔNG phải đàn sư tử trên bục gỗ cũng không phải cảnh "
                "hai nhân viên đang cân. Frame gần như chắc chắn sai cửa sổ.",
    ),
    ("round1", "query-p1-5-kis"): dict(
        kenh_video="tieu_de", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Tiêu đề 'MỰC XÀO ĐẬU HÀ LAN' so với đối thủ 'MỰC XÀO XỐT TIÊU XANH' — "
                   "VLM chấm hoà 1.00 cả hai, tiêu đề mới tách được.",
    ),
    ("round1", "query-p1-6-kis"): dict(
        kenh_video="loi_thoai", kenh_khung="moc_loi_thoai", kiem_lai_lane="chua_kiem",
        loi_thoai_can=r"kim cương thô",
        bang_chung="'kim cương thô' chỉ có ở video này trong 217 transcript (5 lần, "
                   "702–760 s).",
        ghi_chu="Khung chốt ở 707 s, còn câu 'đã đến xem khối kim cương thô này' ở 725 s — "
                "lệch ~18 s, rộng hơn nhiều so với cửa sổ chấm. Frame đáng ngờ.",
    ),
    ("round1", "query-p1-8-kis"): dict(
        kenh_video="tieu_de", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Tiêu đề 'TRỨNG HẤP HÀU RONG BIỂN' — đề tả đĩa đang được HẤP trong nồi.",
        ghi_chu="Đề p1-8 và p1-14 GIỐNG HỆT NHAU từng byte (md5 trùng) — script gộp "
                "thành một mục để không đếm hai lần.",
    ),
    ("round1", "query-p1-14-kis"): dict(
        kenh_video="tieu_de", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Đề GIỐNG HỆT p1-8-kis từng byte nên bắt buộc cùng đáp án.",
        ghi_chu="Bản sao — script tự hạ xuống suy_ra để không đếm hai lần một câu.",
    ),
    ("round1", "query-p1-18-kis"): dict(
        kenh_video="loi_thoai", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Tiêu đề 'BÚN GÀ XÁO SẢ'; lời thoại 4:24 đủ 4 nguyên liệu của đề. "
                   "Engine KHÔNG hề xếp video này vào 100 dòng — nên nhãn này độc lập "
                   "hoàn toàn với đường truy xuất, rất đáng soát tay.",
    ),
    ("round1", "query-p1-24-kis"): dict(
        kenh_video="tieu_de", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Tiêu đề 'Đôi Mắt MeKong — Tập 3: Nghề đan đát' khớp phóng sự đan lát "
                   "lục bình.",
    ),
    ("round1", "query-p1-16-trake"): dict(
        kenh_video="loai_tru", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Đối chiếu TỪNG SỰ KIỆN trên cả 43 video nhóm L24: chỉ video này có "
                   "đủ hai rồng vàng + lân xoay trên trụ + dùi chạm kẻng đồng. Tiêu đề "
                   "'Con Rồng Cháu Tiên' củng cố.",
        ghi_chu="Ba mốc chưa ai xem lại. Đây là ứng viên TRAKE tốt nhất đang có.",
    ),
    ("round1", "query-p1-10-kis"): dict(kenh_video="engine", kenh_khung="vlm_cham",
                                        kiem_lai_lane="chua_kiem",
                                        bang_chung="Engine + lời thoại + VLM cùng dẫn đầu."),
    ("round1", "query-p1-1-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                       kiem_lai_lane="chua_kiem", bang_chung="Engine dẫn đầu, VLM xác nhận."),
    ("round1", "query-p1-2-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                       kiem_lai_lane="chua_kiem", bang_chung="Engine dẫn đầu, VLM xác nhận."),
    ("round1", "query-p1-7-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                       kiem_lai_lane="chua_kiem", bang_chung="Engine dẫn đầu, VLM xác nhận."),
    ("round1", "query-p1-11-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                        kiem_lai_lane="chua_kiem", bang_chung="Engine dẫn đầu, VLM xác nhận."),
    ("round1", "query-p1-13-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                        kiem_lai_lane="chua_kiem", bang_chung="Engine dẫn đầu, VLM xác nhận."),
    ("round1", "query-p1-21-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                        kiem_lai_lane="chua_kiem", bang_chung="Engine dẫn đầu, VLM xác nhận."),
    ("round1", "query-p1-23-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                        kiem_lai_lane="chua_kiem", bang_chung="Engine dẫn đầu, VLM xác nhận."),
    ("round1", "query-p1-20-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                        kiem_lai_lane="chua_kiem",
                                        bang_chung="File picks tự đánh dấu YẾU: điểm cao nhất chỉ 0,35."),
    ("round1", "query-p1-25-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                        kiem_lai_lane="chua_kiem",
                                        bang_chung="File picks tự đánh dấu YẾU: điểm cao nhất chỉ 0,40."),
    ("round1", "query-p1-3-qa"): dict(kenh_video="engine", kenh_khung="khong",
                                      kiem_lai_lane="chua_kiem",
                                      bang_chung="Đáp án 'tham khảo của đội bạn', không tự kiểm chứng."),

    # ------------------------------------------------------------ sơ tuyển 2
    ("round2", "query-p2-12-qa"): dict(
        kenh_video="tieu_de", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        dap_an_kiem="khop",
        bang_chung="Tiêu đề 'BÁNH KHỌT LÁ CẨM NHÂN NẤM HẠT SEN' — khớp bánh màu tím và "
                   "hạt sen. LANE MỞ KHUNG: khuôn có đúng 7 lỗ xếp 2-3-2, mỗi lỗ một hạt "
                   "sen, có cà rốt sợi.",
    ),
    ("round2", "query-p2-17-kis"): dict(
        kenh_video="chu_tren_hinh", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        bang_chung="LANE MỞ KHUNG: chữ nổi 3D phủ kim tuyến đặt ở mép trước sân khấu, "
                   "đọc được '…C CỔ … XƯA'. Tiêu đề 'Sinh viên dựng kịch nói tôn vinh "
                   "Việt phục' củng cố.",
        ghi_chu="Hai diễn viên che mất vài chữ; phần đọc được khớp mẫu 'SẮC CỔ …'.",
    ),
    ("round2", "query-p2-20-kis"): dict(
        kenh_video="tieu_de", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        bang_chung="Tiêu đề 'Môn Địa lý — Chuyên đề 4 — Địa lý dân cư Việt Nam'. "
                   "LANE MỞ KHUNG: bảng 'Phân bố đô thị không đồng đều giữa các vùng', "
                   "ba số viền ĐỎ (172/124/148), hai số viền XANH (58/47), và vùng ít đô "
                   "thị nhất (47) đúng là vùng có dân số đô thị cao nhất (10493,2) — "
                   "khớp cả ba mệnh đề của đề.",
    ),
    ("round2", "query-p2-22-kis"): dict(
        kenh_video="tieu_de", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        bang_chung="Tiêu đề 'MỰC QUE CHIÊN GIÒN XỐT CAM' — khớp 'cắt thành từng que'. "
                   "LANE MỞ KHUNG: miếng mực trắng khứa vuông góc hai mặt đang được cắt "
                   "thành que trên thớt.",
    ),
    ("round2", "query-p2-26-kis"): dict(
        kenh_video="tieu_de", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        bang_chung="Tiêu đề 'Môn GDCD — Các quy luật sản xuất và lưu thông hàng hoá'. "
                   "LANE MỞ KHUNG: slide 'NỘI DUNG 2 — QUY LUẬT CẠNH TRANH' chứa CẢ HAI "
                   "hình đề đòi (nhóm người 3D trắng vây quanh một nhân vật đỏ, và hai "
                   "nhân vật hoạt hình kéo co) trong cùng một khung.",
    ),
    ("round2", "query-p2-29-qa"): dict(
        kenh_video="tieu_de", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        dap_an_kiem="khop",
        bang_chung="Tiêu đề 'ỐC XÀO LĂN LÁ CÁCH'. LANE MỞ KHUNG: bảng NGUYÊN LIỆU đúng 9 "
                   "dòng, dòng 1 'Thịt ốc lác: 300g'; nền có bó lá xanh góc trái, gói hạt "
                   "nêm, hũ nước cốt dừa, nấm mèo khô, sả cây, ớt hiểm — khớp toàn bộ đề.",
    ),
    ("round2", "query-p2-7-qa"): dict(
        kenh_video="chu_tren_hinh", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        dap_an_kiem="tranh_chap",
        bang_chung="LANE MỞ KHUNG: xe trắng có số '1204' trên hông, phía trên khung là "
                   "biển hiệu ĐỎ gồm đúng 6 ký tự chữ Hán (复合宴会中心) — đúng tổ hợp đề tả.",
        ghi_chu="ĐÁP ÁN ĐANG TRANH CHẤP: lượt cuối ghi 1204, nhưng có một lượt người soát "
                "khẳng định '5' và cho rằng 1204 là xe khác trong cùng phóng sự. Khung "
                "hình lane mở ra ủng hộ 1204, nhưng chưa đủ để đóng tranh chấp — nhãn "
                "TRUY XUẤT dùng được, nhãn ĐÁP ÁN thì không.",
    ),
    ("round2", "query-p2-27-qa"): dict(
        kenh_video="chu_tren_hinh", kenh_khung="chu_tren_hinh", kiem_lai_lane="khop",
        dap_an_kiem="chua_kiem", bo_qua_mau_thuan=True,
        bang_chung="LANE MỞ KHUNG: lân biểu diễn trên các trụ CÓ DÁN SỐ, phía sau có mô "
                   "hình rồng uốn lượn — đúng đề; khung ở giây 10,1 nên nằm trong 16 giây "
                   "đầu. Ứng viên đối thủ đã được LOẠI bằng cách mở khung của nó: 16 giây "
                   "đầu của video kia là cảnh khiêng bao tải cạnh lò gạch, không có trụ số.",
        ghi_chu="ĐÁP ÁN KHÔNG ĐÁNG TIN: các lượt ghi lần lượt '1 2 8' rồi '2'. Khung lane "
                "mở ra thấy rõ các số 1, 3, 4, 6, 7, 8, 9 — tức 1 và 8 CÓ nhìn thấy, mâu "
                "thuẫn với đáp án '1 2 8'; thiếu 2 và 5 ở riêng khung này.",
    ),
    ("round2", "query-p2-19-qa"): dict(
        kenh_video="tieu_de", kenh_khung="chu_tren_hinh", kiem_lai_lane="mot_phan",
        dap_an_kiem="khop",
        bang_chung="Mô tả video của BTC: 'Quán trọ TP.HCM bao dung ... hoạt động từ 2020'. "
                   "LANE MỞ KHUNG: bảng hiệu ghi '552 Lý Thường Kiệt, P7, Tân Bình, HCM'.",
        ghi_chu="Khung này chứng minh ĐÁP ÁN (tên đường) chứ không phải CẢNH đề tả (mạnh "
                "thường quân trao hỗ trợ). Nhãn đáp án dùng được, nhãn truy xuất thì chưa.",
    ),
    ("round2", "query-p2-9-qa"): dict(
        kenh_video="tieu_de", kenh_khung="vlm_cham", kiem_lai_lane="khong_khop",
        dap_an_kiem="khop",
        bang_chung="Tiêu đề 'CÁ SÒNG NƯỚNG MUỐI HẠT' — chốt thẳng loài cá, tức ĐÁP ÁN.",
        ghi_chu="LANE MỞ KHUNG: khung chốt là cảnh sơ chế cá trong tô, chưa phải cảnh "
                "nhồi tiêu xanh/lá chanh/sả vào bụng 4 con cá.",
    ),
    ("round2", "query-p2-30-qa"): dict(
        kenh_video="engine", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        dap_an_kiem="khop",
        bang_chung="Cùng video với p2-9; tiêu đề chốt loài cá nên ĐÁP ÁN đáng tin, "
                   "nhưng video của riêng câu này chưa có kênh độc lập nào chốt.",
    ),
    ("round2", "query-p2-10-kis"): dict(
        kenh_video="tieu_de", kenh_khung="moc_loi_thoai", kiem_lai_lane="khong_khop",
        bang_chung="Tiêu đề 'Dồi trường xào Bông hẹ' — khớp đúng hai nguyên liệu của đề.",
        ghi_chu="LANE MỞ KHUNG: khung chốt (suy từ mốc lời thoại 2:40) chỉ thấy phi hành "
                "trong chảo, chưa có dồi trường trắng và bông hẹ. Video chắc, frame sai.",
    ),
    ("round2", "query-p2-24-kis"): dict(
        kenh_video="engine", kenh_khung="vlm_cham", kiem_lai_lane="khong_khop",
        bang_chung="Hai lượt chọn hai video khác nhau (một video 'chặng 4 cho Vĩnh Long', "
                   "một video 'tại Quảng trường Thống Nhất'); đề nói đích đến ở một thành "
                   "phố thuộc Quảng Nam cũ — không tiêu đề nào xác nhận điều đó.",
        ghi_chu="LANE MỞ KHUNG khung chốt: không thấy tay đua áo xanh buông hai tay ăn mừng.",
    ),
    ("round2", "query-p2-2-kis"): dict(
        kenh_video="loi_thoai", kenh_khung="moc_loi_thoai", kiem_lai_lane="chua_kiem",
        bang_chung="Hai bản tin cùng đưa loạt tranh Banksy (tê giác, khỉ): một video nói "
                   "'tê giác trèo lên nóc ô tô hỏng' @742 s, video kia nói 'bức tranh thứ "
                   "chín ... tại cổng vào sở thú' @879 s. Hai lượt picks chọn hai video này.",
        ghi_chu="Mâu thuẫn CHƯA ĐÓNG. Đây là câu đáng soát nhất trong nhóm suy_ra: cả hai "
                "ứng viên đều có lời thoại đỡ, chỉ cần mở khung là chốt được.",
    ),
    ("round2", "query-p2-4-kis"): dict(
        kenh_video="engine", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Người soát lập luận băng rôn trường học thuộc video thiện nguyện chứ "
                   "không phải bài giảng. Cùng video với câu p1-15-qa vòng luyện tập.",
    ),
    ("round2", "query-p2-6-kis"): dict(
        kenh_video="loai_tru", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Chỉ bản tin buổi SÁNG có 2 vòng tròn đỏ khoanh như đề (bản chiều 1 vòng).",
    ),
    ("round2", "query-p2-8-trake"): dict(
        kenh_video="loi_thoai", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Tiêu đề video được chọn cuối = tập quay ở miệt vườn, khớp 'khu vườn "
                   "cây ăn trái miền Tây'.",
        ghi_chu="ĐỔI VIDEO HAI LẦN trong ba lượt, và lượt giữa khẳng định video cuối "
                "'không có sầu riêng/măng cụt/bưởi nào'. Hai khẳng định loại trừ nhau.",
    ),
    ("round2", "query-p2-11-kis"): dict(kenh_video="engine", kenh_khung="vlm_cham",
                                        kiem_lai_lane="chua_kiem",
                                        bang_chung="gpt xác minh video có cả hai giai đoạn lăn bột."),
    ("round2", "query-p2-14-kis"): dict(kenh_video="engine", kenh_khung="engine",
                                        kiem_lai_lane="chua_kiem",
                                        bang_chung="'khung cận cảnh tốt nhất của video hạng 1' — hoàn toàn engine."),
    ("round2", "query-p2-16-kis"): dict(kenh_video="engine", kenh_khung="vlm_cham",
                                        kiem_lai_lane="chua_kiem",
                                        bang_chung="VLM chấm 1.00 cho 3 vùng cạnh thớt gỗ."),
    ("round2", "query-p2-1-kis"): dict(
        kenh_video="tieu_de", kenh_khung="vlm_cham", kiem_lai_lane="chua_kiem",
        bang_chung="Cùng video với câu p1-19-kis vòng sơ tuyển 1 (tiêu đề 'Nam Sư Du Hí "
                   "Ăn Bông Bí'), khớp tiểu phẩm giấu quả bí. Đây là một CHỖ TRÙNG giữa "
                   "hai vòng đề — hai câu khác nhau chỉ về một video.",
    ),
    ("round2", "query-p2-18-kis"): dict(kenh_video="engine", kenh_khung="vlm_cham",
                                        kiem_lai_lane="chua_kiem",
                                        bang_chung="Ba lượt đổi qua đổi lại giữa hai video đua xe đạp."),
    ("round2", "query-p2-21-trake"): dict(kenh_video="engine", kenh_khung="vlm_cham",
                                          kiem_lai_lane="chua_kiem",
                                          bang_chung="Chuỗi 4 cảnh liên tiếp trong 6 giây, chọn bằng mắt trên trang review."),
    ("round2", "query-p2-23-qa"): dict(kenh_video="engine", kenh_khung="moc_loi_thoai",
                                       kiem_lai_lane="chua_kiem",
                                       bang_chung="Suy từ lời giảng 'từ 0 tới 15 phần nghìn ... tăng lên'."),
    ("round2", "query-p2-28-qa"): dict(kenh_video="engine", kenh_khung="vlm_cham",
                                       kiem_lai_lane="chua_kiem",
                                       bang_chung="2/3 khung đọc ra 'chà bông/ruốc tôm'."),
}

#: Câu KHÔNG có pick nào — bài nộp giữ nguyên đầu ra engine. Ghi lại để người
#: soát biết chỗ trống, không sinh mục ground truth.
KHONG_CO_PICK_GHI_CHU = "không có pick nào; bài nộp giữ nguyên đầu ra engine"


# ---------------------------------------------------------------------------
# Đọc dữ liệu nền
# ---------------------------------------------------------------------------


def nap_metadata(data_dir: Path):
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    theo_video = defaultdict(list)
    for r in meta:
        theo_video[r["video_id"]].append(r)
    for v in theo_video:
        theo_video[v].sort(key=lambda r: r["frame_idx"])
    return theo_video


def nap_tieu_de(data_dir: Path):
    """Tiêu đề + mô tả của cả 873 video, từ gói media-info của BTC."""
    z = data_dir / "media-info-aic25-b1.zip"
    if not z.exists():
        return {}
    out = {}
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if not n.endswith(".json"):
                continue
            vid = n.rsplit("/", 1)[-1][:-5]
            try:
                d = json.loads(zf.read(n).decode("utf-8"))
            except Exception:  # noqa: BLE001
                continue
            out[vid] = {"title": d.get("title", "") or "", "description": d.get("description", "") or ""}
    return out


def nap_loi_thoai(data_dir: Path):
    """Lời thoại có mốc thời gian. CHÚ Ý: repo này chỉ có 217/873 video có nội
    dung thật (các file còn lại tồn tại nhưng RỖNG) — nên 'không tìm thấy' ở đây
    KHÔNG phải bằng chứng phủ định."""
    out = {}
    d = data_dir / "captions"
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if raw:
            out[p.stem] = [(float(a), str(b)) for a, b in raw]
    return out


def chuan(s: str) -> str:
    return unicodedata.normalize("NFC", str(s or "")).lower()


# ---------------------------------------------------------------------------
# Đọc file picks
# ---------------------------------------------------------------------------

_DONG_PICK = re.compile(r"^\s*(query-[\w\-]+)\s*=\s*(.*)$")


def doc_picks(path: Path):
    """[(dòng, mã câu, video, [frame], đáp án, lý do ghi ngay trên nó)]"""
    if not path.exists():
        return []
    out = []
    ly_do: list[str] = []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if ln.strip().startswith("#"):
            ly_do.append(ln.strip().lstrip("#").strip())
            continue
        m = _DONG_PICK.match(ln)
        if not m:
            if not ln.strip():
                ly_do = []
            continue
        q, rest = m.group(1), m.group(2)
        parts = rest.split(":", 2)
        video = parts[0].strip()
        raw = parts[1].strip() if len(parts) > 1 else ""
        frames = [int(x) for x in raw.split("|") if x.strip()] if raw else []
        answer = parts[2].strip() if len(parts) > 2 else ""
        out.append(
            {
                "dong": i,
                "ma": q,
                "video_id": video,
                "frames": frames,
                "dap_an": answer,
                "ly_do": " / ".join(ly_do[-4:]),
            }
        )
        ly_do = []
    return out


# ---------------------------------------------------------------------------
# Luật xếp độ tin — đây là chỗ duy nhất quyết định mục nào được dùng để chấm
# ---------------------------------------------------------------------------

KENH_VIDEO_DOC_LAP = {"tieu_de", "loi_thoai", "chu_tren_hinh", "loai_tru"}


def xep_do_tin(nhan: dict, co_mau_thuan: bool, co_frame: bool) -> tuple[str, list[str]]:
    """Nhãn TRUY XUẤT (video_id + frame_idx) có được dùng để chấm không.

    Bốn điều kiện, mỗi điều kiện chặn một kiểu sai đã gặp thật:

    1. VIDEO phải do một kênh ĐỘC LẬP với SigLIP chốt — nếu không thì chấm đường
       truy xuất trên chính đầu ra của nó (vòng lặp, nền bị thổi lên).
    2. Pick phải CÓ frame — cửa sổ chấm của BTC là (video ĐÚNG *và* frame trong
       cửa sổ), nhãn chỉ-có-video không chấm được.
    3. Lane này phải đã MỞ khung hình ra xem và thấy khớp. Bốn mục có bằng chứng
       nghe rất chắc đã bị chính bước này bác bỏ.
    4. Không được còn mâu thuẫn giữa các lượt chọn (trừ khi mâu thuẫn đã được
       đóng bằng cách mở khung của ứng viên kia — cờ ``bo_qua_mau_thuan``).
    """
    ly_do = []
    if nhan.get("kenh_video") not in KENH_VIDEO_DOC_LAP:
        ly_do.append("video do engine/VLM chốt, không có kênh độc lập")
    if not co_frame:
        ly_do.append("pick không ghi frame")
    if nhan.get("kiem_lai_lane") != "khop":
        ly_do.append(f"lane mở khung: {nhan.get('kiem_lai_lane', 'chua_kiem')}")
    if co_mau_thuan and not nhan.get("bo_qua_mau_thuan"):
        ly_do.append("các lượt chọn mâu thuẫn nhau")
    return ("nguoi_kiem_chung" if not ly_do else "suy_ra"), ly_do


def xep_do_tin_dap_an(nhan: dict, dap_an: str) -> str | None:
    if not dap_an:
        return None
    return "nguoi_kiem_chung" if nhan.get("dap_an_kiem") == "khop" else "suy_ra"


# ---------------------------------------------------------------------------
# Dựng một mục ground truth
# ---------------------------------------------------------------------------


def tra_keyframe(theo_video, video_id: str, frame_idx: int):
    a = theo_video.get(video_id)
    if not a:
        return None
    r = min(a, key=lambda x: abs(int(x["frame_idx"]) - int(frame_idx)))
    return r


def dung_muc(vong, ma, qfile: Path, pick, cac_luot, nhan, theo_video, tieu_de, loi_thoai):
    text = read_query_text(qfile) or ""
    task = detect_task(qfile.stem)
    en = read_en_override(qfile)

    if task == "qa":
        canh, cau_hoi = split_qa(text)
    elif task == "trake":
        canh, cau_hoi = text.strip(), ""
    else:
        canh, cau_hoi = text.strip(), ""

    frames = list(pick["frames"])
    frame_idx = frames[0] if frames else None

    muc = {
        "ma": f"{vong}/{ma}",
        "dang": task,
        "kis_query_vi": canh,
        "kis_query_en": en or "",
        "vqa_context": canh if task == "qa" else "",
        "vqa_question": cau_hoi,
        "vqa_answer": pick["dap_an"],
        "video_id": pick["video_id"],
        "frame_idx": frame_idx,
    }

    canh_bao = []
    kf = tra_keyframe(theo_video, pick["video_id"], frame_idx) if frame_idx is not None else None
    if pick["video_id"] not in theo_video:
        canh_bao.append("video_id KHÔNG có trong metadata")
    if kf is not None:
        muc["n"] = int(kf["n"])
        muc["frame_filename"] = kf["frame_filename"]
        muc["pts_time"] = float(kf["pts_time"])
        muc["cdn_url"] = f"{CDN}/{pick['video_id']}/{kf['frame_filename']}"
        lech = int(kf["frame_idx"]) - int(frame_idx)
        if lech != 0:
            muc["keyframe_gan_nhat"] = int(kf["frame_idx"])
            muc["lech_keyframe"] = lech
            if abs(lech) > 60:
                canh_bao.append(f"frame_idx cách keyframe gần nhất {lech} khung")
    else:
        muc.update({"n": None, "frame_filename": "", "pts_time": None, "cdn_url": ""})

    if task == "trake":
        muc["trake_frames"] = frames
        muc["trake_su_kien"] = split_events(text)
        if frames and len(frames) != len(muc["trake_su_kien"]):
            canh_bao.append(
                f"số mốc ({len(frames)}) khác số sự kiện đọc được từ đề "
                f"({len(muc['trake_su_kien'])})"
            )

    videos = {p["video_id"] for _f, p in cac_luot}
    co_mau_thuan = len(videos) > 1
    do_tin, ly_do = xep_do_tin(nhan, co_mau_thuan, bool(frames))

    td = tieu_de.get(pick["video_id"], {})
    kiem_loi_thoai = None
    if nhan.get("loi_thoai_can"):
        pat = nhan["loi_thoai_can"]
        khop = {v for v, segs in loi_thoai.items() if any(re.search(pat, chuan(x)) for _t, x in segs)}
        kiem_loi_thoai = {
            "mau": pat,
            "so_video_khop": len(khop),
            "co_trong_video_nay": pick["video_id"] in khop,
            "pham_vi": f"{len(loi_thoai)}/873 video có transcript trong repo",
        }
        if not kiem_loi_thoai["co_trong_video_nay"]:
            canh_bao.append("cụm từ bằng chứng KHÔNG tìm thấy trong transcript của video này")

    muc.update(
        {
            "do_tin": do_tin,
            "do_tin_dap_an": xep_do_tin_dap_an(nhan, pick["dap_an"]),
            "ly_do_khong_dat": ly_do,
            "nguon": [f"{f}:{p['dong']}" for f, p in cac_luot],
            "nguon_de": str(qfile.relative_to(ROOT)).replace("\\", "/"),
            "kenh_video": nhan.get("kenh_video"),
            "kenh_khung": nhan.get("kenh_khung"),
            "kiem_lai_lane": nhan.get("kiem_lai_lane", "chua_kiem"),
            "bang_chung": nhan.get("bang_chung", ""),
            "ghi_chu": nhan.get("ghi_chu", ""),
            "ly_do_trong_file_picks": pick["ly_do"],
            "cac_luot_chon": [
                {"file": f, "dong": p["dong"], "video_id": p["video_id"],
                 "frames": p["frames"], "dap_an": p["dap_an"]}
                for f, p in cac_luot
            ],
            "mau_thuan_video": sorted(videos) if co_mau_thuan else [],
            "tieu_de_video": td.get("title", ""),
            "kiem_loi_thoai": kiem_loi_thoai,
            "canh_bao": canh_bao,
        }
    )
    return muc


# ---------------------------------------------------------------------------
# Sai số nhị thức
# ---------------------------------------------------------------------------


def bang_sai_so(n_test_cu: int, them: list[int]):
    import math

    dong = []
    for n in [n_test_cu] + [n_test_cu + t for t in them]:
        sd = math.sqrt(0.25 / n)
        dong.append((n, sd, 2 * sd, math.sqrt(n_test_cu / n)))
    return dong


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "data" / "ground_truth_de_that.json"))
    ap.add_argument("--merge-out", default=str(ROOT / "data" / "ground_truth_hop_nhat.json"))
    ap.add_argument("--in-bang", action="store_true", help="in bảng từng mục")
    args = ap.parse_args()

    data_dir = Path(args.data)
    theo_video = nap_metadata(data_dir)
    tieu_de = nap_tieu_de(data_dir)
    loi_thoai = nap_loi_thoai(data_dir)
    print(f"nền: {len(theo_video)} video / {sum(len(v) for v in theo_video.values())} keyframe, "
          f"{len(tieu_de)} tiêu đề, {len(loi_thoai)} transcript có nội dung")

    muc_all = []
    thieu_nhan = []
    khong_pick = []
    trung_de = defaultdict(list)

    for vong, cfg in VONG.items():
        qdir = ROOT / cfg["queries"]
        if not qdir.is_dir():
            print(f"  !! bỏ qua {vong}: không thấy {qdir}")
            continue
        # gom picks theo mã câu, giữ thứ tự file (lượt sau đè lượt trước)
        theo_ma = defaultdict(list)
        for pf in cfg["picks"]:
            for p in doc_picks(ROOT / pf):
                theo_ma[p["ma"]].append((pf, p))

        qfiles = [p for p in sorted(qdir.glob("*.txt")) if not p.name.endswith(".en.txt")]
        for qfile in qfiles:
            ma = qfile.stem
            txt = read_query_text(qfile) or ""
            trung_de[hashlib.md5(txt.strip().encode("utf-8")).hexdigest()].append(f"{vong}/{ma}")
            cac_luot = theo_ma.get(ma, [])
            if not cac_luot:
                khong_pick.append(f"{vong}/{ma}")
                continue
            pick = cac_luot[-1][1]  # quyết định cuối cùng của người soát
            nhan = NHAN.get((vong, ma))
            if nhan is None:
                thieu_nhan.append(f"{vong}/{ma}")
                nhan = {"kenh_video": "engine", "kenh_khung": "engine",
                        "kiem_lai_lane": "chua_kiem",
                        "bang_chung": "CHƯA XẾP NHÃN — mặc định coi như engine dẫn đầu."}
            muc_all.append(dung_muc(vong, ma, qfile, pick, cac_luot, nhan,
                                    theo_video, tieu_de, loi_thoai))

    # đề trùng nhau từng byte -> chỉ giữ mục đầu để không đếm hai lần
    trung = {h: v for h, v in trung_de.items() if len(v) > 1}
    bo_vi_trung = set()
    for _h, ds in trung.items():
        for d in ds[1:]:
            bo_vi_trung.add(d)
    for m in muc_all:
        if m["ma"] in bo_vi_trung:
            m["trung_de_voi"] = [d for d in next(v for v in trung.values() if m["ma"] in v)
                                 if d != m["ma"]]
            if m["do_tin"] == "nguoi_kiem_chung":
                m["do_tin"] = "suy_ra"
                m["ly_do_khong_dat"].append("đề trùng byte với một câu khác, giữ bản đầu")

    tot = [m for m in muc_all if m["do_tin"] == "nguoi_kiem_chung"]
    xau = [m for m in muc_all if m["do_tin"] != "nguoi_kiem_chung"]

    goi = {
        "quy_uoc": {
            "do_tin": {
                "nguoi_kiem_chung": "người soát ra quyết định TRONG VÒNG THI và ghi bằng "
                                    "chứng kiểm được, VÀ lane harness đã mở khung hình "
                                    "1280px xem lại thấy khớp, VÀ video được một kênh độc "
                                    "lập với SigLIP chốt, VÀ không còn mâu thuẫn giữa các "
                                    "lượt. CHỈ nhóm này được dùng để chấm.",
                "suy_ra": "mọi thứ còn lại. GHI RA để người soát sau xử lý, KHÔNG dùng chấm.",
            },
            "do_tin_dap_an": "độ tin của riêng trường vqa_answer, tách khỏi nhãn truy xuất: "
                             "một câu có thể có khung hình chắc mà đáp án vẫn sai (và ngược lại).",
            "canh_bao_lon": "KHÔNG mục nào ở đây được BTC xác nhận. Điểm công bố của cả ba "
                            "vòng (8,6/24 và 10,0/30) nói rằng phần lớn pick trong các file "
                            "nguồn SAI. 'nguoi_kiem_chung' nghĩa là bằng chứng đứng vững khi "
                            "mở khung hình ra xem, không phải BTC chấm đúng.",
        },
        "thong_ke": {
            "tong_muc": len(muc_all),
            "nguoi_kiem_chung": len(tot),
            "suy_ra": len(xau),
            "theo_dang_nguoi_kiem_chung": {
                d: sum(1 for m in tot if m["dang"] == d) for d in ("kis", "qa", "trake")
            },
            "dap_an_nguoi_kiem_chung": sum(1 for m in muc_all
                                           if m.get("do_tin_dap_an") == "nguoi_kiem_chung"),
            "cau_khong_co_pick": khong_pick,
            "cau_chua_xep_nhan": thieu_nhan,
            "de_trung_nhau": {h: v for h, v in trung.items()},
        },
        "muc": muc_all,
    }
    Path(args.out).write_text(json.dumps(goi, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- file hợp nhất: 60 câu cũ + các mục nguoi_kiem_chung có frame, KHÔNG TRAKE
    gt60 = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    truong = ["kis_query_vi", "kis_query_en", "vqa_context", "vqa_question", "vqa_answer",
              "video_id", "frame_idx", "n", "frame_filename", "pts_time", "cdn_url"]
    them = []
    for m in tot:
        if m["dang"] == "trake" or m["frame_idx"] is None:
            continue
        r = {k: m.get(k) for k in truong}
        r["nguon_de_that"] = m["ma"]
        r["do_tin"] = m["do_tin"]
        r["do_tin_dap_an"] = m["do_tin_dap_an"]
        them.append(r)
    hop = gt60 + them
    Path(args.merge_out).write_text(json.dumps(hop, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- báo cáo ----------------------------------------------------------
    print("\n=== THU HOẠCH ===")
    print(f"  đọc {len(muc_all)} mục từ 3 vòng ({len(khong_pick)} câu không có pick nào)")
    print(f"  nguoi_kiem_chung : {len(tot)}")
    for d in ("kis", "qa", "trake"):
        n = sum(1 for m in tot if m["dang"] == d)
        print(f"      {d:6s}: {n}")
    print(f"  suy_ra           : {len(xau)}")
    print(f"  đáp án đạt nguoi_kiem_chung: {goi['thong_ke']['dap_an_nguoi_kiem_chung']}")
    if trung:
        print(f"  đề TRÙNG NHAU từng byte: {[v for v in trung.values()]}")
    if thieu_nhan:
        print(f"  chưa xếp nhãn (mặc định engine): {len(thieu_nhan)} câu")

    print("\n=== BÁC BỎ KHI MỞ KHUNG (bằng chứng cũ không đứng vững) ===")
    for m in muc_all:
        if m["kiem_lai_lane"] == "khong_khop":
            print(f"  {m['ma']:28s} {m['video_id']}:{m['frame_idx']}")
            print(f"      {m['ghi_chu'][:150]}")

    print("\n=== MÂU THUẪN GIỮA CÁC LƯỢT (ít nhất một lựa chọn phải sai) ===")
    for m in muc_all:
        if m["mau_thuan_video"]:
            print(f"  {m['ma']:28s} {' vs '.join(m['mau_thuan_video'])}")

    print("\n=== SAI SỐ CHUẨN (nhị thức, p=0,5 — trường hợp xấu nhất) ===")
    print("  Giả định: mỗi câu là một Bernoulli độc lập cùng xác suất; độ đo là TỈ LỆ")
    print("  (vd: tỉ lệ trả lời đúng Q&A, tỉ lệ câu có video đúng trong 100 dòng).")
    print("  Chia TUNE/TEST theo chỉ số chẵn/lẻ nên N mục mới rơi ~N/2 vào TEST.")
    n_them_test = len(them) // 2
    print(f"\n  {'n TEST':>8}{'1 sd':>9}{'cổng 2 sd':>12}{'so với n=30':>14}")
    for n, sd, hai, ti in bang_sai_so(30, [n_them_test, 15, 30, 90]):
        print(f"  {n:8d}{100*sd:8.2f}%{100*hai:11.2f}%{ti:13.3f}")
    print("\n  Đọc bảng: cột cuối là hệ số ngưỡng so với hiện tại. Muốn hạ ngưỡng")
    print("  1,41 lần thì TEST phải đi từ 30 lên 60 câu, tức phải thu thêm ~60 câu")
    print("  đạt nguoi_kiem_chung — gấp bốn lần những gì lượt này thu được.")

    if args.in_bang:
        print("\n=== TỪNG MỤC ===")
        for m in muc_all:
            print(f"  [{m['do_tin']:16s}] {m['ma']:28s} {m['dang']:5s} "
                  f"{m['video_id']}:{m['frame_idx']}  {m['tieu_de_video'][:40]}")
            if m["ly_do_khong_dat"]:
                print(f"      vì: {'; '.join(m['ly_do_khong_dat'])}")

    print(f"\nĐã ghi: {args.out}")
    print(f"Đã ghi: {args.merge_out}  ({len(gt60)} câu cũ + {len(them)} câu đề thật)")
    print("Cả hai nằm dưới data/ nên .gitignore (dòng 'data/*') đã chặn — đừng ép commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
