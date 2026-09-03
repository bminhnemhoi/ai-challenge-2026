# Quy trình làm việc tự động — harness cho phiên không có người trực

Viết 01/09/2026. Đây là **luật ràng buộc**, không phải gợi ý: mọi việc làm trong
phiên tự động phải kiểm được lại theo tài liệu này.

---

## 0. Nguyên tắc gốc

> Mục tiêu không phải là làm nhiều việc. Mục tiêu là **mỗi thay đổi vào sản xuất
> đều có một con số đứng sau, và mỗi con số đều có một file cache đứng sau.**

Một phiên tự động kết thúc với 3 kết luận âm có số liệu là phiên **thành công**:
nó đóng ba cửa, tiết kiệm hàng tuần công sức về sau. Một phiên kết thúc với 5
thay đổi chưa đo là phiên **thất bại**, dù trông có vẻ năng suất.

---

## 1. Thứ tự ưu tiên (chọn việc tiếp theo theo đúng thứ tự này)

1. **Sửa thứ đang hỏng.** Test đỏ, đường sản xuất chạy lỗi, số liệu trong tài
   liệu mâu thuẫn với thực tế → sửa trước mọi thứ khác.
2. **Xác minh thứ vừa ship.** Mã sản xuất chưa từng chạy end-to-end là mã chưa
   ship. Bài học: đường Q&A và cờ `--canh-b` đều suýt được coi là "xong" khi mới
   chỉ có unit test.
3. **Việc nhắm vào nghẽn đã lượng hoá.** Hiện tại: định vị nội-video, trần
   oracle **+126%** (`docs/BO_DO_KHOP_PHAN_BO.md`).
4. **Mở lại cửa đã đóng bằng bộ đo cũ.** Bảng tín hiệu có ~35 dòng, phần lớn đo
   trên bộ 60 câu cũ — bộ đo *không nhìn thấy* nghẽn hiện tại.
5. **Nghiên cứu paper/công nghệ mới**, nhưng chỉ hướng nhắm vào (3).
6. Hạ tầng và tài liệu.

**Không bao giờ** làm việc ở bậc thấp khi bậc cao còn tồn đọng.

---

## 2. Cổng bắt buộc trước khi đưa BẤT KỲ thứ gì vào sản xuất

Đủ **cả năm**, thiếu một là không ship:

| # | cổng | vì sao |
|---|---|---|
| 1 | Chia TUNE/TEST **phân tầng** theo nhóm bị tác động | chia chẵn/lẻ thô từng cho TUNE 16/16 câu qua cổng và TEST 0/24 — số TEST khi đó là phép đồng nhất mà script vẫn in ra con số trông bình thường |
| 2 | Chọn trên TUNE, đọc TEST **đúng một lần** | mọi thứ khác là tự chấm bài mình |
| 3 | **Bootstrap theo câu**, không phải 2σ hạt giống | σ hạt giống đo nhiễu bốc thăm; tăng số lần bốc là nó nhỏ đi dù chẳng biết thêm gì về câu hỏi. Đã có ca 2σ nói "GIỮ ĐƯỢC" còn bootstrap nói 14% khả năng hoà |
| 4 | **Bất biến bằng assert**: câu không thuộc diện tác động ra 100 dòng giống hệt nền | chặn cứng thiệt hại tối đa |
| 5 | Chấm qua `allocate_rows` **thật** của make_submission | chấm tắt đo một hệ thống không tồn tại |

**Ngoại lệ duy nhất** cho cổng 3: khi có **bằng chứng cơ chế tất định** (một phép
đếm, không phải ước lượng) *cộng* đường tham số phẳng *cộng* rủi ro có chặn cứng.
Tiền lệ: cảnh B — đếm được keyframe đáp án có mặt trong pool 53% → 76%. Khi dùng
ngoại lệ này phải ghi rõ cả ba điều kiện trong tài liệu.

---

## 3. Sau khi ship — bắt buộc, không được bỏ

1. **Smoke-test end-to-end đường sản xuất thật** (không phải unit test).
2. **Diff cấu trúc trên đề thật** — nó đổi những gì trên phân bố BTC thực sự ra?
   Đặc biệt đếm **video dòng 1 có đổi không**: R@1 đáng 1,0, đắt nhất.
3. Đối chiếu các câu bị đổi với đáp án đã biết (`data/ground_truth_de_that.json`,
   chỉ dùng mục `nguoi_kiem_chung`).
4. Ghi **đường rút lui một cờ** vào tài liệu và runbook.
5. Cập nhật bảng tín hiệu `docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md`.

---

## 4. Luật riêng cho phiên tự động

- **Không chạy hai workflow cùng lúc.** Chờ cái đang chạy xong rồi mới phóng cái
  mới — các lane sửa file chung sẽ đâm nhau.
- **Lane song song chỉ được TẠO FILE MỚI.** Tích hợp sản xuất làm tuần tự, do
  người điều phối (tôi) làm sau khi lane xong.
- **Commit thường xuyên, mỗi commit một ý.** Thông điệp commit phải chứa con số
  và cách tái lập, vì đó là thứ người đọc lại sau này cần.
- **Không bao giờ** `git add -f` bất cứ thứ gì dưới `data/`, `round1/`,
  `round2/`, `round_p1/` — đó là đề thi và đáp án. `.gitignore` đã chặn sẵn.
- **Không in API key**, không commit `.env`.
- Việc cần **quyết định của người** thì DỪNG và ghi vào báo cáo, không tự quyết:
  cookie tài khoản YouTube, xoá dữ liệu, đổi mặc định có rủi ro chưa đo được.
- Quota Gemini free hết thì **chuyển sang việc thuần CPU**, đừng ngồi chờ.
  Có rất nhiều việc CPU: oracle, chẩn đoán, quét tham số, phân tích.
- Chi tiền OpenAI: chỉ cho việc **hạ tầng dùng lâu dài** (xác minh neo bộ đo) và
  chỉ khi rẻ hơn $1/lượt. Đã đo: gpt-5.2 **thua** Gemini free ở bước trả lời.

---

## 5. Vòng lặp làm việc

```
1. Đọc trạng thái: git log -5, kiểm tra background task, đọc bảng tín hiệu.
2. Chọn việc theo mục 1 (ưu tiên cao nhất còn tồn đọng).
3. Nếu là thí nghiệm: viết script MỚI, chạy, qua đủ 5 cổng ở mục 2.
4. Nếu ăn: tích hợp sản xuất + làm đủ mục 3. Nếu hoà/âm: ghi vào bảng cửa đóng.
5. Commit kèm số. Cập nhật tài liệu.
6. Ghi memory nếu là bài học dùng lại được ở phiên sau.
7. Quay lại bước 1.
```

**Mỗi vòng phải kết thúc bằng một commit** — hoặc mã, hoặc một kết luận có số.
Vòng nào không sinh ra được cái nào thì ghi rõ lý do trong báo cáo.

---

## 6. Trạng thái hiện tại (cập nhật 03/09/2026, sau ba lane trake-them / pe-core / paper)

**Đã ship, có số đứng sau:**

| thay đổi | số | cờ rút lui |
|---|---|---|
| bộ phân bổ phủ xác suất | +15,3%/+16,0% TEST | `--allocator hybrid` |
| đường Q&A đa kênh | 70,0% → 93,3% đáp án đúng | `--neo 0` |
| `reserve_tail_rows` | vá lỗ hổng mất p1-19/p1-22 | — |
| truy xuất thêm cảnh B | +23,3% nhóm qua cổng; pool 53%→76% | `--canh-b 0` |
| hoán vị điểm nội-video theo cảnh B | +57,6% cả 66 câu hai cảnh, hạt độc lập; bền qua 4 mô hình bốc | `--hoan-vi-canh-b 0` |

**Nghẽn đang nhắm:** định vị nội-video. Trần oracle **+139,7%** trên bộ mới sạch
(câu hai cảnh +337,5%), đã **phân rã thành ba tầng rời nhau**
(`scripts/phan_ra_tran.py`, 01/09):

| tầng | phần của trần | ai đang nhắm |
|---|---|---|
| 1. SINH ứng viên (keyframe đáp án có lọt vào 400 không) | **17%** | cảnh B đã ship (+23,3% nhóm qua cổng) |
| 2. XẾP HẠNG nội-video (đã lọt thì có được hạng-1 không) | **40%** | hai cảnh: hoán vị cảnh B đã ship; MỘT cảnh: mọi tín hiệu 0 đồng đã chết → pre-test PE-Core (dở) + ②③ của lane paper |
| 3. PHÂN BỔ dòng (đã hạng-1 thì 100 dòng có dồn quanh nó không) | **43%** | **thuần phân bổ — không cần model mới** |

**Hình dạng chỗ ăn ở nhóm MỘT cảnh** (`dem_bao_hoa_noi_video.py`, 03/09): đáp
án đã đứng trung vị **hạng 2** nội-video (≤3 ở 68% câu) — bài toán là **trận
tay đôi hạng 1↔2**, và mọi phép-đếm-trước từ nay đo theo thước ấy (ngưỡng
thắng ≥62% số trận). Nhóm HAI cảnh trượt hệ thống (lệch 752 frame — text khớp
cảnh A): encoder mới không tự sửa nhóm này.

**PE-Core (encoder thứ hai): NO_GO — cửa đóng kèm số, GPU vẫn KHOÁ**
(`PRETEST_ENCODER.md` §5, 03/09 trưa): trung vị HAI cảnh 11,5 (cần ≤6,8);
hạng-1 MỘT cảnh 45,0% (cần ≥47,0%), KTC chứa 0. Text tower 32 token BPE Anh
cắt 40/40 câu vi — encoder ứng viên tương lai cần context ≥64 token rồi đo
lại đúng cổng này. Quyết định + lý do: `docs/QUYET_DINH_ENCODER_TRAKE.md`.

**TRAKE: đã có bộ đo n=24** (`data/gt_trake.json`, `docs/GT_TRAKE.md`): nền
0,2275, ORACLE-MỐC +107,8%, ORACLE-VIDEO +148,1%, video dòng 1 đúng 21/24;
phân rã: định vị sự kiện 72,8% / sai video 27,2%. Sản xuất giữ
`align_mode=ordered`, `min_gap=2`. Hướng còn mở: lưới bù trừ phi-đều theo bất
định từng sự kiện (chưa ai đo); mục 4 sự kiện chưa có.

**Cửa đã đóng trên bộ đo MỚI (đáng tin):**

- cặp thời gian (+6,7%, KTC chứa 0, hiệu ứng co khi thêm dữ liệu);
- 12 cửa phân bổ/quy tắc dòng (`KE_HOACH_DINH_VI.md` §2.1);
- **toàn trục khử-hub NNN/QB-Norm** (paper ①): V1 cổng đếm 51%; V2 qua cổng
  57% sát nút rồi đo đầy đủ **TEST −1,7%, P(≤0)=77,7%** (03/09);
- **TRỤC NỘI-VIDEO MỘT CẢNH QUÉT SẠCH 03/09 — 4 cơ chế chết trên cùng thước
  trận-tay-đôi**: ① NNN (trên); ③ PRF Rocchio 31%; ② GQE paraphrase 12%
  (paraphrase soi tay đạt — lỗi ở cơ chế, không phải ở sinh); cut-score
  tương đối 44%/56%. Kết luận §10d PAPER_XEP_HANG_NOI_VIDEO.md: training-free
  trên cặp sim SigLIP nội-video ĐÃ CẠN; đầu tư tiếp = ngoài-SigLIP
  (VLM/OCR/lời thoại) hoặc encoder context ≥64 token;
- **TRAKE, có TEST đứng sau:** soft_order (TEST +0,0000, 0/12 mục đổi điểm);
  đóng lần hai trên n=24: unordered (−24,7%), min_gap 0–1 (−9,7%), hedge video
  (hạng-1 = top-3 = 21/24 ⇒ bất khả thi cấu trúc). Lưu ý:
  `data/gt_trake_test_moi.json` đã cháy vai trò TEST cho họ giả thuyết căn
  chỉnh — giả thuyết mới cần TEST mới.

**Cửa đóng trên bộ đo CŨ (đáng nghi ngờ, nên mở lại):** làm mượt thời gian,
chuẩn hoá theo video, VLM xếp lại nội-video, chia lại ngân sách rộng/sâu,
pha CLIP-B32. Tất cả đều đóng vì bộ cũ *không còn headroom nội-video* — điều
kiện đó đã thay đổi. (Riêng làm mượt và chuẩn hoá video đã ÂM lại cả trên bộ
mới — xem §2.1 kế hoạch định vị.)

**Việc xếp hàng kế tiếp** (cập nhật 03/09 trưa — mục 1-2 của quyết định D đã
XONG, đều âm kèm số): 1) hoàn tất đợt sinh GT shard h (L28,L29) + i (L30,L25),
16+16 mục, $0,021 — đang kiểm nhãn hai cảnh độc lập + kiểm neo một-ảnh; sau đó
gộp (`gop_bo_do_moi.py --shard a..i`) và dựng cache ứng viên; 2) với n lớn
hơn: đo lại các "hoà" ở §2.1 kế hoạch định vị (hoà = chưa chứng minh được);
3) tín hiệu ngoài-SigLIP theo giá trị: VLM mở rộng tầm nhìn (§4.1 KIEN_TRUC),
OCR toàn kho (§4.2), lời thoại nguồn ứng viên (§4.3). Bộ test: 223/223 xanh
(03/09 trưa). Ghi chú cũ: nửa TEST bộ 132
đã bị đọc ≥5 lần.

**Việc treo cần người:** cookie YouTube cho 490 video còn thiếu lời thoại;
chỉ mục batch 2 khi BTC phát dữ liệu.
