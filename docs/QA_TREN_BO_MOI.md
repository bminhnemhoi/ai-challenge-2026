# Đường Q&A đo lại trên bộ đo khớp phân bố — hiệu chỉnh con số đã công bố

Chốt 01/09/2026. Lệnh tái lập:

```
python -u scripts/experiment_qa_answer.py --gt data/ground_truth_moi.json \
    --chi-sach --trong-tai --bien-the goc,net_loi_doan_lan_nhieu \
    --cache data/cache_qa_moi
```

## 0. Vì sao phải đo lại

Đường Q&A đa kênh được ship với con số **70,0% → 93,3% trên TEST (+23,3%)**.
Con số đó đo trên bộ 60 câu ground truth **cũ** — thứ ta nay biết là lệch phân bố
(`docs/BO_DO_KHOP_PHAN_BO.md`). Theo mục 2 của `docs/QUY_TRINH_TU_DONG.md`
("xác minh thứ vừa ship"), mọi thay đổi đã ship phải được đo lại trên công cụ đo
tốt hơn khi có nó.

## 1. Kết quả trên bộ mới (132 mục sạch, trọng tài LLM)

| biến thể | TUNE | TEST | cả 132 | số câu bỏ trống |
|---|---|---|---|---|
| `goc` — đường sản xuất CŨ | 75,8% | 78,8% | 77,3% | **11** |
| `net_loi_doan_lan_nhieu` — đường ĐANG SHIP | 84,8% | 83,3% | **84,1%** | **0** |

Chênh trên TEST: **+4,5 điểm phần trăm**. Sai số nhị thức 1 sd trên 66 câu ≈ 5,0%
⇒ **chưa vượt 2 sd, tính là hoà theo luật của dự án.** Trên cả bộ: +6,8 điểm
(77,3% → 84,1%, tức +8,8% tương đối).

## 2. Hiệu chỉnh phải nói thẳng

**Con số +23,3% là con số của bộ đo cũ, và nó thổi phồng.** Trên bộ đo khớp phân
bố đề thật, cùng một thay đổi chỉ đáng **+8,8% tương đối**, và phần trên nửa TEST
không vượt được ngưỡng 2 sd.

Vì sao chênh lệch lớn thế: nền trên bộ mới **cao hơn** (77,3% so với 70,0%). Câu
Q&A của bộ mới do máy sinh từ đoạn video nên nhiều câu hỏi màu sắc, tên hiện trên
màn hình — dễ hơn câu Q&A của bộ cũ. Nền cao thì phần cải thiện còn lại nhỏ.

## 3. Nhưng KHÔNG rút lại thay đổi — và lý do là một phép đếm

Thành phần chắc chắn nhất của cải tiến này không phải điểm số mà là **phép đếm
tất định**:

| | số câu model trả về CHUỖI RỖNG |
|---|---|
| đường cũ | **11 / 132** |
| đường đang ship | **0 / 132** |

Mười một câu đó là **0 điểm bảo đảm** theo luật 2.1.2, bất kể khung hình đúng hay
sai. Đáp án đoán sai cũng 0 điểm, nên đoán là trội tuyệt đối — không có kịch bản
nào bỏ trống tốt hơn. Con số này lặp lại trên cả hai bộ đo (bộ cũ: 11/60 → 0/60)
và không phụ thuộc vào bộ đo nào đúng hơn.

Kết luận vận hành: **giữ nguyên đường đang ship**, nhưng mọi tài liệu và kế hoạch
điểm phải dùng con số **+8,8% tương đối**, không dùng +23,3%.

## 4. Nơi còn điểm để lấy

Các câu đường ship vẫn sai chia làm hai loại, và chỉ một loại là lỗi thật:

- **Lỗi thật:** đọc sai màu ("áo trắng" bị đọc thành "xanh dương nhạt"), đọc sai
  chi tiết nhỏ. Đây là giới hạn của model đọc ảnh, cần khung hình rõ hơn hoặc
  model mạnh hơn — mà gpt-5.2 đã đo là **thua** Gemini free ở bước này.
- **Lỗi của bộ đo:** đáp án chuẩn và đáp án máy chỉ khác dấu/hoa-thường của một
  tên riêng. Trọng tài LLM bắt được phần lớn nhưng không phải tất cả.

Việc đáng làm tiếp: soi bằng mắt ~10 câu sai còn lại để tách hai loại này, rồi
mới quyết có đáng đầu tư thêm vào bước đọc hay không. Nếu quá nửa là lỗi bộ đo
thì đường Q&A đã gần trần và nên chuyển toàn bộ công sức sang định vị.
