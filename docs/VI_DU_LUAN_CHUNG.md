# Cách chốt một câu: sáu ca thật, đã che đáp án

Tài liệu này không dạy chạy lệnh — phần đó ở [QUY_TRINH_NOP.md](QUY_TRINH_NOP.md).
Nó dạy thứ khó hơn: **khi ba kênh nói ba điều khác nhau thì tin ai**.

Sáu ca dưới đây là thật, lấy từ vòng sơ tuyển 1. Mã video và đáp án đã đổi thành
`Lxx_Vnnn` / `<đáp án>` vì repo này công khai. Cái cần học là *hình dạng của lập
luận*, không phải con số.

> Muốn xem bản đầy đủ có mã video thật? Nó nằm ở `round1/picks_verified.txt`
> trên máy, và `.gitignore` chặn không cho lên GitHub. Đừng dán nó vào chat nhóm mở.

---

## Ba kênh, và điểm mù của từng kênh

| kênh | trả lời được | mù chỗ nào |
|---|---|---|
| **Hình ảnh** — SigLIP-2 trên 177k keyframe | "khung nào *trông giống* câu mô tả" | màu sắc, số lượng, hành động, chữ, thứ tự — tất cả bị nén vào một vector 1152 chiều |
| **Lời thoại** — BM25 trên 873 transcript | "phóng sự này *nói về* cái gì" | video không có lời thoại; và từ đồng âm khác nghĩa |
| **Mắt VLM** — Gemini xem từng khung | gần như mọi thứ, nếu hỏi đúng cách | đắt, có hạn ngạch, và **đếm rất tệ** |

Nguyên tắc: **một kênh nói không đủ; hai kênh đồng thuận thì tin; hai kênh chọi
nhau thì mở khung hình ra xem.**

---

## Ca 1 — tiêu đề video thắng cả hai kênh còn lại

Đề: người đầu bếp cho đậu Hà Lan vào chảo mực đang xào.

- Engine xếp `Lxx_V035` hạng 1.
- VLM chấm `Lxx_V035` **1.00** và `Lxx_V177` cũng **1.00**. Hoà.
- Mở tiêu đề ra: `Lxx_V035` = *"MỰC XÀO ĐẬU HÀ LAN"*, `Lxx_V177` = *"MỰC XÀO XỐT
  TIÊU XANH"*.

**Chốt `Lxx_V035`.** VLM chấm hoà vì nhìn ảnh nào cũng thấy "mực xào có rau xanh".
Nhưng món ăn thì tiêu đề nói thẳng ra. Bài học: **khi VLM hoà, tìm một kênh có
tín hiệu rời rạc** (tiêu đề, chữ trên hình, tên file) thay vì hỏi lại VLM lần nữa.

## Ca 2 — lời thoại cứu một câu mà engine đánh rơi hoàn toàn

Đề: bún gà, có cà rốt, sả, nấm mèo, kết bằng cọng ngò thả lên trên.

- Engine: video đúng **không nằm trong 100 dòng nộp**. Câu này đang chắc chắn 0 điểm.
- BM25 trên lời thoại: `Lxx_V389` @4:24 — *"món bún gà xào sả, thịt gà chín mềm
  ngọt, nấm mèo giòn giòn lẫn cùng rau củ"*. Đủ bốn nguyên liệu trong đề.
- VLM đối chứng: `Lxx_V389` **1.00** so với ứng viên số 1 của engine **0.35**.

**Chốt `Lxx_V389`.** Bài học: **khi đề liệt kê nhiều danh từ cụ thể, hãy chạy
`search_transcripts.py` trước khi tin engine.** Danh sách nguyên liệu, tên riêng,
số liệu — đó là chỗ BM25 mạnh và SigLIP yếu.

## Ca 3 — cùng một cô giáo trong ba video, chỉ nội dung bài giảng tách được

Đề: phụ nữ mặc áo dài hồng, đeo kính, giảng về các cách dùng động từ `remember`.

- VLM chấm **1.00 cho cả ba** video `Lxx_V041`, `Lxx_V050`, `Lxx_V015`, mỗi video
  ~20 khung. Đúng người, đúng áo, đúng kính — cả ba.
- Tiêu đề: V041 = *"Chuyên đề 6 — Động từ theo sau bởi..."*, V050 = *"Chuyên đề 5
  — Đảo ngữ"*.
- Lời thoại V041 @10:33: *"động từ tiếp theo là remember..."*, và @18:51 nhắc lại.

**Chốt `Lxx_V041`, và lấy frame từ mốc thời gian lời thoại** — 10:33 với fps 25
là khung ~15828. Bài học: **transcript không chỉ chọn video, nó chọn luôn frame.**
`best_segment()` trả về mốc của *câu nói khớp*, không phải đầu cửa sổ tìm kiếm —
lệch 15 giây là đủ trượt cửa sổ chấm điểm.

## Ca 4 — chấm điểm hoà vì cả video đều giống đề

Đề: con lân đứng xoay trên đỉnh cột, rồi nhảy sang cột bên, chúi đầu ngoạm quả bí.

Hỏi VLM bằng **nguyên văn đề** trên 193 keyframe: **72 khung đạt ≥0.60**. Vô dụng —
cả video đều là lân trên cột nên khung nào cũng "giống câu mô tả".

Hỏi lại bằng **câu hỏi phân biệt**, chỉ nhắm chi tiết thoáng qua:

> *"MIỆNG con lân có đang NGẬM/CẮN vào quả bí đỏ kèm bông hoa màu vàng không?
> Chấm 100 CHỈ KHI thấy rõ quả bí trong hoặc sát miệng lân. Nếu chỉ thấy lân trên
> cột mà không thấy quả bí thì chấm 0."*

Kết quả: `Lxx_V035` có **24/193** khung đạt, đối thủ chỉ **2/204**. Dứt khoát.

**Đây là kỹ thuật quan trọng nhất trong tài liệu này.** Đề bài luôn tả *cả bối
cảnh lẫn khoảnh khắc*; nếu đưa nguyên văn cho VLM thì nó chấm theo bối cảnh. Muốn
tìm khoảnh khắc, phải tự tay viết một câu hỏi:

1. chỉ hỏi về **chi tiết chỉ đúng ở đúng giây đó** (miệng đang ngậm, dùi đang chạm);
2. dạng **có/không**, ép model phải quyết, thay vì chấm độ giống;
3. **nói rõ khi nào chấm 0** — "chỉ thấy lân trên cột thì chấm 0".

Các câu hỏi phân biệt của vòng 1 nằm ở `round1/sharp_questions.json` (đã gitignore).
Xem cách viết ở docstring đầu `scripts/verify_hypotheses.py`.

## Ca 5 — chữ trên hình là kênh duy nhất trả lời được

Đề (Q&A): sạt lở làm tắc nghẽn một đoạn đường đèo. **Tên của con đèo là gì?**

Không transcript nào nêu tên đèo. Cả engine lẫn VLM đều chỉ nói được "đây là cảnh
sạt lở". Cách giải: quét **chữ chạy dưới màn hình** trong đoạn tin đó.

```bash
python scripts/verify_hypotheses.py --pairs "query-p1-17-qa=Lxx_V020" \
  --range "5800-9200" --model gemini-3.5-flash-lite \
  --question "Đọc CHỮ trên khung hình này. Có dòng nào ghi TÊN MỘT CON ĐÈO không?
              Trong phần lý do, GHI NGUYÊN VĂN dòng chữ bạn đọc được."
```

Lần một đọc ra *"QUỐC LỘ 4H SẠT LỞ LỚN..."* — là quốc lộ, không phải đèo, nên
**loại video đó**. Sau vài lần lần theo cách này mới tìm ra đúng bản tin có tên đèo.

Bài học kép: **(a)** yêu cầu model *chép nguyên văn* chữ nó đọc được, đừng để nó
diễn giải; **(b)** một câu trả lời "sai loại" (quốc lộ thay vì đèo) là bằng chứng
đủ mạnh để **loại** một ứng viên, chứ không phải để chấp nhận nó.

## Ca 6 — chỗ tôi làm sai, và cái đã sửa được nó

Vẫn câu trên. Tôi chốt một cái tên đèo, nộp file, rồi mới đối chiếu với bản của
một thành viên khác trong đội. Bạn ấy trả lời khác. Kiểm tra lại bằng transcript:

> *"tại Điện Biên, sáng nay một đoạn đường qua **đèo `<tên>`** ... đã xảy ra vụ sạt
> lở, đất đá tràn ra mặt đường ước tính gần 1000 m³ khiến hàng trăm phương tiện
> ách tắc"*

Cụm tên đèo đó xuất hiện **đúng một lần trong cả 873 transcript**, ở chính video
bạn ấy chọn. Tôi sai; đáp án của bạn ấy đúng. Sửa lại trước khi nộp bản cuối.

Bài học: **luôn đối chiếu chéo với bản của người khác trước khi nộp.** Lần đối
chiếu đó sửa được 2 câu — và cũng phát hiện bản của bạn ấy ghi câu TRAKE **2 cột
cho đề 3 sự kiện** (sai định dạng, mất trắng câu đó), nên cả hai bên đều có lợi.

---

## Sổ tay rút gọn

| tình huống | làm gì |
|---|---|
| đề liệt kê nhiều danh từ cụ thể | `search_transcripts.py` trước |
| đề hỏi tên riêng / con số | đọc **chữ trên hình**, bắt model chép nguyên văn |
| đề hỏi đếm | đọc ở **độ phân giải gốc** (`read_answer.py`), và đừng tin số đầu tiên |
| VLM chấm hoà nhiều video | tìm kênh rời rạc: tiêu đề, chữ, tên file |
| VLM chấm cao cho nửa số khung | câu hỏi sai — viết lại thành **câu hỏi phân biệt** |
| hành động lặp lại nhiều lần trong video | nộp **chuỗi frame** `F1\|F2\|F3`, R@k lấy max nên không mất gì |
| hai kênh chọi nhau | `verify_hypotheses.py` — mở khung hình ra xem, đừng đoán |
| sắp nộp | đối chiếu chéo với bản của thành viên khác |

## Ba cái bẫy đã làm mất điểm thật

1. **Đo độ chính xác cấp video.** Đó là lý do đội dừng ở 5.8. BTC chấm video **và**
   frame phải nằm trong cửa sổ. Nhiều thay đổi làm tăng R@1 cấp video mà **giảm**
   điểm thi — đã gặp ít nhất ba lần.
2. **Tin ground truth có sẵn.** 93% frame đáp án trong bộ GT trùng keyframe. Đo
   thẳng trên đó thì "chỉ lấy keyframe" thắng 0.562/0.526; đo trên bộ **không bám
   keyframe** thì đảo ngược, 0.257/0.333. Mọi script `experiment_*` đều bốc lại
   khoảnh khắc thật ở một điểm ngẫu nhiên trong khe giữa hai keyframe.
3. **Chèn dòng vào giữa.** R@k lấy max trên k dòng đầu, nên **thêm** dòng ở cuối
   không bao giờ giảm điểm — nhưng **chèn** vào giữa thì có. Đã một lần làm xáo
   trộn 4 câu vốn đang đúng; nay các ứng viên tìm được từ lời thoại chỉ được
   *nối vào cuối*.
