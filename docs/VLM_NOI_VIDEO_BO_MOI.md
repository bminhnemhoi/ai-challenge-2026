# VLM xếp lại nội-video, đo lại trên bộ khớp phân bố — và cái cửa mở ra thứ khác

Chốt 01/09/2026. Script: `scripts/do_vlm_noi_video_moi.py`.
Bộ đo: `data/ground_truth_moi.json`, bộ **sạch 132 mục** (bỏ shard `c` lẫn trục),
**66 câu hai cảnh**. Nền = đường sản xuất hiện tại, allocator `coverage`,
đã bật lever cảnh B (`--canh-b 100`).

> Không mã video, không đáp án trong tài liệu này.

**Kết luận một dòng:** VLM xếp lại nội-video **KHÔNG đáng ship** — nó bị một tín
hiệu **0 đồng** đánh bại trên cùng một cơ chế, và trộn nó vào làm *tệ đi*. Nhưng
phép đo để đi tới đó phát hiện hai thứ đắt hơn câu trả lời: phép hoán vị mà kết
luận cũ dựa vào **là phép đồng nhất trên đường sản xuất**, và cơ chế đúng thay
cho nó mở ra một lever lớn cho nhóm câu hai cảnh.

---

## 0. Vì sao mở lại cửa này

Cửa bị đóng bằng một câu rất mạnh trong `KIEN_TRUC_VA_HUONG_CAI_THIEN.md`:

> trong các slot đã truy xuất, SigLIP thuần đã đặt keyframe gần đáp án ở hạng
> nội-video trung vị **1,0** (hạng-1: 60%) — không còn chỗ cho bộ xếp lại nào

Đó là **đặc điểm của bộ 60 câu cũ**, nơi câu hỏi được viết bằng cách nhìn đúng
cái keyframe mà bộ truy xuất đã đánh chỉ mục. Trên bộ mới, hạng nội-video trung
vị là **2** (một cảnh) và **8** (hai cảnh). Điều kiện đã đổi.

---

## 1. Phát hiện thứ nhất: phép đo cũ giờ là PHÉP ĐỒNG NHẤT

Bản "giữ-slot" sạch (`experiment_sharp_rerank.xep_lai_giu_slot`) hoán vị các
**đối tượng ứng viên** giữa các vị trí trong danh sách, rồi đưa vào
`allocate_hybrid_rows`. Nhưng bộ phân bổ sản xuất bây giờ là `coverage`, và tiên
nghiệm của nó là

```
mass(v, x) = Σ_i  w_i · exp(−½((x − f_i)/σ)²),      w = softmax(điểm / 0,02)
```

— một **tổng trên TẬP** ứng viên. Đổi thứ tự các phần tử không đổi một bit nào
của tổng đó.

Đo bằng tín hiệu ngẫu nhiên, 66 câu hai cảnh:

| bộ phân bổ | số câu đổi dòng |
|---|---|
| `coverage` (sản xuất hiện tại) | **0/66** |
| `hybrid` (bộ mà phép đo cũ dùng) | 66/66 |

Hệ quả phải nói thẳng: **chạy lại thí nghiệm cũ trên đường sản xuất hôm nay sẽ
in ra đúng 0,0% ở mọi cấu hình, mọi trọng số, mọi cách hỏi** — và con số 0,0% ấy
trông y hệt một kết quả "hoà" bình thường. Kết luận "không còn chỗ cho bộ xếp
lại nào" không thể kiểm lại bằng con đường đã sinh ra nó.

## 2. Cơ chế đúng cho `coverage`: hoán vị ĐIỂM trong cùng video

Trong mỗi video, giữ nguyên **đa tập điểm** của video đó, chỉ gán lại điểm nào
thuộc khung nào. Khi ấy tổng khối lượng softmax của mỗi video **không đổi**, nên
bề rộng phủ video không thể bị phá — đúng thứ đã gây artifact −35% của phép
gom-khối năm ngoái. Chỉ *hình dạng khối lượng bên trong video* dịch đi.

Bất biến, kiểm bằng `assert` chứ không bằng mắt:

- `w = 0` (khoá xếp chính là điểm) ⇒ 100 dòng **giống hệt** nền, **132/132 mục**;
- câu **không** qua cổng hai cảnh ⇒ dòng giống hệt nền ở **mọi** cấu hình.

> Một cái bẫy đã bị `assert` bắt và đáng ghi lại: pool sản xuất có **khung
> trùng** — cùng một keyframe xuất hiện hai lần với hai điểm khác nhau (22/132
> câu, tới 10 cặp một câu). Đánh khoá theo cặp `(video, khung)` thì hai bản sao
> chung một khoá, phép hoán vị đổi điểm giữa chúng, và bất biến `w=0` vỡ. Khoá
> phải theo **chỉ số ứng viên**.

## 3. Trần — đo trước khi tiêu một đồng quota nào

Oracle của **đúng cơ chế này** (dồn điểm lớn nhất của video đúng về khung gần
khoảnh khắc thật), hạt giống 77000:

| nhóm | n | nền | oracle | trần |
|---|---|---|---|---|
| tất cả | 132 | 0,2078 | 0,3637 | +75,0% |
| MỘT cảnh | 66 | 0,2760 | 0,3754 | +36,0% |
| **HAI cảnh** | 66 | **0,1396** | **0,3520** | **+152,2%** |

Và trần **khả thi** theo ngân sách chấm khung — bộ xếp lại chỉ hoán vị được
những khung nó chấm, nên đây mới là con số quyết định:

| top-V video × F khung | ảnh | lô gọi | điểm | trần | với tới |
|---|---|---|---|---|---|
| 2×10 | 1.154 | 145 | 0,2311 | +65,6% | 43% |
| **3×12 (đã chạy)** | 1.915 | 240 | 0,2527 | **+81,1%** | 53% |
| 5×12 | 3.064 | 383 | 0,2725 | +95,2% | 63% |
| 5×20 | 4.054 | 507 | 0,2976 | +113,2% | 74% |

## 4. Câu hỏi định vị: chia việc đúng chỗ

Bài học đã ghi: VLM trả lời *"khung này có khớp mô tả không"*, còn thứ quyết định
điểm là *"khung này có gần khoảnh khắc đúng nhất không"*. Nhưng bắt VLM tự trả
lời "đây có phải khung ĐẦU TIÊN của cảnh B" từ **một** ảnh là bắt nó làm việc bất
khả; còn đưa **dãy ảnh đánh số** thì rơi đúng vào lỗi đã phá bộ sinh ground truth
(model tả đúng nội dung nhưng **đánh sai số thứ tự ảnh**).

Nên chia đôi: **VLM chấm phân loại từng ảnh** ("ảnh này có phải cảnh B không" —
thứ nó làm tốt), **phép định vị do ta suy ra** từ điểm phân loại cộng trục thời
gian đã biết chắc từ `frame_idx`:

```
loc(f) = B(f) · (1 − α · B(khung được chấm liền trước trong cùng video))
```

α = 0 là phân loại thuần; α = 1 là "khung đầu tiên của cảnh B". Không còn chỗ nào
cho model đánh nhầm số thứ tự.

**Mục tiêu này được kiểm riêng, 0 đồng.** Với 61 câu hai cảnh tách được ranh giới
A→B (chỗ sim(cảnh B) vượt sim(cảnh A)):

| khoảng cách neo ↔ khung cảnh-B ĐẦU TIÊN | tỷ lệ |
|---|---|
| đúng bằng 0 keyframe | **39/61 = 64%** |
| ≤ 1 keyframe | 43/61 = 70% |
| ≤ 2 keyframe | 47/61 = 77% |

Trung vị lệch **0**. Neo cũng nằm bên cảnh B ở **60/66** câu. Vậy "khung đầu tiên
của cảnh B" đúng là mục tiêu cần nhắm — giả thuyết được xác nhận *trước* khi nhìn
điểm.

## 5. Kết quả TUNE — và cái đối chứng 0 đồng lật ngược kết luận

33 câu hai cảnh, nền 0,1648. Chấm 1.915 ảnh, 213 lần gọi Gemini
(gemini-3.5-flash-lite + xoay vòng), ≈ \$0,18 theo giá niêm yết.

| nguồn tín hiệu | tốt nhất | α | w |
|---|---|---|---|
| **VLM, hỏi bằng cảnh B** | **+32,7%** | 0 | ≥1 |
| **SigLIP, chính cảnh B (0 đồng)** | **+40,9%** | 0,5 | ≥1 |
| trộn VLM + SigLIP | +35,3% | 0,5 | 1 |
| *đối chứng* SigLIP cảnh **A** | −11,0% → −35,2% | | |
| *đối chứng* SigLIP cảnh B **đảo dấu** | −32,0% → −36,7% | | |
| *đối chứng* khoá **ngẫu nhiên** | −0,5% → −8,1% | | |

Ba điều đọc được, theo thứ tự quan trọng:

**(1) Đối chứng artifact sạch.** Khoá **ngẫu nhiên** qua đúng cơ chế ấy **không
bao giờ dương** (−0,5% → −8,1%). Cơ chế tự nó không chế ra điểm — đây là phép
kiểm mà bài học −34,7% bắt buộc phải có. Tín hiệu **đảo dấu** và tín hiệu **cảnh
A** đều âm mạnh, đối xứng với mức dương của cảnh B: dấu đi đúng chiều ở cả bốn
nguồn.

**(2) Đường tham số phẳng.** Mọi tín hiệu thật bão hoà từ w = 1 tới w = 100 (điểm
y hệt nhau tới chữ số thứ tư). Không có đỉnh nhọn để trượt xuống — cùng dấu hiệu
"cơ chế thật" mà lever cảnh B đã dùng, ngược hẳn với lever ③ vốn đổi cấu hình
thắng khi thêm dữ liệu.

**(3) VLM THUA tín hiệu 0 đồng, và trộn vào thì tệ hơn.** SigLIP +40,9% >
VLM +32,7%; trộn hai cái được +35,3%, tức **thấp hơn SigLIP một mình**. Phần đóng
góp riêng của VLM không phải nhỏ — nó **âm**.

Ghi chú về α: với SigLIP, phép định vị ăn thật (α=0,5 cho +40,9% so với α=0 cho
+32,7%). Với VLM thì ngược lại, α=0 tốt nhất. Tức bước suy ra "khung đầu tiên"
chỉ khai thác được khi tín hiệu phân loại đủ mịn theo thời gian.

## 6. TEST

**Cấu hình chốt trên TUNE là `siglipB, α=0,5, w=1,0`** — tức thứ được chốt
**không phải VLM**. 33 câu hai cảnh:

| | nền | chốt | chênh |
|---|---|---|---|
| điểm | 0,1146 | 0,2069 | **+80,5%** |

- bootstrap **theo câu**: chênh +0,0923; KTC 95% **[+0,0337, +0,1604]**; P(≤0) = **0,0%**
- phân rã tất định: **10 câu tốt lên, 1 câu xấu đi, 22 câu không đổi**

Dấu thì chắc (10 trên 11 câu có thay đổi đi đúng chiều — phép thử dấu cho
p ≈ 0,6%). **Độ lớn thì không:** ba câu đắt nhất chiếm **52%** toàn bộ mức tăng.
Con số +80,5% là ước lượng của một hiệu ứng tập trung ở vài câu, và với n = 33
nó gần như chắc chắn **thổi phồng**.

Riêng VLM, ở lượt đọc TEST của chính nó (xem §7): 0,1073 → 0,1732 = **+61,3%**,
KTC [+0,0255, +0,1157]. VLM **có** ăn — chỉ là ăn ít hơn thứ miễn phí.

## 7. Kỷ luật đo lường: chỗ đã tiêu mất, nói rõ

**TEST đã bị đọc HAI lần có thông tin, trên HAI phép chia khác nhau.** Trình tự:

1. Lượt 1: chia phân tầng theo (cổng, shard) rồi luân phiên. Chốt VLM
   (α=0, w=1,0) → TEST **+61,3%**.
2. Sau đó mới phát hiện phép chia ấy **hỏng**: nó sắp giảm dần rồi luân phiên
   thẳng, nên câu **cao hơn của mỗi cặp luôn rơi về TUNE** — TUNE 0,178 vs TEST
   0,100. Bản "chia khối cho cân" tự tay tạo ra độ lệch nó định xoá.
3. Sửa thành **rắn bò** (đảo bên ở các cặp lẻ), chạy lại toàn bộ giao thức, chốt
   `siglipB` → TEST **+80,5%**.

Lượt 3 là số chính vì nó là giao thức đúng; nhưng bảo đảm "đọc TEST đúng một
lần" **đã tiêu mất**, và phải đọc con số +80,5% với hiểu biết đó. Mọi lượt khảo
sát nguồn tín hiệu và đối chứng ở §5 chạy với `--khong-doc-test`, sống trọn trong
TUNE.

## 8. Kết luận cho lane này

**Với câu hỏi được giao — "VLM xếp lại nội-video có ăn trên bộ đo mới không" —
câu trả lời là ÂM.** Không phải âm vì cửa đóng, mà vì:

1. VLM ăn **+32,7% TUNE / +61,3% TEST**, nhưng tín hiệu **0 đồng** ăn **+40,9%
   TUNE / +80,5% TEST** trên **đúng cùng một cơ chế, cùng tập khung, cùng công
   thức**;
2. trộn VLM vào SigLIP cho **+35,3%**, tức **thấp hơn** SigLIP một mình — đóng
   góp biên của VLM là **âm**;
3. VLM tốn 213 lần gọi cho 66 câu và một model đã cháy quota ngày trong lúc đo.

Đây là kết luận âm **có số liệu**, không phải "chưa đủ tốt, cần prompt hay hơn".
Cách hỏi đã được sửa đúng như bài học yêu cầu (câu hỏi định vị, chia việc để model
không phải đánh số ảnh), mục tiêu định vị đã được xác nhận riêng bằng cấu trúc
(64% neo trùng khít khung cảnh-B đầu tiên) — và nó vẫn thua.

**Thứ đáng mang đi tiếp không phải VLM mà là cơ chế.** Hoán vị ĐIỂM trong video
theo `sim(cảnh B)` là một lever **0 đồng, 0 lần gọi LLM**, dùng lại đúng vector
`simsB` mà lever cảnh B đã tính sẵn và cache. Nó **chưa được ship** và không thuộc
quyền lane này (lane chỉ được tạo file mới).

### Điều chưa biết — đọc trước khi ai đó ship cái lever ấy

1. **Văn bản cảnh B trên bộ đo là bản phân rã của chính bộ sinh** — sạch gần như
   oracle. Trên đề THẬT nó do cổng `gan_nhan_hai_canh` trích ra, và độ chính xác
   từng câu của cổng ấy **chưa ai đo** (đã ghi trong `UNG_VIEN_CANH_B.md`). Lever
   này để văn bản cảnh B quyết định **chỗ đặt dòng**, không chỉ thêm ứng viên,
   nên nó nhạy với cổng sai hơn hẳn lever cảnh B. Ước lượng +80,5% **bị thổi
   phồng** bởi lượng chưa đo được này.
2. Hiệu ứng **tập trung**: 3/33 câu chiếm 52% mức tăng.
3. Chỉ đo trên nhóm **hai cảnh**; nhóm một cảnh chưa động tới (bất biến bằng
   assert). Trần một cảnh là +36,0% — chưa thử.
4. `w` bão hoà nghĩa là tín hiệu **thay hẳn** thứ tự nội-video của SigLIP trong
   các khung được chấm. Rủi ro tối đa bị chặn ở nhóm qua cổng, nhưng đó là một
   thay thế mạnh chứ không phải một hiệu chỉnh nhẹ.
5. Câu của bộ đo do **máy sinh**; cấu trúc hai cảnh dứt khoát hơn đề người viết.

## 9. Tái lập

```bash
# giai đoạn 1 — cơ chế + trần, KHÔNG gọi API
python -u scripts/do_vlm_noi_video_moi.py --giai-doan co-che

# giai đoạn 2 — bảng TUNE đầy đủ + đối chứng, KHÔNG tiêu lần đọc TEST nào
python -u scripts/do_vlm_noi_video_moi.py --giai-doan vlm --cach-hoi canhB \
    --videos 3 --frames 12 --weights 0,0.02,0.2,1.0,5.0,100.0 --khong-doc-test

# chỉ đối chứng 0 đồng, không cần GEMINI_API_KEY cho phần VLM
python -u scripts/do_vlm_noi_video_moi.py --giai-doan vlm --cach-hoi chi-siglip \
    --videos 3 --frames 12 --khong-doc-test
```

Điểm VLM đã cache trong `data/vlm/` theo (model, câu hỏi, video, khung), nên chạy
lại **không** tốn thêm lần gọi nào.
