# Mô hình bốc khoảnh khắc — có đóng nhầm cửa nào không?

Chốt 01/09/2026, theo `docs/KE_HOACH_DINH_VI.md` §4.2(a). Script:
`scripts/do_lai_cua_da_dong.py` (0 API, ~9 phút CPU).

## 0. Vì sao phải kiểm

Bộ chấm bốc khoảnh khắc thật **đều** trên ô keyframe chứa khung neo. Nhưng với
câu HAI cảnh, khung neo *được định nghĩa* là khung **đầu tiên của cảnh B** — một
cú cắt. Khoảnh khắc thật **không thể** nằm trước cú cắt ấy, trong khi mô hình
ĐỀU đặt gần **một nửa** khối lượng xác suất vào cảnh A.

Phản biện đã chỉ ra điều này **quyết định dấu** của trục sigma (bốc đều → sigma
60 thắng; bốc Gauss quanh neo → sigma 15 thắng). Nếu một chi tiết của thiết bị đo
quyết định được dấu của kết luận, thì phải hỏi: **có cửa nào bị đóng chỉ vì thiết
bị đo sai giả định không?**

## 1. Kết quả — nhóm HAI cảnh, n = 66, hạt độc lập 771000

| cấu hình | bốc ĐỀU | so nền | bốc **SAU_NEO** | so nền |
|---|---|---|---|---|
| nền (sản xuất hôm nay) | 0,1049 | — | 0,1084 | — |
| + ứng viên cảnh B *(đã ship)* | 0,1400 | **+33,4%** | 0,1445 | **+33,3%** |
| + hoán vị nội-video *(đã ship)* | 0,2205 | **+110,1%** | 0,2362 | **+117,9%** |
| sigma 45 *(cửa đã đóng)* | 0,1261 | +20,2% | 0,1144 | **+5,5%** |
| sigma 15 *(hướng ngược lại)* | 0,0868 | −17,3% | 0,1069 | −1,4% |

## 2. Ba kết luận

**(a) Không cửa nào bị đóng nhầm.** Không cấu hình nào đổi dấu giữa hai mô hình.

**(b) Hai lever đã ship BỀN với giả định của thiết bị đo** — và lever hoán vị còn
mạnh **hơn** dưới mô hình đúng hơn (+110,1% → +117,9%). Cộng dồn, hai lever đưa
nhóm câu hai cảnh từ 0,1049 lên 0,2205 — **hơn gấp đôi**. Đây là bằng chứng độc
lập với phép đo TUNE/TEST đã dùng khi ship, trên hạt giống chưa từng dùng.

**(c) Trục sigma teo 4 lần dưới mô hình đúng** (+20,2% → +5,5%). Tức phần lớn
"lợi ích" của sigma lớn là **ảo ảnh của giả định bốc sai**: sigma lớn rải khối
lượng rộng ra hai phía, và mô hình ĐỀU thưởng cho việc rải sang cảnh A — vùng mà
khoảnh khắc thật **không bao giờ** rơi vào. Cửa này đã đóng bằng TEST (−3,8% trên
nhóm hai cảnh), và mô hình bốc đúng khiến nó **đáng đóng hơn**, không phải ngược lại.

## 3. Giới hạn của chính phép đo này

Đây là **kiểm định độ bền**, không phải cổng mới: chấm trên **cả 66 câu**, không
chia TUNE/TEST, nên không được dùng để *chốt* điều gì. Nó chỉ trả lời đúng một câu
hỏi — "kết luận có đảo khi đổi giả định bốc không?" — và câu trả lời là không.

Con số dùng để quyết định vẫn là con số TUNE/TEST đã ghi trong
`docs/KE_HOACH_DINH_VI.md` và `docs/UNG_VIEN_CANH_B.md`.

## 4. Luật từ nay

Mọi phép đo trên nhóm câu hai cảnh phải **báo cáo song song hai mô hình bốc**.
Chênh lệch giữa chúng là thước đo trực tiếp cho việc kết luận phụ thuộc bao nhiêu
vào giả định của thiết bị đo — và như trục sigma cho thấy, nó có thể là 4 lần.
