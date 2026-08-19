# Quy tắc định dạng toán học và văn bản (Math & Text Formatting)

## Tránh Sử Dụng Inline LaTeX ($...$)
- Giao diện chat của Antigravity IDE chưa hỗ trợ phân giải cú pháp inline LaTeX toán học bằng dấu dollar đơn (`$...$`), dẫn đến việc các công thức như `$\text{Top}_{100}$` bị vỡ định dạng và hiển thị thô.
- **Quy định bắt buộc:**
  1. KHÔNG sử dụng ký hiệu `$...$` cho công thức toán hoặc chỉ số dưới trong câu trả lời.
  2. Sử dụng văn bản thuần túy, Unicode toán học (`∑`, `∈`, `≤`, `≥`, `Top₁`, `Top₅`, `Top₂₀`, `Top₅₀`, `Top₁₀₀`) hoặc viết rõ ràng như `Top 1`, `Top 5`, `Top 20`, `Top 50`, `Top 100`.
  3. Đối với các công thức toán học nhiều dòng, hãy đặt trong khối code block ````text ... ```` hoặc ````math ... ```` để đảm bảo hiển thị đẹp mắt, rõ ràng và chuẩn xác 100%.
