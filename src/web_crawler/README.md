# dethitiengnhat.com Crawler

Crawler này sẽ lấy 35 câu hỏi đầu tiên từ một đề thi, phân loại scope (kanji/vocab), lưu vào bảng question và option trong database.

## Sử dụng

1. Cài đặt thư viện:
   ```
   pip install requests beautifulsoup4 psycopg2
   ```
2. Sửa thông tin kết nối DB trong file dethitiengnhat_crawler.py hoặc đặt biến môi trường tương ứng.
3. Chạy script:
   ```
   python src/web_crawler/dethitiengnhat_crawler.py
   ```

## Logic
- 20 câu đầu: scope = kanji
- 15 câu tiếp: scope = vocab
- type = multiple choice cho tất cả
- Bảng option sẽ lưu 4 đáp án, chỉ 1 đáp án đúng (is_correct)