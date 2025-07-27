# 🇯🇵 Trợ lý ảo Tiếng Nhật – Agentic RAG

Một hệ thống Trợ lý ảo AI đa năng được thiết kế để hỗ trợ người học tiếng Nhật.  
Dự án sử dụng kiến trúc **Agentic RAG (Retrieval-Augmented Generation)**, kết hợp mô hình ngôn ngữ lớn (LLM) với khả năng truy xuất kiến thức từ cơ sở dữ liệu chuyên biệt để cung cấp câu trả lời chính xác, ngữ cảnh phù hợp và cá nhân hóa.

---

## ✨ Tính năng chính

Hệ thống bao gồm nhiều **Agent chuyên biệt**, mỗi agent phục vụ một mục tiêu riêng trong hành trình học tập:

- 🎓 **Gia sư QnA (QnAAgent)**  
  Trả lời kiến thức chung, dịch thuật, tạo bài kiểm tra (quiz), sửa lỗi ngữ pháp – tất cả dựa trên kho tài liệu học tập cá nhân hóa.

- 🎯 **Cố vấn Lộ trình học (PlannerAgent)**  
  Tương tác để xây dựng, cập nhật, quản lý lộ trình học tập (CRUD), dựa trên trình độ, mục tiêu và thời gian người học.

- 📚 **Trợ lý Học tập theo Ngữ cảnh (LearningAgent)**  
  Chỉ trả lời trong phạm vi **1 bài học cụ thể** hoặc **1 unit học**, giúp người dùng tập trung sâu.

- 📝 **Gia sư Chữa bài (ReviewerAgent)**  
  Phân tích, đánh giá bài kiểm tra, chỉ ra lỗi sai và cách cải thiện chi tiết theo từng kỹ năng.

- 🧠 **Cơ sở tri thức RAG**  
  Xử lý các file PDF (giáo trình, từ vựng...), chia nhỏ thành "chunk", embedding và lưu vào **PostgreSQL + pgvector** để truy xuất theo ngữ nghĩa.

---

## 🏗️ Kiến trúc hệ thống

Hệ thống sử dụng kiến trúc **microservices** và đóng gói bằng **Docker**, bao gồm:

- ⚙️ **API Gateway (FastAPI)**  
  Cung cấp các RESTful API endpoints, định tuyến đến từng Agent, xử lý xác thực và điều phối logic.

- 🔧 **Core Services**
  - `LLM Core`: Tích hợp GPT-4 (hoặc tương đương) qua LangChain
  - `Vector Store`: PostgreSQL + pgvector để lưu & tìm kiếm embedding
  - `Session Manager`: Quản lý lịch sử hội thoại và ngữ cảnh từng user
  - `Data Processing`: Script xử lý PDF, chunking và embedding

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Backend | Python, FastAPI |
| AI/ML | LangChain, OpenAI, Sentence Transformers |
| Database | PostgreSQL, pgvector |
| Triển khai | Docker, Docker Compose, Uvicorn |

---

## 🚀 Bắt đầu

### 🔧 Yêu cầu

- Cài đặt **Docker** và **Docker Compose**
- Tạo tệp `.env` ở thư mục gốc

### 📥 Clone repository

```bash
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag
