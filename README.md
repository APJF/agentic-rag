# 🇯🇵 Trợ Lý ảo Tiếng Nhật – Agentic RAG

Một hệ thống Trợ lý ảo AI đa năng được thiết kế để hỗ trợ người học tiếng Nhật.
Dự án sử dụng kiến trúc **Agentic RAG (Retrieval-Augmented Generation)**, kết hợp mô hình ngôn ngữ lớn (LLM) với khả năng truy xuất kiến thức từ cơ sở dữ liệu chuyên biệt để cung cấp câu trả lời chính xác, ngữ cảnh phù hợp và cá nhân hóa.

---

## ✨ Tính năng chính

Hệ thống bao gồm nhiều **Agent chuyên biệt**, mỗi agent phục vụ một mục tiêu riêng trong hành trình học tập:

* 🎓 **Gia sư QnA (QnAAgent)**
  Trả lời kiến thức chung, dịch thuật, tạo bài kiểm tra (quiz), sửa lỗi ngữ pháp – tất cả dựa trên kho tài liệu học tập cá nhân hóa.

* 🎯 **Cố vấn Lộ trình học (PlannerAgent)**
  Tương tác để xây dựng, cập nhật, quản lý lộ trình học tập (CRUD), dựa trên trình độ, mục tiêu và thời gian người học.

* 📚 **Trợ lý Học tập theo Ngữ cảnh (LearningAgent)**
  Chỉ trả lời trong phạm vi **1 bài học cụ thể** hoặc **1 unit học**, giúp người dùng tập trung sâu.

* 📝 **Gia sư Chữa bài (ReviewerAgent)**
  Phân tích, đánh giá bài kiểm tra, chỉ ra lỗi sai và cách cải thiện chi tiết theo từng kỹ năng.

* 🧠 **Cơ sở tri thức RAG**
  Xử lý các file PDF (giáo trình, từ vựng...), chia nhỏ thành "chunk", embedding và lưu vào **PostgreSQL + pgvector** để truy xuất theo ngữ nghĩa.

---

## 🏗️ Kiến trúc hệ thống

Hệ thống sử dụng kiến trúc **microservices** và đóng gói bằng **Docker**, bao gồm:

* ⚙️ **API Gateway (FastAPI)**
  Cung cấp các RESTful API endpoints, định tuyến đến từng Agent, xử lý xác thực và điều phối logic.

* 🔧 **Core Services**

  * `LLM Core`: Tích hợp GPT-4 (hoặc tương đương) qua LangChain
  * `Vector Store`: PostgreSQL + pgvector để lưu & tìm kiếm embedding
  * `Session Manager`: Quản lý lịch sử hội thoại và ngữ cảnh từng user
  * `Data Processing`: Script xử lý PDF, chunking và embedding

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ                                |
| ---------- | ---------------------------------------- |
| Backend    | Python, FastAPI                          |
| AI/ML      | LangChain, OpenAI, Sentence Transformers |
| Database   | PostgreSQL, pgvector                     |
| Triển khai | Docker, Docker Compose, Uvicorn          |

---

## 🚀 Bắt đầu

### 🔧 Yêu cầu

* Cài đặt **Docker** và **Docker Compose**
* Tạo tệp `.env` ở thư mục gốc

### 📅 Clone repository

```bash
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag
```

---

## ⚙️ Cấu hình `.env`

Tạo file `.env` với nội dung mẫu như sau:

```bash
DB_HOST=db
DB_PORT=5432
DB_NAME=agentic_rag_db
DB_USER=postgres
DB_PASSWORD=your_strong_password

OPENAI_API_KEY=your_openai_api_key
DEFAULT_LLM_MODEL=gpt-4-turbo
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

PGADMIN_EMAIL=admin@example.com
PGADMIN_PASSWORD=admin
```

> 💡 Bạn có thể đổi `EMBEDDING_MODEL_NAME` thành `intfloat/multilingual-e5-base` nếu muốn embedding chất lượng cao hơn cho tiếng Nhật.

---

## 🌐 Truy cập hệ thống

| Thành phần               | URL                                                      |
| ------------------------ | -------------------------------------------------------- |
| FastAPI API              | [http://localhost:8000](http://localhost:8000)           |
| Swagger UI (Docs)        | [http://localhost:8000/docs](http://localhost:8000/docs) |
| pgAdmin (PostgreSQL GUI) | [http://localhost:5050](http://localhost:5050)           |

---

## 📁 Cấu trúc thư mục

```text
/
├── .idea/                 # Cấu hình cho IDE (nếu dùng PyCharm)
├── data/
│   ├── input_pdfs/        # Chứa các file PDF đầu vào
│   └── manifest.json      # Thông tin cấu trúc dữ liệu đầu vào
├── src/
│   ├── api/               # FastAPI endpoints & schemas
│   ├── core/              # Thành phần cốt lõi: LLM, DB, session
│   ├── data_processing/   # Script xử lý PDF, chunking, embedding
│   ├── features/          # Logic riêng cho từng Agent (qna, planner, ...)
│   └── utils/             # Các hàm tiện ích dùng chung
├── Dockerfile             # Dockerfile để build API container
└── docker-compose.yml     # Tập tin cấu hình Docker Compose
```

---

## 📌 Ghi chú thêm

* Cấu hình embedding model, vector store và agent logic đều có thể mở rộng.
* Các agent có thể tương tác trực tiếp thông qua API hoặc tích hợp vào frontend như React/Vue.
* Bạn có thể tạo file `.env.example` và thêm vào `.gitignore` để hỗ trợ cộng tác viên.

---

## 📩 Liên hệ / Góp ý

Nếu bạn thấy dự án hữu ích, hãy ⭐ nó hoặc fork về để tùy biến theo nhu cầu.

---

## 📄 License

MIT License
