# main_cli.py
import sys
import os
import json
import tempfile

# Import các hàm và service cần thiết
from src.core.rag_services import answer_question
from src.data_processing.pdf_document_processor import process_pdf_to_chunks, batch_insert_chunks_to_db, \
    get_db_connection
from src.data_processing.vocabulary_parser import VocabularyParser  # <-- THÊM IMPORT MỚI
from src.utils.storage_client import download_file_from_r2


def show_help():
    """Hiển thị hướng dẫn sử dụng."""
    print("\n--- AgenticRAG CLI Tool ---")
    print("Cách dùng:")
    print("  python main_cli.py chat                      : Bắt đầu chế độ chat AI.")
    print("  python main_cli.py process-doc <file_name>   : Tải và xử lý một file tài liệu chung từ R2.")
    print(
        "  python main_cli.py parse-vocab <file_name>   : Bóc tách file từ vựng và hiển thị ra JSON.")
    print("---------------------------\n")


def run_chat_mode():
    # ... (giữ nguyên hàm này) ...
    pass


def run_single_doc_processing(file_name: str):
    # ... (giữ nguyên hàm này) ...
    pass


# === THÊM HÀM MỚI ĐỂ PARSE TỪ VỰNG ===
def run_vocab_parsing(file_name: str):
    """
    Điều phối luồng bóc tách cho một file từ vựng cụ thể và in ra JSON.
    """
    print(f"\nBắt đầu quy trình bóc tách từ vựng cho file: '{file_name}'")

    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(project_root_dir, "data", "input_pdfs", file_name)

    # Khởi tạo và sử dụng Parser
    parser = VocabularyParser()
    structured_data = parser.parse(file_path)

    # Hiển thị kết quả dưới dạng JSON
    if structured_data:
        print("\n--- Kết quả bóc tách (JSON format): ---")
        # In ra dạng JSON cho dễ đọc, đảm bảo hiển thị đúng tiếng Việt
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
    else:
        print("Không có dữ liệu nào được trích xuất.")


# === CẬP NHẬT LOGIC MAIN ===
if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        show_help()
    else:
        command = args[1]
        if command == "chat":
            run_chat_mode()
        elif command == "process-doc":
            if len(args) < 3:
                print("Lỗi: Vui lòng cung cấp tên file cần xử lý.")
            else:
                run_single_doc_processing(args[2])
        # Thêm logic để xử lý lệnh mới
        elif command == "parse-vocab":
            if len(args) < 3:
                print("Lỗi: Vui lòng cung cấp tên file từ vựng cần bóc tách.")
            else:
                run_vocab_parsing(args[2])
        else:
            print(f"Lỗi: Lệnh '{command}' không được nhận dạng.")
            show_help()