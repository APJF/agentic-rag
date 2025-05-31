from src.core.rag_services import answer_question

if __name__ == "__main__":
    print("Chào mừng bạn đến với Demo RAG Nhật-Anh-Việt!")
    print("Nhập 'thoát', 'quit' hoặc 'exit' để dừng chương trình.")

    while True:
        user_query = input("\nBạn muốn hỏi gì về tiếng Nhật, tiếng Anh (hoặc tiếng Việt)?: ")
        if user_query.lower() in ["thoát", "quit", "exit"]:
            print("Cảm ơn bạn đã sử dụng demo! Tạm biệt.")
            break

        if not user_query.strip():
            print("Vui lòng nhập câu hỏi của bạn.")
            continue

        # Gọi hàm RAG mới từ service đã được module hóa
        answer = answer_question(user_query)
        print(f"\nTrả lời: {answer}")