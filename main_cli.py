# main_cli.py
import sys
import os

# === IMPORT CÁC MODULE CHỨC NĂNG MỚI ===
from src.core.rag_services import answer_question
from src.core.quiz_generator import generate_quiz_question
from src.core.conversational_agent import initialize_planning_agent

def show_help():
    """Hiển thị hướng dẫn sử dụng mới."""
    print("\n--- Trợ lý Học tiếng Nhật CLI ---")
    print("Cách dùng:")
    print("  python main_cli.py chat              : Bắt đầu chế độ chat hỏi-đáp thông thường.")
    print("  python main_cli.py plan              : Bắt đầu chế độ tư vấn lộ trình học tập.")
    print("  python main_cli.py quiz <level> [topic] : Tạo câu hỏi (topic là tùy chọn).")
    print("    Ví dụ 1: python main_cli.py quiz N4")
    print("    Ví dụ 2: python main_cli.py quiz N3 Business")
    print("----------------------------------\n")

def run_chat_mode():
    """Chức năng 1: Chat hỏi-đáp thông thường."""
    print("\n--- Chế độ Hỏi-Đáp ---")
    print("Bạn có thể hỏi bất cứ điều gì về tiếng Nhật. Gõ 'exit' để thoát.")
    print("-----------------------\n")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("AI: Tạm biệt!")
                break
            if user_input.strip():
                ai_response = answer_question(user_input)
                print(f"AI: {ai_response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nAI: Tạm biệt!")
            break


def run_planning_mode():
    """Chức năng 3: Tư vấn lộ trình học tập (Agentic)."""
    print("\n--- Chế độ Tư vấn Lộ trình ---")
    print("AI sẽ trò chuyện để xây dựng lộ trình. Gõ 'exit' để thoát.")
    print("Hãy bắt đầu bằng cách chào hoặc nêu yêu cầu của bạn (ví dụ: 'tư vấn cho tôi').")
    print("--------------------------------\n")

    # Khởi tạo agent và bộ nhớ của nó MỘT LẦN DUY NHẤT
    agent_executor = initialize_planning_agent()
    if not agent_executor:
        print("Không thể khởi tạo Agent. Vui lòng kiểm tra lại cấu hình.")
        return

    # Bắt đầu vòng lặp hội thoại
    while True:
        try:
            # Chờ người dùng nhập liệu
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("AI: Tạm biệt! Chúc bạn học tốt.")
                break

            if user_input.strip():
                # Gọi agent với input của người dùng
                # Agent sẽ tự động sử dụng bộ nhớ để truy cập lịch sử hội thoại
                response = agent_executor.invoke({"input": user_input})
                print(f"AI: {response['output']}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nAI: Tạm biệt! Chúc bạn học tốt.")
            break
        except Exception as e:
            print(f"AI: Đã có lỗi xảy ra: {e}")


def run_quiz_mode(level: str, topic: str = None):
    """Chức năng 2: Tạo câu hỏi trắc nghiệm với level và topic tùy chọn."""
    topic_info = f" và chủ đề '{topic}'" if topic else ""
    print(f"\n--- Chế độ Quiz (Level: {level}{topic_info}) ---")
    print("Đang tạo câu hỏi cho bạn, vui lòng chờ...")

    question = generate_quiz_question(level, topic)
    print("\n" + "="*15 + " CÂU HỎI MỚI " + "="*15)
    print(question)
    print("="*42 + "\n")


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        show_help()
        exit()

    command = args[1].lower()

    if command == "chat":
        run_chat_mode()
    elif command == "plan":
        run_planning_mode()
    elif command == "quiz":
        if len(args) < 3:
            print("Lỗi: Vui lòng cung cấp ít nhất level (N5, N4, N3...).")
            show_help()
        else:
            level_arg = args[2].upper()
            # Lấy topic nếu có
            topic_arg = args[3] if len(args) > 3 else None

            if level_arg not in ["N5", "N4", "N3", "N2", "N1"]:
                 print(f"Lỗi: Level '{level_arg}' không hợp lệ.")
            else:
                run_quiz_mode(level_arg, topic_arg)
    else:
        print(f"Lỗi: Lệnh '{command}' không được nhận dạng.")
        show_help()