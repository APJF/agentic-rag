# main_cli.py

import sys
import traceback
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage

# Import các hàm logic chính và quản lý phiên
# Giả sử orchestrator nằm trong src/main_orchestrator.py
# Nếu bạn đã chuyển sang router.py, hãy đổi tên import thành from src.router import agentic_router_chain
from src.main_orchestrator import run_orchestrator
from src.core.session_manager import (
    get_or_create_user,
    list_sessions_for_user,
    create_new_session,
    load_chat_history,
    add_new_messages
)


def select_or_create_session(user_id: str) -> Optional[int]:
    """
    Hàm này có nhiệm vụ hiển thị menu lựa chọn phiên cho người dùng.
    - Liệt kê các phiên đang có.
    - Cho phép người dùng chọn một phiên cũ hoặc tạo một phiên mới.
    - Trả về ID (kiểu số) của phiên được chọn, hoặc None nếu người dùng muốn thoát.
    """
    print("\nĐang tải các phiên làm việc của bạn...")
    sessions = list_sessions_for_user(user_id)

    if sessions:
        print(f"Chào mừng {user_id}! Vui lòng chọn một phiên để tiếp tục:")
        for i, session in enumerate(sessions):
            # Hiển thị thời gian cập nhật một cách thân thiện
            updated_time = session['updated_at'].strftime("%Y-%m-%d %H:%M")
            print(f"  {i + 1}. {session['name']} (Cập nhật lúc: {updated_time})")
    else:
        print(f"Chào {user_id}! Bạn chưa có phiên làm việc nào.")

    print("---")
    print("  N. Bắt đầu một phiên mới")
    print("  Q. Thoát chương trình")

    while True:
        choice = input("Lựa chọn của bạn: ").strip().upper()
        if choice == 'Q':
            return None  # Trả về None để báo hiệu thoát chương trình
        elif choice == 'N':
            session_name = input("Vui lòng đặt tên cho phiên mới: ").strip()
            if session_name:
                return create_new_session(user_id, session_name)
            else:
                print("Tên phiên không được để trống.")
        elif choice.isdigit() and 1 <= int(choice) <= len(sessions):
            # Lấy ID của session từ danh sách đã tải
            return sessions[int(choice) - 1]['id']
        else:
            print("Lựa chọn không hợp lệ, vui lòng chọn lại.")


def run_chat_session(session_id: int, user_id: str):
    """
    Khởi tạo và chạy vòng lặp chat chính cho một phiên cụ thể,
    đồng thời hiển thị lại lịch sử chat cũ cho người dùng.
    """
    # Tải toàn bộ lịch sử chat của phiên này từ database
    chat_history = load_chat_history(session_id)

    print("\n" + "=" * 20 + " BẮT ĐẦU PHIÊN LÀM VIỆC " + "=" * 20)

    # === BẮT ĐẦU SỬA LỖI: THÊM VÒNG LẶP ĐỂ IN LỊCH SỬ ===
    if chat_history:
        print("   --- Lịch sử trò chuyện trước đó ---")
        for message in chat_history:
            if isinstance(message, HumanMessage):
                print(f"{user_id}: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"AI: {message.content}\n")
        print("   --- Cuộc trò chuyện tiếp tục ---\n")
    else:
        print("--- Bắt đầu phiên trò chuyện mới. Bạn hãy bắt đầu nhé! ---\n")
    # === KẾT THÚC SỬA LỖI ===

    # Vòng lặp chat chính
    while True:
        try:
            # Dấu nhắc nhập liệu được chuyển ra ngoài để gọn hơn
            user_input = input(f"{user_id}: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                # Khi thoát, sẽ quay lại menu chọn phiên
                print("Đang quay lại menu chính...")
                break

            if user_input.strip():
                # Gọi orchestrator với input và lịch sử chat đã tải
                ai_response_text = run_orchestrator(user_input, chat_history)
                print(f"AI: {ai_response_text}\n")

                # Chuẩn bị tin nhắn mới để cập nhật
                human_msg = HumanMessage(content=user_input)
                ai_msg = AIMessage(content=ai_response_text)

                # Cập nhật lịch sử trong bộ nhớ
                chat_history.extend([human_msg, ai_msg])

                # Lưu 2 tin nhắn mới nhất vào database
                add_new_messages(session_id, [human_msg, ai_msg])

        except Exception as e:
            print(f"AI: Rất tiếc, đã có một lỗi nghiêm trọng xảy ra: {e}")
            traceback.print_exc()


def main():
    """
    Hàm chính điều khiển toàn bộ ứng dụng, từ lúc đăng nhập đến khi thoát.
    """
    print("\n--- TRỢ LÝ ẢO TIẾNG NHẬT ---")
    # Bước A: Đăng nhập
    user_id = input("Vui lòng nhập tên người dùng của bạn để bắt đầu: ").strip()

    if not user_id:
        print("Tên người dùng không được để trống. Thoát chương trình.")
        sys.exit(1)

    get_or_create_user(user_id)

    while True:
        session_id_to_run = select_or_create_session(user_id)

        if session_id_to_run is None:
            break

        run_chat_session(session_id_to_run, user_id)

    print("\nCảm ơn bạn đã sử dụng. Hẹn gặp lại!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChương trình đã dừng.")