# main_cli.py

import sys
import traceback
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage
from src.core.session_manager import rewind_last_turn
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
            print(f"  {i + 1}. {session['session_name']} (Cập nhật lúc: {updated_time})")
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
    Khởi tạo và chạy vòng lặp chat chính, có thêm chức năng sửa tin nhắn cuối cùng.
    """
    chat_history = load_chat_history(session_id)

    print("\n" + "=" * 20 + " BẮT ĐẦU PHIÊN LÀM VIỆC " + "=" * 20)

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

    print("Gõ '/edit' để sửa tin nhắn vừa gửi, hoặc 'exit' để quay lại menu.")
    print("------------------------------------------------------------------\n")

    while True:
        try:
            user_input = input(f"{user_id}: ")
            if user_input.strip().lower() == '/edit':
                if len(chat_history) < 2:
                    print("AI: Không có đủ lịch sử để sửa. Vui lòng gửi ít nhất một tin nhắn.\n")
                    continue
                if rewind_last_turn(session_id):
                    chat_history.pop()
                    chat_history.pop()
                    print("\nAI: Đã xóa lượt nói cuối cùng. Mời bạn nhập lại tin nhắn.")
                    if chat_history:
                        print(f"AI (lượt trước đó): {chat_history[-1].content}\n")
                    new_input = input(f"[{user_id} sửa lại]: ")
                    user_input = new_input
                else:
                    print("AI: Có lỗi xảy ra khi cố gắng sửa tin nhắn trên database. Vui lòng thử lại.\n")
                    continue
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break

            if user_input.strip():
                ai_response_text = run_orchestrator(user_input, chat_history)
                print(f"AI: {ai_response_text}\n")

                human_msg = HumanMessage(content=user_input)
                ai_msg = AIMessage(content=ai_response_text)

                chat_history.extend([human_msg, ai_msg])
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