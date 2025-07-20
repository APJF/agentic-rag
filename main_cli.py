# main_cli.py

import sys
import traceback
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage

from src.core.session_manager import (
    get_or_create_user,
    list_sessions_for_user,
    create_new_session,
    load_chat_history,
    add_new_messages,
    rewind_last_turn
)


def select_or_create_session(user_id: str) -> Optional[int]:
    """Hiển thị menu cho người dùng chọn hoặc tạo phiên mới."""
    print("\nĐang tải các phiên làm việc của bạn...")
    sessions = list_sessions_for_user(user_id)

    if not sessions:
        print(f"Chào {user_id}! Bạn chưa có phiên làm việc nào.")
    else:
        print(f"Chào mừng trở lại, {user_id}! Các phiên làm việc gần đây của bạn:")
        for i, session in enumerate(sessions):
            updated_time = session['updated_at'].strftime("%H:%M %d-%m-%Y")
            print(f"  {i + 1}. {session['session_name']} (Cập nhật lúc: {updated_time})")

    print("---")
    print("  N. Bắt đầu một phiên mới")
    print("  Q. Thoát chương trình")

    while True:
        choice = input("Lựa chọn của bạn: ").strip().upper()
        if choice == 'Q':
            return None
        elif choice == 'N':
            session_name = input("Vui lòng đặt tên cho phiên mới: ").strip()
            if session_name:
                return create_new_session(user_id, session_name)
            else:
                print("Tên phiên không được để trống.")
        elif choice.isdigit() and 1 <= int(choice) <= len(sessions):
            return sessions[int(choice) - 1]['id']
        else:
            print("Lựa chọn không hợp lệ, vui lòng chọn lại.")


def run_chat_session(session_id: int, user_id: str):
    """
    Khởi tạo và chạy vòng lặp chat chính cho một phiên cụ thể.
    Đã được sửa lỗi gọi invoke hai lần.
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
                # === SỬA LỖI Ở ĐÂY ===

                # 1. Chuẩn bị dữ liệu đầu vào
                input_data = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "input": user_input,
                    "chat_history": chat_history
                }

                # 2. Chỉ cần gọi invoke MỘT LẦN DUY NHẤT
                # agentic_router_chain sẽ tự động định tuyến và chạy agent phù hợp
                result = input_data

                # 3. Xử lý kết quả trả về
                if isinstance(result, dict) and 'output' in result:
                    ai_response_text = result['output']
                else:
                    ai_response_text = str(result)

                print(f"AI: {ai_response_text}\n")

                # 4. Lưu lại lịch sử
                human_msg = HumanMessage(content=user_input)
                ai_msg = AIMessage(content=ai_response_text)
                chat_history.extend([human_msg, ai_msg])
                add_new_messages(session_id, [human_msg, ai_msg])

        except Exception as e:
            print(f"AI: Rất tiếc, đã có một lỗi nghiêm trọng xảy ra: {e}")
            traceback.print_exc()


def main():
    """Hàm chính điều khiển toàn bộ ứng dụng CLI."""
    print("--- TRỢ LÝ ẢO TIẾNG NHẬT ---")
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
