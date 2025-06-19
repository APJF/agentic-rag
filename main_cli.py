# main_cli.py

import sys
from langchain_core.messages import AIMessage, HumanMessage
from src.main_orchestrator import run_orchestrator

def run_unified_chat_mode():
    """
    Một vòng lặp chat duy nhất, thống nhất, sử dụng orchestrator để định tuyến mọi yêu cầu.
    """
    print("\n--- Trợ lý ảo tiếng Nhật đã sẵn sàng! ---")
    print("Bạn cần giúp gì hôm nay? (Ví dụ: 'tạo lộ trình cho người mới bắt đầu', 'cho tôi một câu hỏi N3', 'sửa câu này giúp mình', ...)")
    print("Gõ 'exit' hoặc 'quit' để thoát.")
    print("------------------------------------------\n")

    chat_history = []

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("AI: Tạm biệt! Hẹn gặp lại bạn.")
                break

            if user_input.strip():
                ai_response_text = run_orchestrator(user_input, chat_history)

                print(f"AI: {ai_response_text}\n")

                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=ai_response_text))

        except (KeyboardInterrupt, EOFError):
            print("\nAI: Tạm biệt! Hẹn gặp lại bạn.")
            break
        except Exception as e:
            print(f"AI: Rất tiếc, đã có một lỗi nghiêm trọng xảy ra: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_unified_chat_mode()