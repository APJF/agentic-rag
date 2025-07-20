# test_qna_agent.py

import sys
import traceback
from langchain_core.messages import AIMessage, HumanMessage

# Import trực tiếp hàm khởi tạo của QnAAgent
from src.features.qna.agent import initialize_qna_agent


def run_qna_test():
    """
    Hàm này tạo ra một môi trường chat giả lập để test riêng QnAAgent.
    """
    print("\n" + "=" * 20 + " BẮT ĐẦU PHIÊN TEST QNA AGENT " + "=" * 20)

    # --- 1. KHỞI TẠO AGENT ---
    # Khởi tạo trực tiếp QnAAgent
    print("Initializing QnA Agent...")
    qna_agent_executor = initialize_qna_agent()
    if not qna_agent_executor:
        print("Không thể khởi tạo QnA Agent. Dừng chương trình.")
        sys.exit(1)
    print("QnA Agent is ready.\n")

    # --- 2. THIẾT LẬP THÔNG TIN GIẢ LẬP ---
    # Cung cấp một user_id giả lập để agent có thể chạy
    user_id_for_test = "test_user_qna"

    # Bộ nhớ tạm thời cho phiên test này, sẽ được reset mỗi khi chạy lại file
    chat_history = []

    print(f"Bắt đầu phiên test cho người dùng: '{user_id_for_test}'")
    print("Gõ 'exit' hoặc 'quit' để thoát.")
    print("--------------------------------------------------\n")

    # --- 3. VÒNG LẶP CHAT TRỰC TIẾP VỚI AGENT ---
    while True:
        try:
            user_input = input(f"{user_id_for_test}: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break

            if user_input.strip():
                input_data = {
                    "user_id": user_id_for_test,
                    "input": user_input,
                    "chat_history": chat_history
                }
                result = qna_agent_executor.invoke(input_data)
                ai_response_text = result.get('output', "Lỗi: Agent không có output.")

                print(f"QnA Agent: {ai_response_text}\n")

                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=ai_response_text))

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"AI: Rất tiếc, đã có một lỗi nghiêm trọng xảy ra: {e}")
            traceback.print_exc()

    print("\n--- KẾT THÚC PHIÊN TEST ---")


# --- Điểm khởi đầu để chạy file test ---
if __name__ == "__main__":
    run_qna_test()