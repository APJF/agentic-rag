# run_learning_path_service.py

import sys
import traceback
from langchain_core.messages import AIMessage, HumanMessage

# Import trực tiếp agent mới
from src.features.learning_path.agent import initialize_learning_path_agent


def run_path_service_test():
    """
    Hàm này tạo ra một môi trường chat giả lập để test riêng LearningPathAgent.
    """
    print("\n" + "=" * 20 + " BẮT ĐẦU PHIÊN TEST LEARNING PATH SERVICE " + "=" * 20)

    lp_agent_executor = initialize_learning_path_agent()
    if not lp_agent_executor:
        print("Không thể khởi tạo Learning Path Agent. Dừng chương trình.")
        sys.exit(1)

    user_id_for_test = input("Vui lòng nhập User ID để test: ").strip()
    chat_history = []

    print(f"\nBắt đầu phiên làm việc cho người dùng: '{user_id_for_test}'")
    print("Gõ 'exit' hoặc 'quit' để thoát.")
    print("--------------------------------------------------\n")

    # Bắt đầu với một input rỗng để kích hoạt agent kiểm tra trạng thái ban đầu
    initial_input = "Bắt đầu"

    while True:
        try:
            if initial_input:
                user_input = initial_input
                print(f"{user_id_for_test}: (Bắt đầu phiên)")
                initial_input = None  # Chỉ chạy lần đầu
            else:
                user_input = input(f"{user_id_for_test}: ")

            if user_input.lower() in ['exit', 'quit', 'bye']:
                break

            if user_input.strip():
                input_data = {
                    "user_id": user_id_for_test,
                    "input": user_input,
                    "chat_history": chat_history
                }

                result = lp_agent_executor.invoke(input_data)
                ai_response_text = result.get('output', "Lỗi: Agent không có output.")

                print(f"LP Agent: {ai_response_text}\n")

                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=ai_response_text))

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"AI: Rất tiếc, đã có một lỗi nghiêm trọng xảy ra: {e}")
            traceback.print_exc()

    print("\n--- KẾT THÚC PHIÊN TEST ---")


if __name__ == "__main__":
    run_path_service_test()