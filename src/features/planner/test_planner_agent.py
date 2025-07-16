# test_planner_agent.py
import sys
import traceback
from langchain_core.messages import AIMessage, HumanMessage

from src.features.planner.agent import initialize_planning_agent


def run_planner_test():
    """
    Hàm này tạo ra một môi trường chat giả lập để test riêng PlannerAgent.
    """
    print("\n" + "=" * 20 + " BẮT ĐẦU PHIÊN TEST PLANNER AGENT " + "=" * 20)

    print("Initializing Planner Agent...")
    planner_agent_executor = initialize_planning_agent()
    if not planner_agent_executor:
        print("Không thể khởi tạo Planner Agent. Dừng chương trình.")
        sys.exit(1)
    print("Planner Agent is ready.\n")

    user_id_for_test = "test_user_planner"
    chat_history = []

    print(f"Bắt đầu phiên test cho người dùng: '{user_id_for_test}'")
    print("Gõ 'exit' hoặc 'quit' để thoát.")
    print("--------------------------------------------------\n")

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
                result = planner_agent_executor.invoke(input_data)
                ai_response_text = result.get('output', "Lỗi: Agent không có output.")
                print(f"Planner Agent: {ai_response_text}\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=ai_response_text))

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"AI: Rất tiếc, đã có một lỗi nghiêm trọng xảy ra: {e}")
            traceback.print_exc()

    print("\n--- KẾT THÚC PHIÊN TEST ---")

if __name__ == "__main__":
    run_planner_test()