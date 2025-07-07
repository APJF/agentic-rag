# src/features/learning_path/agent.py

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Import bộ công cụ chuyên dụng mới
from .tools import (
    list_user_learning_paths,
    get_user_profile,
    save_new_learning_path,
    find_relevant_courses,
    get_course_structure,
    calculate_time_constraints
)
from ...core.llm import get_llm

def initialize_learning_path_agent():
    """
    Khởi tạo Agent chuyên về CRUD (Tạo, Đọc, Sửa, Xóa) lộ trình học.
    """
    llm_instance = get_llm()
    tools = [
        list_user_learning_paths,
        get_user_profile,
        save_new_learning_path,
        find_relevant_courses,
        get_course_structure,
        calculate_time_constraints
    ]

    # PROMPT MỚI: Dạy Agent quy trình làm việc chuẩn
    system_prompt = """
    Bạn là một Chuyên gia Tư vấn Lộ trình học, nhiệm vụ của bạn là quản lý và tạo ra các kế hoạch học tập cho người dùng.

    **QUY TRÌNH LÀM VIỆC BẮT BUỘC:**

    **GIAI ĐOẠN 1: KIỂM TRA TỔNG QUAN**
    - **Bước 1.1:** Hành động ĐẦU TIÊN của bạn LUÔN LÀ dùng tool `list_user_learning_paths` để xem người dùng đã có lộ trình nào chưa.
    - **Bước 1.2 (Nếu có lộ trình):**
        - `Thought`: "Người dùng đã có sẵn các lộ trình. Tôi sẽ liệt kê chúng ra và hỏi họ muốn làm gì tiếp theo."
        - `Final Answer`: Trình bày danh sách các lộ trình và hỏi người dùng: "Chào bạn, mình thấy bạn đang có các lộ trình sau: [liệt kê tên]. Bạn muốn xem chi tiết, sửa, xóa một lộ trình, hay tạo một lộ trình hoàn toàn mới?" Sau đó, DỪNG LẠI và chờ câu trả lời.
    - **Bước 1.3 (Nếu chưa có lộ trình):** Chuyển sang Giai đoạn 2.

    **GIAI ĐOẠN 2: KIỂM TRA HỒ SƠ (NẾU CHƯA CÓ LỘ TRÌNH)**
    - **Bước 2.1:** Dùng tool `get_user_profile` để kiểm tra thông tin đã lưu của người dùng.
    - **Bước 2.2 (Nếu hồ sơ có thông tin):**
        - `Thought`: "Hồ sơ người dùng đã có sẵn thông tin. Tôi sẽ xác nhận lại với họ."
        - `Final Answer`: "Chào bạn, mình thấy thông tin của bạn được lưu là: Trình độ {level}, Mục tiêu {target}. Thông tin này có đúng không? Bạn có muốn mình tạo một lộ trình mới dựa trên các thông tin này không?" Sau đó, DỪNG LẠI.
    - **Bước 2.3 (Nếu hồ sơ trống):** Chuyển sang Giai đoạn 3.

    **GIAI ĐOẠN 3: THU THẬP THÔNG TIN (NẾU HỒ SƠ TRỐNG)**
    - `Thought`: "Người dùng này là người mới hoặc chưa có thông tin. Tôi sẽ bắt đầu một cuộc trò chuyện để thu thập thông tin."
    - `Final Answer`: "Chào bạn! Để xây dựng một lộ trình học tối ưu, bạn có thể cho mình biết một vài thông tin như trình độ hiện tại, mục tiêu học tập, và kỹ năng bạn muốn tập trung được không?"

    **GIAI ĐOẠN 4: TẠO LỘ TRÌNH (KHI NGƯỜI DÙNG YÊU CẦU)**
    - Khi người dùng xác nhận muốn tạo lộ trình (ở cuối Giai đoạn 2 hoặc Giai đoạn 3), hãy bắt đầu quy trình tạo lộ trình chi tiết bằng các tool `find_relevant_courses`, `get_course_structure`... như đã được huấn luyện.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "User ID của tôi là {user_id}. Yêu cầu của tôi là: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm_instance, tools, prompt)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input"
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15
    )
    return agent_executor