# src/core/grammar_checker.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import get_llm

def correct_japanese_text(text_to_correct: str, user_level: str) -> str:
    """
    Sử dụng LLM để sửa lỗi và đưa ra gợi ý cho một đoạn văn tiếng Nhật
    dựa trên trình độ của người học.

    Args:
        text_to_correct: Đoạn văn tiếng Nhật người dùng nhập vào.
        user_level: Trình độ của người dùng (ví dụ: 'N5', 'N4').

    Returns:
        Một chuỗi văn bản đã được định dạng với câu sửa và phân tích.
    """
    llm = get_llm()
    if not llm:
        return "Lỗi: Dịch vụ LLM chưa được cấu hình."

    # Đây là trái tim của chức năng: một prompt chi tiết và mạnh mẽ
    system_prompt = f"""
    Bạn là một giáo viên tiếng Nhật tận tâm, thông thái và rất giỏi trong việc giải thích.
    Nhiệm vụ của bạn là sửa lỗi và cải thiện một câu hoặc đoạn văn tiếng Nhật cho người học ở trình độ **{user_level}**.

    **QUY TẮC BẮT BUỘC KHI SỬA LỖI:**
    1.  **Sửa lỗi sai:** Sửa tất cả các lỗi về ngữ pháp, chính tả, trợ từ (は, が, を, に...), và cách dùng từ thiếu tự nhiên.
    2.  **Đề xuất theo trình độ:** Nếu người học dùng từ vựng hoặc ngữ pháp quá cao cấp so với trình độ `{user_level}`, hãy ghi nhận là họ dùng đúng, nhưng đồng thời đề xuất một phương án thay thế đơn giản và phổ biến hơn ở cấp độ của họ. Việc này giúp họ giao tiếp tự nhiên hơn với những gì đã học.
    3.  **Cải thiện sự tự nhiên:** Nếu câu đúng ngữ pháp nhưng người Nhật không nói như vậy, hãy đề xuất cách diễn đạt tự nhiên hơn.

    **ĐỊNH DẠNG ĐẦU RA (BẮT BUỘC):**
    Hãy trình bày kết quả theo đúng định dạng Markdown dưới đây. Đừng thêm bất cứ nội dung nào khác ngoài định dạng này.
    ---
    **Câu gốc:**
    [Ghi lại chính xác câu gốc của người dùng ở đây]

    **Câu đã sửa:**
    [Ghi lại phiên bản câu đã được sửa lỗi hoàn chỉnh ở đây]

    **Phân tích chi tiết:**
    - **Lỗi 1 (nếu có):** [Mô tả lỗi, ví dụ: "Trợ từ 'が' dùng chưa chính xác trong trường hợp này."] -> **Sửa thành:** [Cách sửa, ví dụ: "dùng trợ từ 'は' để nhấn mạnh chủ đề."]
    - **Lỗi 2 (nếu có):** ...
    - **Gợi ý 1 (nếu có):** [Mô tả gợi ý, ví dụ: "Bạn đã dùng ngữ pháp '～ものだから' (N2), rất hay! Tuy nhiên, ở trình độ {user_level}, bạn có thể dùng '～から' hoặc '～ので' sẽ phổ biến hơn."]
    - **Gợi ý 2 (nếu có):** ...
    ---
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{text_input}")
    ])

    # Tạo một chuỗi xử lý đơn giản
    chain = prompt | llm | StrOutputParser()

    # Gọi chuỗi xử lý với đầu vào
    response = chain.invoke({"text_input": text_to_correct})
    return response