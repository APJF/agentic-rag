from langchain.tools import tool

@tool
def get_speaking_topic(level: str) -> str:
    """
    Trả về một chủ đề luyện nói phù hợp với level (N5-N1).
    """
    topics = {
        "N5": "Giới thiệu bản thân, gia đình, sở thích.",
        "N4": "Kể về một ngày đi học/làm việc.",
        "N3": "Thảo luận về du lịch, sở thích, công việc.",
        "N2": "Tranh luận về một vấn đề xã hội.",
        "N1": "Thuyết trình về chủ đề chuyên sâu."
    }
    return topics.get(level.upper(), "Hãy giới thiệu bản thân bằng tiếng Nhật.")

@tool
def analyze_speaking_answer(answer: str) -> str:
    """
    Phân tích câu trả lời speaking, chỉ ra lỗi và gợi ý cải thiện.
    """
    # Ở đây có thể tích hợp LLM hoặc logic đơn giản
    return "Câu trả lời của bạn khá tốt! Hãy chú ý ngữ pháp và phát âm."