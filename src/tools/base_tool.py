# src/core/agent_tools.py
from langchain.agents import tool
from src.core.vector_store_interface import retrieve_relevant_documents_from_db

@tool
def find_learning_materials(level: str, skill: str) -> str:
    """
    Sử dụng tool này để tìm kiếm các tài liệu học tập (bài học, chương)
    dựa trên cấp độ (level) và kỹ năng (skill) người dùng yêu cầu.
    Ví dụ: level='N4', skill='Kanji'.
    Các kỹ năng hợp lệ: 'Vocabulary', 'Grammar', 'Kanji', 'Reading', 'Example'.
    """
    print(f"Tool 'find_learning_materials' được gọi với: level={level}, skill={skill}")

    # Chuyển đổi skill người dùng thành skill_type trong DB
    skill_map = {
        "từ vựng": "Vocabulary",
        "ngữ pháp": "Grammar",
        "kanji": "Kanji",
        "đọc": "Reading",
        "ví dụ": "Example"
    }
    skill_type_for_db = skill_map.get(skill.lower(), "Vocabulary")

    query = f"Tài liệu học {skill} cấp độ {level}"
    filters = {"level": level, "skill_type": skill_type_for_db}

    docs = retrieve_relevant_documents_from_db(query, top_k=5, filters=filters)

    if not docs:
        return f"Không tìm thấy tài liệu nào cho cấp độ {level} và kỹ năng {skill}."

    # Tổng hợp kết quả
    response_lines = [f"Đây là các tài liệu gợi ý cho {skill} cấp độ {level}:"]
    seen = set()
    for doc in docs:
        meta = doc.get("metadata", {})
        doc_name = meta.get("document", "N/A")
        lesson = meta.get("lesson", "N/A")
        # Tránh lặp lại cùng một bài học
        if (doc_name, lesson) not in seen:
            response_lines.append(f"- Trong tài liệu '{doc_name}', hãy xem qua '{lesson}'.")
            seen.add((doc_name, lesson))

    return "\n".join(response_lines)

# Bạn có thể thêm các tool khác ở đây nếu cần