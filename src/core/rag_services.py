import re
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.core.llm_services import get_llm
from src.core.vector_store_interface import (
    retrieve_relevant_documents_from_db,
    find_precise_definitional_source_from_db,
)
from src.core.prompt_templates import main_rag_prompt, explanation_prompt
from src.utils.text_utils import contains_japanese_char, contains_kanji


def format_docs_for_rag(docs: list[dict]) -> str:

    if not docs: return "Hiện tại không tìm thấy thông tin liên quan trực tiếp trong cơ sở tri thức."
    formatted_strings = []
    for i, doc_item in enumerate(docs):
        text = doc_item.get("text", "")
        meta = doc_item.get("metadata", {})
        source_parts = []
        if meta.get('document') and meta.get('document') != 'N/A': source_parts.append(
            f"Tài liệu: {meta.get('document')}")
        if meta.get('lesson') and meta.get('lesson') != "Unknown Lesson": source_parts.append(
            f"Bài: {meta.get('lesson')}")
        if meta.get('page'): source_parts.append(f"Trang: {meta.get('page')}")
        source_str = ", ".join(source_parts) if source_parts else "Nguồn không rõ"
        formatted_strings.append(f"Đoạn trích {i + 1} (từ {source_str}):\n---\n{text}\n---")
    return "\n\n".join(formatted_strings)


def get_output_instruction_for_rag(query_text_original: str, is_translation_request: bool,
                                vietnamese_phrase: str | None = None) -> str:
    base_instruction = "Hãy trả lời bằng tiếng Việt một cách chi tiết, rõ ràng và dễ hiểu."
    if is_translation_request and vietnamese_phrase:
        return f"""{base_instruction}
Khi người dùng yêu cầu dịch từ "{vietnamese_phrase}" sang tiếng Nhật, hãy cung cấp thông tin theo cấu trúc sau:
1. **Bản dịch trực tiếp:** "{vietnamese_phrase}" trong tiếng Nhật là: [Kanji nếu có]（[Hiragana/Katakana bắt buộc], [Romaji]）
2. **Phân tích chi tiết (nếu là một từ đơn có Kanji và bạn có thông tin):**
   * Kanji: [Chữ Kanji của từ chính]
   * Âm Hán Việt: [Âm Hán Việt (nếu biết)]
   * Nghĩa gốc của Kanji/Từ: [Giải thích ý nghĩa]
   * Các từ liên quan thường gặp (2-3 từ, Kanji/Kana - Hiragana): [Từ 1], [Từ 2]
3. **Lời khuyên và Ví dụ:** Hãy luyện tập sử dụng từ này nhé! Ví dụ: [Câu ví dụ tiếng Nhật] – [Dịch nghĩa tiếng Việt]
---
Áp dụng cấu trúc trên cho "{vietnamese_phrase}":"""
    else:
        return base_instruction


llm_instance = get_llm()

if llm_instance:
    _main_rag_chain = (
        {
            "context": lambda x: format_docs_for_rag(
                retrieve_relevant_documents_from_db(x["question_for_rag"], table_name=settings.CHUNK_TABLE_NAME)
            ),
            "question": lambda x: x["original_question"],
            "output_instruction_and_format": lambda x: get_output_instruction_for_rag(
                x["original_question"],
                x["is_translation_request"],
                x.get("vietnamese_phrase")
            )
        }
        | main_rag_prompt
        | llm_instance
        | StrOutputParser()
    )
else:
    _main_rag_chain = None


def answer_question(question: str) -> str:
    if not _main_rag_chain:
        return "LLM hoặc RAG chain chưa được cấu hình."

    main_llm_response = ""
    identified_jp_term_for_source_lookup = None
    original_term_for_suggestion = question

    is_translation = False
    vietnamese_phrase_to_translate = None
    question_for_rag_context_retrieval = question

    translation_query_match = re.match(
        r"^(dịch\s+(?:sang\s+)?tiếng\s+nhật|translate\s+to\s+japanese)\s*[:]*\s*(.+)$",
        question.strip(), re.IGNORECASE
    )

    if translation_query_match:
        vietnamese_phrase_to_translate = translation_query_match.group(2).strip()
        if not vietnamese_phrase_to_translate:
            return "Vui lòng cung cấp từ/cụm từ tiếng Việt bạn muốn dịch."
        is_translation = True
        original_term_for_suggestion = vietnamese_phrase_to_translate
        question_for_rag_context_retrieval = f"Thông tin chi tiết về từ tiếng Việt '{vietnamese_phrase_to_translate}' và bản dịch, cách dùng tương ứng trong tiếng Nhật."

    main_llm_response = _main_rag_chain.invoke({
        "original_question": question,
        "question_for_rag": question_for_rag_context_retrieval,
        "is_translation_request": is_translation,
        "vietnamese_phrase": vietnamese_phrase_to_translate if is_translation else None
    })

    if is_translation:
        jp_match_in_translation = re.search(r"([一-龯ぁ-んァ-ンヴー]+)\s*（\s*([ぁ-んァ-ンヴー]+)\s*,", main_llm_response)
        if jp_match_in_translation:
            identified_jp_term_for_source_lookup = jp_match_in_translation.group(1)
            if not identified_jp_term_for_source_lookup or not contains_japanese_char(identified_jp_term_for_source_lookup):
                identified_jp_term_for_source_lookup = jp_match_in_translation.group(2)
        else:
            first_line_of_response = main_llm_response.split('\n')[0] if '\n' in main_llm_response else main_llm_response
            translation_part_match = re.search(r"là:\s*([一-龯ぁ-んァ-ンヴー]+)", first_line_of_response)
            if translation_part_match: identified_jp_term_for_source_lookup = translation_part_match.group(1)
    elif not is_translation:
        question_words = re.findall(r"[一-龯ぁ-んァ-ンヴー]+", question)
        if question_words:
            kanji_words = [w for w in question_words if contains_kanji(w)]
            if kanji_words: identified_jp_term_for_source_lookup = max(kanji_words, key=len)
            else: identified_jp_term_for_source_lookup = max(question_words, key=len)


    precise_source_info_text = ""
    if identified_jp_term_for_source_lookup:
        source_metadata = find_precise_definitional_source_from_db(
            identified_jp_term_for_source_lookup,
            table_name=settings.CHUNK_TABLE_NAME
        )
        if source_metadata:
            doc = source_metadata.get('document', 'tài liệu không rõ')
            lesson = source_metadata.get('lesson', 'bài không rõ')
            page = source_metadata.get('page')
            page_info = f", trang {page}" if page else ""
            term_display = f"'{identified_jp_term_for_source_lookup}'"
            if is_translation and original_term_for_suggestion and original_term_for_suggestion.lower() != identified_jp_term_for_source_lookup.lower():
                term_display = f"từ khóa '{original_term_for_suggestion}' (tương ứng tiếng Nhật là '{identified_jp_term_for_source_lookup}')"
            precise_source_info_text = f"\n\n **Nguồn tham khảo cho {term_display}:** Bạn có thể tìm thấy từ này trong **{lesson}** của tài liệu **{doc}**{page_info}."

    final_response = main_llm_response.strip() + precise_source_info_text
    return final_response

# --- Hàm cho Quiz ---
def generate_explanation_for_quiz(
        question_text: str, options: dict, correct_option_letter: str,
        correct_option_text: str, related_keywords: str | None) -> str:
    if not llm_instance: return "Xin lỗi, dịch vụ LLM chưa được cấu hình."
    context_for_explanation = "Không tìm thấy ngữ cảnh cụ thể."
    if related_keywords:
        retrieved_docs_data = retrieve_relevant_documents_from_db(
            f"Thông tin về: {related_keywords}", top_k=1, table_name=settings.CHUNK_TABLE_NAME
        )
        if retrieved_docs_data: context_for_explanation = format_docs_for_rag(retrieved_docs_data)

    explanation_chain = explanation_prompt | llm_instance | StrOutputParser()
    try:
        explanation = explanation_chain.invoke({
            "question": question_text, "option_a": options.get("A", "N/A"),
            "option_b": options.get("B", "N/A"), "option_c": options.get("C", "N/A"),
            "option_d": options.get("D", "N/A"), "correct_letter": correct_option_letter,
            "correct_text": correct_option_text, "context_from_db": context_for_explanation
        })
        return explanation
    except Exception as e:
        print(f"Lỗi khi tạo giải thích: {e}"); return "Xin lỗi, không thể tạo giải thích."