# src/data_processing/vocabulary_parser.py
import os
import re
import pdfplumber
from typing import List, Dict, Optional

# Sử dụng lại các hàm tiện ích bạn đã có
from src.utils.text_utils import clean_text, is_hiragana, is_katakana, contains_kanji, contains_japanese_char, is_kana


class VocabularyParser:
    """
    Parser cải tiến để xử lý các file PDF từ vựng có định dạng cột linh hoạt.
    """

    def __init__(self):
        self.current_lesson = "Unknown"
        self.current_chapter = None

    def _reset_state(self):
        self.current_lesson = "Unknown"
        self.current_chapter = None

    def _normalize_japanese_parts(self, jp_tokens: List[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Từ một list các token tiếng Nhật, xác định đâu là từ vựng và đâu là cách đọc.
        Ví dụ: ['（お）名前', '（お）なまえ'] -> ('名前', 'なまえ')
               ['アメリカ'] -> ('アメリカ', 'アメリカ')
               ['わたし'] -> (None, 'わたし')
        """
        # Loại bỏ tiền tố (お) và làm sạch
        cleaned_tokens = [re.sub(r'^\(お\)|（お）', '', token).strip() for token in jp_tokens]
        cleaned_tokens = [token for token in cleaned_tokens if token]

        if not cleaned_tokens:
            return None, None

        # Nếu chỉ có 1 token
        if len(cleaned_tokens) == 1:
            token = cleaned_tokens[0]
            if is_hiragana(token) and not contains_kanji(token):
                return None, token  # Chỉ là hiragana -> là cách đọc
            else:
                return token, token  # Là Kanji hoặc Katakana -> vừa là từ, vừa là cách đọc

        # Nếu có 2 tokens
        if len(cleaned_tokens) == 2:
            part1, part2 = cleaned_tokens
            # Giả định một trong hai là cách đọc (toàn là hiragana/katakana)
            if is_kana(part2) and not contains_kanji(part2):
                return part1, part2  # Part 1 là từ, Part 2 là cách đọc
            if is_kana(part1) and not contains_kanji(part1):
                return part2, part1  # Ngược lại

        # Nếu có nhiều hơn 2 token hoặc không xác định được, ghép lại và coi là một
        full_vocab = " ".join(cleaned_tokens)
        return full_vocab, full_vocab

    def _parse_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        Phân tích một dòng để trích xuất các thành phần bằng cách tìm ranh giới
        giữa tiếng Nhật và tiếng Việt.
        """
        cleaned_line = clean_text(line)
        if not cleaned_line:
            return None

        tokens = cleaned_line.split()
        if len(tokens) < 2:
            return None

        # Tìm vị trí bắt đầu của phần nghĩa (từ đầu tiên không phải tiếng Nhật)
        split_index = -1
        for i, token in enumerate(tokens):
            if not contains_japanese_char(token) and re.search(r'[a-zA-ZÀ-ỹ]', token):
                split_index = i
                break

        if split_index == -1:
            # Không tìm thấy phần nghĩa tiếng Việt
            return None

        jp_tokens = tokens[:split_index]
        meaning_tokens = tokens[split_index:]

        if not jp_tokens or not meaning_tokens:
            return None

        # Ghép các phần lại
        meaning = " ".join(meaning_tokens)
        vocabulary, reading = self._normalize_japanese_parts(jp_tokens)

        if reading and meaning:
            return {
                "từ vựng": vocabulary,
                "cách đọc": reading,
                "nghĩa": meaning
            }

        return None

    def parse(self, file_path: str) -> List[Dict[str, any]]:
        """
        Phương thức chính để parse toàn bộ file PDF và trả về cấu trúc JSON.
        """
        self._reset_state()
        extracted_data = []

        if not os.path.exists(file_path):
            print(f"Lỗi: Không tìm thấy file tại '{file_path}'")
            return extracted_data

        print(f"Bắt đầu xử lý file: {os.path.basename(file_path)}...")
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=2, layout=True) or ""
                    for line in page_text.split('\n'):

                        # Tìm tên bài học (Lesson)
                        lesson_match = re.search(r"^(第\s*\w+\s*課)", line.strip())
                        if lesson_match:
                            self.current_lesson = lesson_match.group(1).strip()
                            continue

                        # Tìm tên chương (Chapter) - Dựa vào quy ước, ví dụ: dòng in đậm, lớn
                        # Logic này sẽ được thêm vào sau. Hiện tại chapter là None.
                        # Ví dụ: if line is bold and large -> self.current_chapter = line.strip()

                        # Parse dòng để lấy từ vựng
                        parsed_item = self._parse_line(line)
                        if parsed_item:
                            parsed_item['lesson'] = self.current_lesson
                            parsed_item['chapter'] = self.current_chapter
                            extracted_data.append(parsed_item)

            print(f"Xử lý xong. Trích xuất được {len(extracted_data)} mục từ vựng.")
        except Exception as e:
            print(f"Đã có lỗi xảy ra khi xử lý file PDF: {e}")

        return extracted_data