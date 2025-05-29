import re

def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    cleaned = cleaned.replace('\n', ' ')
    noise_literals = [
        "$=5=5$",
        "$\overline{\tau}=\lambda$",
        "$A!/\dot{A}!$",
        "$7\\times6$ ます"
    ]
    for item in noise_literals:
        cleaned = cleaned.replace(item, "")
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned if cleaned else None

def is_hiragana(s: str | None) -> bool:
    if not s: return False
    return all('\u3040' <= char <= '\u309F' or char == 'ー' or char == '～' for char in s)

def is_katakana(s: str | None) -> bool:
    if not s: return False
    return all('\u30A0' <= char <= '\u30FF' or char == 'ー' or char == '～' for char in s)

def is_kana(s: str | None) -> bool:
    if not s: return False
    return all('\u3040' <= char <= '\u309F' or \
               '\u30A0' <= char <= '\u30FF' or \
               char in 'ーゝゞ々～' for char in s)

def contains_kanji(s: str | None) -> bool:
    if not s: return False
    return any('\u4E00' <= char <= '\u9FFF' or \
               '\u3400' <= char <= '\u4DBF' or \
               '\uF900' <= char <= '\uFAFF' for char in s)

def contains_japanese_char(s: str | None) -> bool:
    if not s: return False
    return contains_kanji(s) or is_kana(s)

def contains_latin_chars(s: str | None) -> bool:
    if not s: return False
    return bool(re.search(r'[a-zA-Z]', s))