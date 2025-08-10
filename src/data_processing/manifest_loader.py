import json
import pathlib
from functools import lru_cache
from typing import List

MANIFEST_PATH = pathlib.Path("data/manifest.json")

@lru_cache(maxsize=1)
def _load_manifest() -> List[dict]:
    """Đọc manifest.json chỉ 1 lần."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {MANIFEST_PATH}")
    with MANIFEST_PATH.open(encoding="utf-8") as fp:
        return json.load(fp)

def _main_level(level: str) -> str:
    """Trả về level chính (N5..N1) từ chuỗi có thể chứa tier, ví dụ 'N4-M' -> 'N4'."""
    lvl = (level or "").upper().strip()
    if '-' in lvl:
        lvl = lvl.split('-')[0]
    return lvl

def courses_by_level(level: str) -> List[str]:
    """Trả về list course_id (đã đúng thứ tự) cho level N5/N4/N3..."""
    level = _main_level(level)
    courses: List[str] = []
    for entry in _load_manifest():
        desc = entry["description"].upper()
        if f" {level}." in desc or f" {level}," in desc or f" {level} " in desc:
            for c in entry["courses"]:
                courses.append(c["course_id"])
    return courses

def course_sequence_between(start_level: str, end_level: str) -> List[str]:
    """Ghép list khóa học từ start_level -> end_level (inclusive). Chấp nhận tham số có tier."""
    order_levels = ["N5", "N4", "N3", "N2", "N1"]
    start_main = _main_level(start_level)
    end_main = _main_level(end_level)
    try:
        idx_start = order_levels.index(start_main)
        idx_end = order_levels.index(end_main)
    except ValueError:
        return []
    if idx_start > idx_end:
        idx_start, idx_end = idx_end, idx_start
    seq: List[str] = []
    for lv in order_levels[idx_start: idx_end + 1]:
        seq.extend(courses_by_level(lv))
    return seq
