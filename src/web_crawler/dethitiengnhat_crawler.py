from datetime import datetime

import requests
from bs4 import BeautifulSoup
import psycopg2
import os
import re

# Đọc config DB từ biến môi trường hoặc sửa trực tiếp ở đây
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'apjf_db'),
    'user': os.getenv('DB_USER', 'apjf_admin'),
    'password': os.getenv('DB_PASS', 'PG16password'),
    'host': os.getenv('DB_HOST', 'apjfdb-server.postgres.database.azure.com'),
    'port': os.getenv('DB_PORT', '5432'),
}


def crawl_questions(url):
    res = requests.get(url)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')

    questions = []

    # Tìm toàn bộ div.question_list
    all_question_divs = soup.find_all('div', class_='question_list')

    # Chuẩn bị list các câu hỏi thật (bỏ "れい") và mapping AS đúng
    filtered_questions = []
    as_divs = []
    all_divs = soup.find_all(['div'])  # duyệt toàn bộ divs theo thứ tự

    skip_next_as = False
    for i, div in enumerate(all_divs):
        if div.get('class') == ['question_list']:
            if 'れい' in div.get_text():
                skip_next_as = True  # bỏ cả câu hỏi ví dụ và AS kế tiếp
            else:
                filtered_questions.append(div)
        elif div.get('id', '').startswith('AS'):
            if skip_next_as:
                skip_next_as = False  # bỏ AS này
            else:
                as_divs.append(div)

    # Kiểm tra khớp số lượng
    if len(filtered_questions) != len(as_divs):
        print("Số lượng câu hỏi và đáp án không khớp!")
        return []

    # Bắt đầu xử lý từng câu
    for q_div, a_div in zip(filtered_questions, as_divs):
        text = q_div.get_text()
        m = re.match(r'\s*(\d+)', text)
        if not m:
            continue  # đề phòng vẫn có gì đó sai

        idx = int(m.group(1))
        if idx > 19:
            continue

        question_text = ''.join(str(c) for c in q_div.contents).strip()

        correct_idx = int(a_div.get_text(strip=True))

        answer_div = q_div.find_next_sibling('div', class_=re.compile(r'answer_'))
        answers = []
        if answer_div:
            answer_labels = answer_div.find_all('label', class_='container')
            for i, label in enumerate(answer_labels, 1):
                ans_div = label.find('div', class_='answers')
                ans_text = ans_div.get_text(strip=True) if ans_div else ''
                answers.append(ans_text)

        scope = "GRAMMAR"
        if len(answers) == 4 and correct_idx is not None:
            questions.append({
                "id": idx,
                "content": question_text,
                "type": "MULTIPLE_CHOICE",
                "scope": scope,
                "answers": answers,
                "correct_idx": correct_idx
            })

    return questions



def insert_to_db(questions):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    ID_PREFIX = "N3-Test1-G"
    created_at = datetime.now()
    for q in questions:
        # Chèn thêm explain='' và file_url=None (hoặc '' nếu không cho NULL)
        question_id = f"{ID_PREFIX}-{q['id']}"
        cur.execute(
            "INSERT INTO question (id, content, created_at, explanation, file_url, scope, type) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)  ON CONFLICT (id) DO NOTHING",
            (question_id, q["content"], created_at, '', None, q["scope"], q["type"])
        )
        # question_id = cur.fetchone()[0]
        # Thêm vào bảng option
        for i, ans in enumerate(q["answers"], 1):
            option_id = f"{question_id}-{i}"
            is_correct = (i == q["correct_idx"])
            cur.execute(
                "INSERT INTO option (id,question_id, content, is_correct) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (option_id, question_id, ans, is_correct)
            )
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    url = "https://dethitiengnhat.com/de_thi_thu/N3/2/3"
    questions = crawl_questions(url)
    insert_to_db(questions)
    print(f"Đã crawl và insert xong {len(questions)} câu hỏi đầu vào DB.")