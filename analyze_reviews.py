"""
第一个Prompt实验：读取差评CSV，逐条调OpenAI分析核心问题
"""
import csv
import requests
from config import OPENAI_API_KEY, MODEL


def analyze_review(review_text):
    """调用OpenAI分析单条差评，返回核心问题"""

    url = "http://154.36.185.222:8080/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""请分析以下用户评论，用一句话总结核心问题（只输出一句话，不要多余解释）：

评论：{review_text}"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[分析失败: {e}]"


def main():
    # 读取CSV
    with open("reviews.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reviews = list(reader)

    print(f"共读取 {len(reviews)} 条差评，开始分析...")
    print("-" * 50)

    # 逐条分析
    for row in reviews:
        review_id = row["id"]
        content = row["content"]
        summary = analyze_review(content)

        print(f"[{review_id}] {summary}")
        print(f"    原文: {content[:40]}...")
        print()

        # 同时写入结果文件
        with open("analysis_result.txt", "a", encoding="utf-8") as out:
            out.write(f"[{review_id}] {summary}\n")
            out.write(f"    原文: {content}\n\n")


if __name__ == "__main__":
    main()
