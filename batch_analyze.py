import csv
import os
import shutil
import requests
from sentence_transformers import SentenceTransformer
import chromadb

import config


def init_vector_store(csv_path):
    """初始化 Embedding 模型，读取 CSV，存入 Chroma 向量库。"""
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    if os.path.exists(config.DB_PATH):
        shutil.rmtree(config.DB_PATH)
    client = chromadb.PersistentClient(path=config.DB_PATH)
    collection = client.get_or_create_collection(name=config.COLLECTION_NAME)

    docs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append({"id": row["id"], "content": row["content"]})

    texts = [d["content"] for d in docs]
    vectors = embedding_model.encode(texts)
    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["content"] for d in docs],
        embeddings=vectors.tolist()
    )
    print(f"已存入 {len(docs)} 条\n")

    return embedding_model, collection


def get_similar_reviews(embedding_model, collection, review_content, review_id, n_results=3, top_k=2):
    """检索与当前差评最相似的历史记录，排除自己。"""
    query_vector = embedding_model.encode([review_content])

    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=n_results
    )

    similar_docs = []
    for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
        if doc_id != review_id:
            similar_docs.append(doc_text)
        if len(similar_docs) >= top_k:
            break

    return similar_docs


def analyze_with_llm(review_content, similar_docs):
    """调用 LLM 分析差评，返回 (回复文本, 本条费用元, token信息字典)。"""
    similar_text = "\n".join([f"- {d}" for d in similar_docs])
    prompt = f"""请分析以下差评的核心问题并给出标签。

当前差评：{review_content}

历史相似差评：
{similar_text}

请用一句话总结，并给出1-2个标签。"""

    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": config.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        response = requests.post(config.BASE_URL, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        ai_reply = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        item_cost = (
            prompt_tokens * config.INPUT_PRICE_PER_1M / 1_000_000 +
            completion_tokens * config.OUTPUT_PRICE_PER_1M / 1_000_000
        )

        token_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

        return ai_reply, item_cost, token_info

    except Exception as e:
        return f"分析失败：{str(e)}", 0, {}


def main():
    """主流程：初始化 -> 逐条分析 -> 输出报告。"""
    csv_path = "reviews.csv"
    report_path = "analysis_report.txt"

    embedding_model, collection = init_vector_store(csv_path)
    total_cost = 0

    with open(report_path, "w", encoding="utf-8") as out:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                review_id = row["id"]
                review_content = row["content"]

                similar_docs = get_similar_reviews(
                    embedding_model, collection, review_content, review_id
                )
                ai_reply, item_cost, token_info = analyze_with_llm(
                    review_content, similar_docs
                )
                total_cost += item_cost

                out.write(f"[{review_id}] {ai_reply}\n\n")
                print(f"[{review_id}] {ai_reply[:30]}...")

                if token_info:
                    print(
                        f"  Token: 输入{token_info['prompt_tokens']} + "
                        f"输出{token_info['completion_tokens']} = "
                        f"{token_info['total_tokens']}, "
                        f"费用: {item_cost:.6f}元"
                    )

    print(f"\n全部完成，总费用: {total_cost:.6f}元，结果在 {report_path}")


if __name__ == "__main__":
    main()
