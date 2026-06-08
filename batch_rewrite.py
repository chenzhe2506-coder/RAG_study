import csv
import os
import json
import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from collections import Counter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class ReviewAnalysis(BaseModel):
    summary: str = Field(description="一句话总结核心问题")
    tags: list[str] = Field(description="问题标签列表")
    severity: str = Field(description="严重程度：高/中/低")

# ========== 配置 ==========
db_path = "my_chroma_db"
cache_file = "analysis_cache.json"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

# ========== 向量库（存在则加载，不存在则创建） ==========
if os.path.exists(db_path):
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print("加载已有向量库\n")
else:
    docs = []
    with open("reviews.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append(Document(
                page_content=row["content"],
                metadata={"id": row["id"]}
            ))
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"已存入 {len(docs)} 条\n")

retriever = db.as_retriever(search_kwargs={"k": 3})

# ========== 缓存加载 ==========
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"加载缓存 {len(cache)} 条\n")
else:
    cache = {}

# ========== LLM（只创建一次） ==========
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=config.API_KEY,
    model="deepseek-v4-pro",
    temperature=0.3
)

# ========== 逐条分析 ==========
all_tags = []
all_results = []
with open("reviews.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        review_id = row["id"]
        review_content = row["content"]

        # 检查缓存
        if review_id in cache:
            print(f"[{review_id}] 命中缓存")
            result = ReviewAnalysis(**cache[review_id])
        else:
            print(f"=== 开始处理 ID: {review_id} ====")

            similar_docs = retriever.invoke(review_content)
            filtered = []
            for doc in similar_docs:
                if doc.metadata.get("id") != review_id:
                    filtered.append(doc.page_content)
                if len(filtered) >= 2:
                    break

            similar_text = "\n".join(f"- {d}" for d in filtered)
            prompt = ChatPromptTemplate.from_messages([
                ("user", """请分析以下差评，严格按 JSON 格式输出，不要有任何额外文字。
当前差评：{review_content}

历史相似差评：
{similar_text}

{format_instructions}""")
            ])
            chain = prompt | llm | parser

            result = chain.invoke({
                "review_content": review_content,
                "similar_text": similar_text,
                "format_instructions": parser.get_format_instructions()
            })
            cache[review_id] = result.model_dump()

        all_results.append({
            "id": review_id,
            "summary": result.summary,
            "tags": result.tags,
            "severity": result.severity
        })
        all_tags.extend(result.tags)
        print(f"[{review_id}] {result.summary} | 标签: {', '.join(result.tags)} | 严重: {result.severity}")

    tag_counts = Counter(all_tags)
    for tag, count in tag_counts.most_common(10):
        print(f"{tag}: {count}次")

# ========== 缓存保存 ==========
with open(cache_file, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)
print(f"\n缓存已保存，共 {len(cache)} 条")

# ========== 报告输出 ==========
with open("analysis_report.json", "w", encoding="utf-8") as f:
    json.dump({
        "total": len(all_results),
        "top_tags": tag_counts.most_common(10),
        "reviews": all_results
    }, f, ensure_ascii=False, indent=2)

with open("report.txt", "w", encoding="utf-8") as f:
    f.write("差评分析报告\n")
    f.write("=" * 30 + "\n\n")
    f.write(f"分析总数：{len(all_results)} 条\n\n")
    f.write("高频问题 TOP10：\n")
    for i, (tag, count) in enumerate(tag_counts.most_common(10), 1):
        f.write(f"{i}. {tag}: {count} 次\n")
    f.write("\n详细结果见：analysis_report.json\n")

print("\n报告已保存")
