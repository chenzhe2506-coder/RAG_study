import csv
import os
import shutil
import config
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
# ========== 第1步：初始化 ==========
  # 加载Embedding模型
  # 创建/连接Chroma向量库（删旧建新）
model = SentenceTransformer("all-MiniLM-L6-v2")
db_path="my_chroma_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="reviews")

  # ========== 第2步：读CSV，存向量库 ==========
  # 打开reviews.csv
  # 用DictReader遍历，提取id和content
  # 所有content组成列表，encode转向量
  # collection.add()入库
docs=[]
with open("reviews.csv","r",encoding="utf-8") as f:
    for row in csv.DictReader(f):
        docs.append({
            "id":row["id"],
            "content":row["content"]
        })
texts = [d["content"] for d in docs]
vectors = model.encode(texts)
collection.add(
    ids=[d["id"] for d in docs],
    documents=[d["content"] for d in docs],
    embeddings=vectors.tolist()
)
print(f"已存入 {len(docs)} 条\n")

  # ========== 第3步：逐条分析 ==========
  # 再打开reviews.csv遍历
  # 每条差评encode转向量
  # collection.query()搜3条相似
  # 排除自己，取前2条

with open("reviews.csv","r",encoding="utf-8") as f:
    count = 0
    for row in csv.DictReader(f):
        count += 1
        if count > 10:
            break
        print(f"=== 开始处理 ID: {row['id']} ====")
        review_id=row["id"]
        review_content=row["content"]
        query_vector = model.encode([review_content])
        results = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=3
        )
        
        similar_docs = []
        for doc_id , doc_text in zip(results["ids"][0],results["documents"][0]):
            if doc_id != review_id:
                similar_docs.append(doc_text)
            if len(similar_docs) >=2:
                break

  # ========== 第4步：调LLM ==========
  # 拼prompt
        similar_text = "\n".join(f"- {d}" for d in similar_docs)
        prompt = ChatPromptTemplate.from_messages([
            ("user","""请分析以下差评，严格按 JSON 格式输出，不要有任何额外文字。
当前差评：{review_content}

历史相似差评：
{similar_text}

请按以下 JSON 格式返回：
{{
    "summary": "一句话总结核心问题",
    "tags": ["标签1", "标签2"],
    "severity": "高/中/低"
}}""")
            ]) 
        llm = ChatOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=config.API_KEY,
            model="deepseek-v4-pro",
            temperature=0.3
        )
        chain = prompt | llm | StrOutputParser()

        ai_reply = chain.invoke({
            "review_content": review_content,
            "similar_text": similar_text
        })
        
        
        print(f"[{review_id}] {ai_reply}")       
          # ========== 第5步：输出 ==========
  # print结果
  # 可选：写入文件

#if __name__ == "__main__":
  # 调用主函数
