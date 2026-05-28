"""
RAG最小闭环：差评存入向量库 + 相似度检索
"""
import csv
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ========== 1. 初始化组件 ==========

# Embedding模型：本地小模型，把文字变成向量
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # 本地模型，80MB，首次运行自动下载
)

# LLM模型：分析用
llm = ChatOpenAI(
    base_url="http://154.36.185.222:8080/v1",
    api_key="sk-846ed0441dacafb720d497bdfe5888965840f345f49a6fc5867b7523c3b90676",
    model="gpt5.4"
)

# Chroma向量库：存向量+检索（数据存在 ./chroma_db 文件夹）
db = Chroma(
    embedding_function=embedding,
    persist_directory="chroma_db"
)

# ========== 2. 读取CSV，存入向量库 ==========

docs = []
with open("reviews.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        docs.append(Document(
            page_content=row["content"],  # 要转向量的文本
            metadata={"id": row["id"]}    # 额外信息（不参与向量计算）
        ))

# 存入Chroma（如果已有同名数据会先清空）
db.add_documents(docs)
print(f"已存入 {len(docs)} 条差评到向量库")

# ========== 3. 用新差评查询最相似的2条历史记录 ==========

new_review = "这个产品用了三天就坏了，客服也不理人"
results = db.similarity_search(new_review, k=2)

print(f"\n新差评：{new_review}")
print(f"找到 {len(results)} 条最相似的历史差评：")
for i, doc in enumerate(results, 1):
    print(f"  [{i}] ID={doc.metadata['id']}: {doc.page_content[:30]}...")

# ========== 4. 把相似差评+新差评一起给LLM分析 ==========

# 拼接相似差评文本
similar_text = "\n".join([
    f"- [{doc.metadata['id']}] {doc.page_content}"
    for doc in results
])

prompt = ChatPromptTemplate.from_messages([
    ("user", """请分析以下新差评的核心问题，并参考历史相似差评给出归类标签。

新差评：{new_review}

历史相似差评：
{similar_reviews}

请用一句话总结核心问题，并给出1-2个标签。""")
])

chain = prompt | llm
response = chain.invoke({
    "new_review": new_review,
    "similar_reviews": similar_text
})

print(f"\nAI分析结果：{response.content}")
