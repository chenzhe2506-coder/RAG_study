# ========== 第1步：import（从 batch_rewrite.py 复制）==========
# TODO: 复制 batch_rewrite.py 的第1-12行 import
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

# ========== 第2步：class ReviewAnalysis（从 batch_rewrite.py 复制）==========
# TODO: 复制 batch_rewrite.py 的第14-17行 class
class ReviewAnalysis(BaseModel):
    summary : str = Field(description="一句话总结核心问题")
    tags : list[str] = Field(description="问题标签列表")
    severity : str = Field(description="严重程度: 高/中/低")

# ========== 第3步：初始化 ==========
db_path = "my_chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

# TODO: 向量库加载逻辑（从 batch_rewrite.py 第24-40行复制）
# 提示：if os.path.exists(db_path): 加载 else: 从CSV创建
if os.path.exists(db_path):
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print("加载已有向量库\n")
else:
    docs=[]
    with open("reviews.csv","r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append(Document(
                page_content=row["content"],
                metadata={"id":row["id"]}
            ))
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"已存入 {len(docs)} 条\n")
retriever = db.as_retriever(search_kwargs={"k": 3})

# TODO: LLM定义（从 batch_rewrite.py 第74-79行复制）
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=config.API_KEY,
    model="deepseek-v4-pro",
    temperature=0.3
)

# ========== 第4步：交互式循环（自己写）==========
while True:
    # TODO: input() 接收用户输入，strip() 去空格
    # TODO: 如果输入 "quit"，break 退出循环
    reviews_content=input("请输入差评(输入quit退出):").strip()
    if reviews_content.lower()=="quit":
        break
    # TODO: retriever.invoke(review_content) 检索相似评论
    similar_docs = retriever.invoke(reviews_content)
    filtered = []
    for doc in similar_docs:
        filtered.append(doc.page_content)
        if len(filtered) >= 2 :
            break
    # TODO: 取前2条（不需要排除自己，因为没有review_id）

    # TODO: 拼 prompt，调 chain.invoke()
    similar_text = "\n".join(f"- {d}" for d in filtered)
    prompt = ChatPromptTemplate.from_messages([
        ("user","""请分析以下差评，严格按 JSON 格式输出，不要有任何额外文字。
当前差评：{review_content}
         
历史相似差评：
{similar_text}
         
{format_instructions}""")
    ])
    chain = prompt | llm |parser
    result = chain.invoke({
        "review_content":reviews_content,
        "similar_text":similar_text,
        "format_instructions": parser.get_format_instructions()
    })
    print(f"summary: {result.summary}")
    print(f"tags: {result.tags}")
    print(f"severity: {result.severity}")
    # TODO: print 输出 result.summary / result.tags / result.severity

    print("-" * 40)  # 分隔线
