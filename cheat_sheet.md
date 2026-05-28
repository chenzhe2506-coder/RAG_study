# RAG 速查表

## 裸API版

### 1. Embedding
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(["文本1", "文本2"]).tolist()  # 别忘了 .tolist()
```

### 2. Chroma
```python
import chromadb
client = chromadb.PersistentClient(path="db路径")
collection = client.get_or_create_collection(name="表名")

# 存
collection.add(ids=[...], documents=[...], embeddings=[...])

# 查
results = collection.query(query_embeddings=[...], n_results=3)
results["documents"][0]  # 查到的原文列表
```

### 3. LLM (requests)
```python
import requests
resp = requests.post(
    url="http://...",
    headers={"Authorization": "Bearer xxx", "Content-Type": "application/json"},
    json={"model": "xxx", "messages": [{"role": "user", "content": prompt}]}
).json()
resp["choices"][0]["message"]["content"]
```

## 易错点
- encode() 参数必须是列表 `["文本"]`
- numpy数组要 .tolist() 才能给Chroma
- Chroma查询返回 n_results，LangChain返回 k
- requests.post(...).json() 别漏括号
