# 差评智能分析系统

基于 RAG + LLM 的电商评论语义聚类分析工具。

## 技术栈

- **Embedding**: SentenceTransformer (all-MiniLM-L6-v2)
- **向量库**: Chroma
- **LLM**: DeepSeek API (deepseek-v4-pro)
- **编排**: LangChain LCEL

## 核心功能

1. 读取 CSV 差评数据，Embedding 转向量存入 Chroma
2. 逐条检索相似历史差评（Top-K 语义相似）
3. 调用 LLM 分析核心问题，输出结构化 JSON（summary + tags + severity）
4. 标签统计，识别高频问题

## 运行

```bash
pip install -r requirements.txt
python batch_rewrite.py
```

## 输出示例

```
[1] {"summary": "手机摄像头三天进灰，客服拒绝保修", "tags": ["产品质量", "客服推诿", "保修纠纷"], "severity": "高"}
[2] {"summary": "物流严重延迟且包装破损导致商品损坏", "tags": ["物流慢", "包装破损", "商品损坏"], "severity": "高"}
```

## 项目结构

```
├── batch_rewrite.py    # 主分析脚本
├── reviews.csv         # 差评数据
├── requirements.txt    # 依赖
└── README.md           # 本文档
```
