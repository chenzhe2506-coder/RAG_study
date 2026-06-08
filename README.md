# 差评智能分析系统

基于 RAG + LLM 的电商评论语义聚类分析工具。

## 技术栈

- **Embedding**: HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- **向量库**: Chroma（支持持久化加载）
- **LLM**: DeepSeek API (deepseek-v4-pro)
- **编排**: LangChain LCEL + Retriever
- **输出**: Pydantic 结构化解析

## 核心功能

1. 读取 CSV 差评数据，Embedding 转向量存入 Chroma
2. 逐条检索相似历史差评（Top-K 语义相似）
3. 调用 LLM 分析核心问题，输出结构化 JSON（summary + tags + severity）
4. 标签统计，识别高频问题
5. **LLM 结果缓存** — 二次运行秒出结果，节省 API 费用
6. **自动生成报告** — JSON 结构化数据 + 纯文本可读报告

## 运行

```bash
pip install -r requirements.txt
python batch_rewrite.py
```

**二次运行：** 自动命中缓存，无需重复调用 LLM API。

## 输出

### 控制台输出

```
[1] 手机摄像头进灰导致拍照模糊，客服拒绝保修 | 标签: 产品质量, 客服推诿 | 严重: 高
虚假宣传: 6次
安全隐患: 6次
...
```

### 生成文件

| 文件 | 内容 |
|---|---|
| `analysis_report.json` | 完整结构化数据（50条分析结果 + Top10标签统计） |
| `report.txt` | 人类可读文本报告 |
| `analysis_cache.json` | LLM 分析结果缓存 |
| `my_chroma_db/` | 向量数据库（二次运行自动加载） |

## 项目结构

```
├── batch_rewrite.py      # 主分析脚本
├── reviews.csv           # 差评数据
├── requirements.txt      # 依赖
├── .gitignore            # 忽略规则
└── README.md             # 本文档
```
