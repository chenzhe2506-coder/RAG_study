# 分析器

LLM应用项目，目标：用AI自动分析商品差评，聚类归纳问题类型。

## 项目结构

```
差评聚类分析器/
├── requirements.txt      # 依赖包
├── config.py            # 配置（API Key、模型名）
├── test_openai.py       # 最小可运行脚本（测试API连通性）
└── README.md            # 本文件
```

## 当前进度

- [x] 项目骨架搭建
- [x] 最小可运行脚本（裸requests调OpenAI）
- [ ] 批量读取差评数据（Excel/CSV）
- [ ] 调用Embedding接口把差评转向量
- [ ] 用聚类算法把相似差评分组
- [ ] 用LLM给每组生成问题标签和总结

## 运行前准备

1. 在 `config.py` 里填写你的 OpenAI API Key
2. 安装依赖：`pip install -r requirements.txt`
3. 运行测试：`python test_openai.py`

## 时间线

- 项目截止：2026-06-09
