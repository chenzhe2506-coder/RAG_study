import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

# DeepSeek API
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-pro"

# Embedding 模型
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 向量库
DB_PATH = "my_chroma_db"
COLLECTION_NAME = "reviews"

# 价格（deepseek-v4-pro 折扣价，美元->人民币，汇率7.2）
INPUT_PRICE_PER_1M = 0.435 * 7.2   # 元/百万token（缓存未命中）
OUTPUT_PRICE_PER_1M = 0.87 * 7.2   # 元/百万token
