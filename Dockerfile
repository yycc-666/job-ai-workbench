# syntax=docker/dockerfile:1

FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com

# 换国内 apt 源，只装 curl（不装 build-essential，除非确实需要编译）
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip 也走国内源
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 第一步：装 CPU 版 torch
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 第二步：装其他依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 第三步：预下载 BGE 模型
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 第四步：拷代码和数据
COPY app.py search.py ./
# resume.txt 是个人简历，不 COPY 进镜像（search.py 并不读取它）
COPY data/ ./data/
COPY eval/ ./eval/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]