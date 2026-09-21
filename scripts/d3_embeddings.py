import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/jd_chunks.jsonl"
RESUME_FILE = "resume.txt"
OUTPUT = "data/embeddings.npz"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def main():
    print("加载模型...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"模型加载 {time.time() - t0:.1f}s")

    # 读 chunks
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"读取 {len(chunks)} 个 chunk")

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    job_ids = [c["job_id"] for c in chunks]

    # 编码
    print("向量化 chunks...")
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"编码 {len(texts)} 块耗时 {time.time() - t0:.1f}s")

    # 简历
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        resume = f.read()
    resume_vec = model.encode([resume], normalize_embeddings=True)[0]
    print(f"简历向量：{resume_vec.shape}")

    # 存
    np.savez(
        OUTPUT,
        chunk_vectors=vectors,
        chunk_ids=np.array(chunk_ids),
        job_ids=np.array(job_ids),
        resume_vector=resume_vec,
    )
    print(f"已保存 {OUTPUT}")


if __name__ == "__main__":
    main()