import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
import time

print("加载 BGE-small-zh-v1.5 ...")
t0 = time.time()
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print(f"加载耗时：{time.time() - t0:.1f} 秒")

print("向量维度：", model.get_sentence_embedding_dimension())

texts = [
    "Python后端开发工程师",
    "Java开发",
    "AI应用开发，熟悉RAG和Prompt工程",
]
t0 = time.time()
vecs = model.encode(texts, normalize_embeddings=True)
print(f"编码 3 句耗时：{time.time() - t0:.2f} 秒")
print("向量形状：", vecs.shape)

# 算相似度
import numpy as np
sim_01 = float(np.dot(vecs[0], vecs[1]))
sim_02 = float(np.dot(vecs[0], vecs[2]))
print(f"相似度(Python后端, Java开发) = {sim_01:.4f}")
print(f"相似度(Python后端, AI应用开发) = {sim_02:.4f}")

print("\n验证成功。可以进 D3。")