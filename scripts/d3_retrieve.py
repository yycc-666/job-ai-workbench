import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import csv
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/jd_chunks.jsonl"
EMB_FILE = "data/embeddings.npz"
QUERIES_FILE = "eval/queries.jsonl"
OUTPUT = "data/match_v1.csv"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def main():
    print("加载 embeddings...")
    data = np.load(EMB_FILE, allow_pickle=True)
    chunk_vectors = data["chunk_vectors"]
    chunk_ids = data["chunk_ids"].tolist()
    job_ids = data["job_ids"].tolist()

    print("加载 chunks 元信息...")
    chunk_meta = {}
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_meta[c["chunk_id"]] = c

    print("加载模型...")
    model = SentenceTransformer(MODEL_NAME)

    print("加载 queries...")
    queries = []
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"共 {len(queries)} 条 query，开始检索...")
    print(f"chunk 向量矩阵：{chunk_vectors.shape}")

    rows = []
    for q in queries:
        qid = q["qid"]
        qtext = q["query"]

        q_vec = model.encode([qtext], normalize_embeddings=True)[0]

        # 所有 chunk 的相似度
        sims = chunk_vectors @ q_vec  # (N,)

        # 按 job_id 聚合 Max
        job_scores = defaultdict(float)
        for i, jid in enumerate(job_ids):
            if sims[i] > job_scores[jid]:
                job_scores[jid] = sims[i]

        # 排序取 Top10
        ranked = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        for rank, (jid, score) in enumerate(ranked, 1):
            # 取该 job 的 meta 信息（任一块的 meta 都行）
            meta = {}
            for cid in chunk_ids:
                if cid.startswith(f"jd_{jid}_"):
                    meta = chunk_meta[cid]["meta"]
                    break

            rows.append({
                "qid": qid,
                "query": qtext,
                "rank": rank,
                "score": round(float(score), 4),
                "job_id": jid,
                "岗位名": meta.get("岗位名", ""),
                "公司": meta.get("公司", ""),
                "城市": meta.get("城市", ""),
                "url": meta.get("url", ""),
            })

    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "qid", "query", "rank", "score", "job_id",
            "岗位名", "公司", "城市", "url"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"已输出 {OUTPUT}，共 {len(rows)} 行")


if __name__ == "__main__":
    main()