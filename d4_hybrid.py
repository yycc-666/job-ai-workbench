import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import csv
import re
import numpy as np
import jieba
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

JD_FILE = "data/jd_llm_fixed.jsonl"
CHUNKS_FILE = "data/jd_chunks.jsonl"
EMB_FILE = "data/embeddings.npz"
QUERIES_FILE = "eval/queries.jsonl"
OUTPUT = "data/match_v3.csv"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

RRF_K = 60          # RRF 常数
VECTOR_WEIGHT = 1.0 # 向量路权重
BM25_WEIGHT = 1.0   # BM25 路权重


STOPWORDS = set(
    "的 了 和 与 及 或 在 是 有 为 等 能够 要求 负责 相关 进行 以及 我们 公司 岗位 职位 工作 招聘 "
    "优先 熟悉 掌握 了解 具备 能力 经验 专业 以上 不限 全职 实习 完成 参与 协助 根据 使用".split()
)


def tokenize(text):
    text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9+#.]", " ", str(text))
    words = jieba.lcut(text)
    return [w.strip().lower() for w in words if w.strip() and w.strip().lower() not in STOPWORDS]


def extract_job_id(url):
    m = re.search(r"/id/(\d+)", url or "")
    return m.group(1) if m else ""


# ========== 城市/学历过滤（沿用 d4_filter.py 逻辑） ==========
def load_all_cities(jd_file):
    cities = set()
    with open(jd_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            c = rec.get("城市", "").strip()
            if c and c.endswith("市"):
                cities.add(c)
    return cities


def extract_city_constraint(query, city_set):
    for city in city_set:
        short = city.rstrip("市")
        if city in query or short in query:
            return city
    return None


EDU_KEEP = {
    "本科": {"本科", "不限", ""},
    "硕士": {"硕士", "不限", ""},
    "博士": {"博士", "不限", ""},
    "大专": {"大专", "不限", ""},
    "不要求硕士": {"本科", "大专", "不限", ""},
}


def extract_edu_constraint(query):
    if "不要求硕士" in query or "不需要硕士" in query:
        return EDU_KEEP["不要求硕士"]
    if "学历不限" in query or "不限学历" in query:
        return {"不限"}
    for kw in ("本科", "硕士", "博士", "大专"):
        if kw in query:
            return EDU_KEEP[kw]
    return None


# ========== 主程序 ==========
def main():
    print("加载数据...")

    # JD 元信息（过滤 + BM25 corpus 用）
    jd_meta = {}
    jd_bm25_text = {}
    with open(JD_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            jid = extract_job_id(rec.get("url", ""))
            jd_meta[jid] = {
                "城市": rec.get("城市", "").strip(),
                "学历要求": rec.get("学历要求", "").strip(),
                "岗位名": rec.get("岗位名", ""),
                "公司": rec.get("公司", ""),
                "url": rec.get("url", ""),
            }
            skills = rec.get("技能关键词", [])
            if isinstance(skills, list):
                skills = "、".join(skills)
            # BM25 语料：岗位名重复3次增强权重 + 技能 + 职责
            jd_bm25_text[jid] = " ".join([
                str(rec.get("岗位名", "")) * 3,
                str(skills) * 2,
                str(rec.get("职责原文", ""))[:1500],
            ])

    city_set = set(m["城市"] for m in jd_meta.values() if m["城市"].endswith("市"))
    print(f"  {len(jd_meta)} 条 JD，{len(city_set)} 个城市")

    # chunk 向量
    data = np.load(EMB_FILE, allow_pickle=True)
    chunk_vectors = data["chunk_vectors"]
    chunk_ids = data["chunk_ids"].tolist()
    job_ids = data["job_ids"].tolist()

    # BM25 corpus（用 job 级全文，不切块）
    bm25_job_list = list(jd_meta.keys())
    bm25_corpus = [tokenize(jd_bm25_text[jid]) for jid in bm25_job_list]
    print("  构建 BM25 索引...")
    bm25 = BM25Okapi(bm25_corpus)

    # 模型
    print("加载模型...")
    model = SentenceTransformer(MODEL_NAME)

    # queries
    queries = []
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"处理 {len(queries)} 条 query...")

    rows = []
    stats = {"both": 0, "city": 0, "edu": 0, "none": 0}

    for q in queries:
        qid, qtext = q["qid"], q["query"]

        # 抽约束
        city_c = extract_city_constraint(qtext, city_set)
        edu_c = extract_edu_constraint(qtext)

        # 过滤出允许的 job 子集
        allowed = []
        for jid, meta in jd_meta.items():
            if city_c and meta["城市"] != city_c:
                continue
            if edu_c is not None and meta["学历要求"] not in edu_c:
                continue
            allowed.append(jid)

        if not allowed:
            allowed = list(jd_meta.keys())

        if city_c and edu_c:
            stats["both"] += 1
        elif city_c:
            stats["city"] += 1
        elif edu_c:
            stats["edu"] += 1
        else:
            stats["none"] += 1

        allowed_set = set(allowed)

        # ---- 向量路：chunk 级 → job 级 Max ----
        keep_idx = [i for i, jid in enumerate(job_ids) if jid in allowed_set]
        q_vec = model.encode([qtext], normalize_embeddings=True)[0]
        sub_vecs = chunk_vectors[keep_idx]
        sims = sub_vecs @ q_vec

        vec_job_scores = defaultdict(float)
        for i, jid in enumerate([job_ids[i] for i in keep_idx]):
            if sims[i] > vec_job_scores[jid]:
                vec_job_scores[jid] = sims[i]

        vec_ranking = sorted(vec_job_scores.items(), key=lambda x: -x[1])

        # ---- BM25 路：job 级 ----
        q_tokens = tokenize(qtext)
        bm25_scores = bm25.get_scores(q_tokens)

        bm25_ranking = sorted(
            [(bm25_job_list[i], bm25_scores[i]) for i in range(len(bm25_job_list)) if bm25_job_list[i] in allowed_set],
            key=lambda x: -x[1]
        )

        # ---- RRF 融合 ----
        vec_rank = {jid: i + 1 for i, (jid, _) in enumerate(vec_ranking)}
        bm25_rank = {jid: i + 1 for i, (jid, _) in enumerate(bm25_ranking)}

        rrf_scores = {}
        for jid in allowed_set:
            s = 0.0
            if jid in vec_rank:
                s += VECTOR_WEIGHT / (RRF_K + vec_rank[jid])
            if jid in bm25_rank:
                s += BM25_WEIGHT / (RRF_K + bm25_rank[jid])
            rrf_scores[jid] = s

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:10]

        for rank, (jid, score) in enumerate(ranked, 1):
            meta = jd_meta[jid]
            rows.append({
                "qid": qid, "query": qtext, "rank": rank,
                "score": round(float(score), 6), "job_id": jid,
                "岗位名": meta["岗位名"], "公司": meta["公司"],
                "城市": meta["城市"], "url": meta["url"],
                "city_constraint": city_c or "",
                "edu_constraint": "、".join(sorted(edu_c)) if edu_c else "",
            })

    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "qid", "query", "rank", "score", "job_id",
            "岗位名", "公司", "城市", "url",
            "city_constraint", "edu_constraint"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"已输出 {OUTPUT}")
    print(f"约束命中统计：{stats}")


if __name__ == "__main__":
    main()