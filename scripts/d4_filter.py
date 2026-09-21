import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import csv
import re
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

JD_FILE = "data/jd_llm_fixed.jsonl"
CHUNKS_FILE = "data/jd_chunks.jsonl"
EMB_FILE = "data/embeddings.npz"
QUERIES_FILE = "eval/queries.jsonl"
OUTPUT = "data/match_v2.csv"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def extract_job_id(url):
    m = re.search(r"/id/(\d+)", url or "")
    return m.group(1) if m else ""


def load_all_cities(jd_file):
    """从JD数据里提取所有城市，作为匹配词典"""
    cities = set()
    with open(jd_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            c = rec.get("城市", "").strip()
            if c and c.endswith("市"):
                cities.add(c)
    return cities


def extract_city_constraint(query, city_set):
    """从query里识别城市约束"""
    for city in city_set:
        # 匹配"武汉市"或"武汉"（去掉"市"）
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
    "不限学历": None,  # None 表示不过滤
}


def extract_edu_constraint(query):
    # 先处理复合词
    if "不要求硕士" in query or "不需要硕士" in query:
        return EDU_KEEP["不要求硕士"]
    # ✅ 修正：学历不限 = 只保留 不限 的岗位
    if "学历不限" in query or "不限学历" in query:
        return {"不限"}
    for kw in ("本科", "硕士", "博士", "大专"):
        if kw in query:
            return EDU_KEEP[kw]
    return None


def main():
    print("加载数据...")
    # 城市列表
    city_set = load_all_cities(JD_FILE)
    print(f"  识别出 {len(city_set)} 个城市：{sorted(city_set)[:10]}...")

    # JD元信息（用于过滤）
    jd_meta = {}
    with open(JD_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            jid = extract_job_id(rec.get("url", ""))
            jd_meta[jid] = {
                "城市": rec.get("城市", "").strip(),
                "学历要求": rec.get("学历要求", "").strip(),
            }

    # chunk向量
    data = np.load(EMB_FILE, allow_pickle=True)
    chunk_vectors = data["chunk_vectors"]
    chunk_ids = data["chunk_ids"].tolist()
    job_ids = data["job_ids"].tolist()

    # chunk 元信息
    chunk_meta = {}
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_meta[c["chunk_id"]] = c

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
    stats = {"city_filtered": 0, "edu_filtered": 0, "both": 0, "none": 0}

    for q in queries:
        qid, qtext = q["qid"], q["query"]

        # 抽约束
        city_c = extract_city_constraint(qtext, city_set)
        edu_c = extract_edu_constraint(qtext)

        # 决定哪些 job 通过过滤
        allowed_jids = set()
        for jid, meta in jd_meta.items():
            if city_c and meta["城市"] != city_c:
                continue
            if edu_c is not None and meta["学历要求"] not in edu_c:
                continue
            allowed_jids.add(jid)

        if city_c and edu_c:
            stats["both"] += 1
        elif city_c:
            stats["city_filtered"] += 1
        elif edu_c:
            stats["edu_filtered"] += 1
        else:
            stats["none"] += 1

        # 过滤向量子集
        keep_idx = [i for i, jid in enumerate(job_ids) if jid in allowed_jids]

        if not keep_idx:
            # 过滤后为空，不过滤退回全量（防止评测集标注问题导致空结果）
            keep_idx = list(range(len(job_ids)))

        sub_vectors = chunk_vectors[keep_idx]
        sub_job_ids = [job_ids[i] for i in keep_idx]

        # 向量化query
        q_vec = model.encode([qtext], normalize_embeddings=True)[0]

        # 相似度
        sims = sub_vectors @ q_vec

        # 按job_id聚合 Max
        job_scores = defaultdict(float)
        for i, jid in enumerate(sub_job_ids):
            if sims[i] > job_scores[jid]:
                job_scores[jid] = sims[i]

        ranked = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        for rank, (jid, score) in enumerate(ranked, 1):
            meta = jd_meta.get(jid, {})
            # 从chunk meta里拿岗位名
            name, comp, city, url = "", "", "", ""
            for cid in chunk_ids:
                if cid.startswith(f"jd_{jid}_"):
                    m = chunk_meta[cid]["meta"]
                    name, comp, city, url = m.get("岗位名", ""), m.get("公司", ""), m.get("城市", ""), m.get("url", "")
                    break
            rows.append({
                "qid": qid, "query": qtext, "rank": rank,
                "score": round(float(score), 4), "job_id": jid,
                "岗位名": name, "公司": comp, "城市": city, "url": url,
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