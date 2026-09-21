# search.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import re
import numpy as np
import jieba
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

JD_FILE = "data/jd_llm_fixed.jsonl"
CHUNKS_FILE = "data/jd_chunks.jsonl"
EMB_FILE = "data/embeddings.npz"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

RRF_K = 60
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 1.0

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


class JobSearcher:
    def __init__(self):
        print("[search] 加载 JD 元信息...")
        self.jd_meta = {}
        self.jd_bm25_text = {}
        with open(JD_FILE, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                jid = extract_job_id(rec.get("url", ""))
                self.jd_meta[jid] = {
                    "城市": rec.get("城市", "").strip(),
                    "学历要求": rec.get("学历要求", "").strip(),
                    "岗位名": rec.get("岗位名", ""),
                    "公司": rec.get("公司", ""),
                    "url": rec.get("url", ""),
                }
                skills = rec.get("技能关键词", [])
                if isinstance(skills, list):
                    skills = "、".join(skills)
                self.jd_bm25_text[jid] = " ".join([
                    str(rec.get("岗位名", "")) * 3,
                    str(skills) * 2,
                    str(rec.get("职责原文", ""))[:1500],
                ])

        self.city_set = set(m["城市"] for m in self.jd_meta.values() if m["城市"].endswith("市"))

        print("[search] 加载 chunk 向量...")
        data = np.load(EMB_FILE, allow_pickle=True)
        self.chunk_vectors = data["chunk_vectors"]
        self.chunk_ids = data["chunk_ids"].tolist()
        self.job_ids = data["job_ids"].tolist()

        print("[search] 构建 BM25 索引...")
        self.bm25_job_list = list(self.jd_meta.keys())
        self.bm25_corpus = [tokenize(self.jd_bm25_text[jid]) for jid in self.bm25_job_list]
        self.bm25 = BM25Okapi(self.bm25_corpus)

        print("[search] 加载 BGE 模型...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("[search] 加载面试题库...")
        self.interview_qs = {}
        try:
            with open("data/interview_qs.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.interview_qs[rec["job_id"]] = rec
            print(f"[search] 题库：{len(self.interview_qs)} 条")
        except FileNotFoundError:
            print("[search] 题库文件不存在，跳过")

        print(f"[search] 就绪：{len(self.jd_meta)} 条 JD，{len(self.city_set)} 个城市")

    def _extract_city(self, query):
        for city in self.city_set:
            short = city.rstrip("市")
            if city in query or short in query:
                return city
        return None

    def _extract_edu(self, query):
        if "不要求硕士" in query or "不需要硕士" in query:
            return {"本科", "大专", "不限", ""}
        if "学历不限" in query or "不限学历" in query:
            return {"不限"}
        for kw in ("本科", "硕士", "博士", "大专"):
            if kw in query:
                return {kw, "不限", ""}
        return None

    def search(self, query, top_k=10):
        city_c = self._extract_city(query)
        edu_c = self._extract_edu(query)

        allowed = []
        for jid, meta in self.jd_meta.items():
            if city_c and meta["城市"] != city_c:
                continue
            if edu_c is not None and meta["学历要求"] not in edu_c:
                continue
            allowed.append(jid)
        if not allowed:
            allowed = list(self.jd_meta.keys())
        allowed_set = set(allowed)

        # 向量路
        keep_idx = [i for i, jid in enumerate(self.job_ids) if jid in allowed_set]
        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        sub_vecs = self.chunk_vectors[keep_idx]
        sims = sub_vecs @ q_vec

        vec_job_scores = defaultdict(float)
        for i, jid in enumerate([self.job_ids[i] for i in keep_idx]):
            if sims[i] > vec_job_scores[jid]:
                vec_job_scores[jid] = sims[i]
        vec_ranking = sorted(vec_job_scores.items(), key=lambda x: -x[1])

        # BM25 路
        q_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranking = sorted(
            [(self.bm25_job_list[i], bm25_scores[i])
             for i in range(len(self.bm25_job_list))
             if self.bm25_job_list[i] in allowed_set],
            key=lambda x: -x[1]
        )

        # RRF 融合
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

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for jid, score in ranked:
            meta = self.jd_meta[jid]
            results.append({
                "job_id": jid,
                "岗位名": meta["岗位名"],
                "公司": meta["公司"],
                "城市": meta["城市"],
                "url": meta["url"],
                "score": round(float(score), 6),
            })
        return {
            "query": query,
            "city_constraint": city_c or "",
            "edu_constraint": "、".join(sorted(edu_c)) if edu_c else "",
            "results": results,
        }

    def get_interview_questions(self, job_id):
        rec = self.interview_qs.get(job_id)
        if not rec:
            return None
        return {
            "job_id": job_id,
            "岗位名": rec.get("岗位名", ""),
            "公司": rec.get("公司", ""),
            "城市": rec.get("城市", ""),
            "url": rec.get("url", ""),
            "questions": rec.get("questions", []),
        }