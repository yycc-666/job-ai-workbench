import csv
import json
import re
import numpy as np
import jieba
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = set(
    "的 了 和 与 及 或 在 是 有 为 等 能够 要求 负责 相关 进行 以及 我们 公司 岗位 职位 工作 招聘 "
    "优先 熟悉 掌握 了解 具备 能力 经验 专业 以上 不限 全职 实习 完成 参与 协助 根据 使用".split()
)


def extract_job_id(url):
    m = re.search(r"/id/(\d+)", url or "")
    return m.group(1) if m else ""


def load_queries(path="eval/queries.jsonl"):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


# ============ v0: TF-IDF 检索 ============
def tokenize(text):
    text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9+#.]", " ", str(text))
    words = jieba.lcut(text)
    return [w.strip().lower() for w in words if w.strip() and w.strip().lower() not in STOPWORDS]


def build_v0_corpus(path="data/jd.jsonl"):
    jobs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            skills = rec.get("技能关键词", [])
            if isinstance(skills, list):
                skills = "、".join(skills)
            rec["_text"] = " ".join([
                str(rec.get("岗位名", "")), str(rec.get("岗位名", "")),
                str(skills), str(skills),
                str(rec.get("城市", "")),
                str(rec.get("学历要求", "")),
                str(rec.get("职责原文", "")),
            ])
            rec["_job_id"] = extract_job_id(rec.get("url", ""))
            jobs.append(rec)
    return jobs


def run_v0(queries, jobs, topk=10):
    corpus = [j["_text"] for j in jobs]
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    X = vectorizer.fit_transform(corpus)

    results = {}
    for q in queries:
        qid = q["qid"]
        qtext = q["query"]
        q_vec = vectorizer.transform([qtext])
        sims = cosine_similarity(q_vec, X).flatten()
        ranked = np.argsort(-sims)[:topk]
        results[qid] = [jobs[i]["_job_id"] for i in ranked]
    return results


# ============ 通用：读 match_*.csv ============
def load_match_csv(path):
    results = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["qid"]].append(row["job_id"])
    return results


# ============ 指标 ============
def hit_at_k(predicted, relevant, k):
    return 1 if set(predicted[:k]) & set(relevant) else 0


def recall_at_k(predicted, relevant, k):
    if not relevant:
        return 0.0
    return len(set(predicted[:k]) & set(relevant)) / len(set(relevant))


def mrr(predicted, relevant):
    rel_set = set(relevant)
    for i, pid in enumerate(predicted, 1):
        if pid in rel_set:
            return 1.0 / i
    return 0.0


def evaluate(predicted_map, queries):
    metrics = {}
    for k in (5, 10):
        hit_sum = 0
        rec_sum = 0
        for q in queries:
            rel = q.get("relevant_job_ids", [])
            pred = predicted_map.get(q["qid"], [])
            hit_sum += hit_at_k(pred, rel, k)
            rec_sum += recall_at_k(pred, rel, k)
        metrics[f"Hit@{k}"] = hit_sum / len(queries)
        metrics[f"Recall@{k}"] = rec_sum / len(queries)

    mrr_sum = sum(
        mrr(predicted_map.get(q["qid"], []), q.get("relevant_job_ids", []))
        for q in queries
    )
    metrics["MRR"] = mrr_sum / len(queries)
    return metrics


def print_metrics_table(v0_m, v1_m, v2_m):
    header = f"{'指标':<12} {'v0 TF-IDF':>12} {'v1 BGE':>12} {'v2 硬过滤':>12} {'v2-v1':>10}"
    print(header)
    print("-" * len(header))
    for k in ["Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR"]:
        a, b, c = v0_m[k], v1_m[k], v2_m[k]
        print(f"{k:<12} {a:>12.4f} {b:>12.4f} {c:>12.4f} {c - b:>+10.4f}")


# ============ 主程序 ============
def main():
    print("加载评测集...")
    queries = load_queries()
    print(f"  {len(queries)} 条 query")

    print("加载 v0 corpus (data/jd.jsonl)...")
    jobs = build_v0_corpus()
    print(f"  {len(jobs)} 条 JD")

    print("跑 v0 (TF-IDF)...")
    v0_results = run_v0(queries, jobs, topk=10)

    print("加载 v1 结果 (data/match_v1.csv)...")
    v1_results = load_match_csv("data/match_v1.csv")
    print(f"  {len(v1_results)} 条 query")

    print("加载 v2 结果 (data/match_v2.csv)...")
    v2_results = load_match_csv("data/match_v2.csv")
    print(f"  {len(v2_results)} 条 query")

    print("\n===== 三版对比 =====")
    v0_metrics = evaluate(v0_results, queries)
    v1_metrics = evaluate(v1_results, queries)
    v2_metrics = evaluate(v2_results, queries)
    print_metrics_table(v0_metrics, v1_metrics, v2_metrics)

    # v2 未命中的 query
    print("\n===== v2 未命中的 query（Top10 里 0 条相关）=====")
    v2_miss = []
    for q in queries:
        rel = q.get("relevant_job_ids", [])
        pred = v2_results.get(q["qid"], [])
        if not (set(pred[:10]) & set(rel)):
            print(f"  {q['qid']} | {q['query']} | 期望: {rel}")
            v2_miss.append(q)

    # v1 vs v2 的差异：哪些 qid 在 v2 里修好了
    print("\n===== v1 未命中、v2 命中的 query（修好了）=====")
    fixed = []
    for q in queries:
        rel = q.get("relevant_job_ids", [])
        v1_pred = v1_results.get(q["qid"], [])
        v2_pred = v2_results.get(q["qid"], [])
        v1_hit = bool(set(v1_pred[:10]) & set(rel))
        v2_hit = bool(set(v2_pred[:10]) & set(rel))
        if (not v1_hit) and v2_hit:
            print(f"  {q['qid']} | {q['query']}")
            fixed.append(q)

    print("\n===== v1 命中、v2 未命中的 query（改坏了）=====")
    broken = []
    for q in queries:
        rel = q.get("relevant_job_ids", [])
        v1_pred = v1_results.get(q["qid"], [])
        v2_pred = v2_results.get(q["qid"], [])
        v1_hit = bool(set(v1_pred[:10]) & set(rel))
        v2_hit = bool(set(v2_pred[:10]) & set(rel))
        if v1_hit and (not v2_hit):
            print(f"  {q['qid']} | {q['query']}")
            broken.append(q)

    # 保存报告
    with open("eval/report_v2.md", "w", encoding="utf-8") as f:
        f.write("# W2 检索实验报告：v0 / v1 / v2\n\n")
        f.write("## 实验设置\n\n")
        f.write(f"- 查询数：{len(queries)}\n")
        f.write("- v0：`data/jd.jsonl`（规则版）+ TF-IDF\n")
        f.write("- v1：`data/jd_llm_fixed.jsonl`（LLM版）+ BGE-small-zh + 分块检索\n")
        f.write("- v2：v1 + 城市/学历元数据硬过滤\n\n")

        f.write("## 指标对比\n\n")
        f.write("| 指标 | v0 TF-IDF | v1 BGE | v2 硬过滤 | v2-v1 |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for k in ["Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR"]:
            f.write(f"| {k} | {v0_metrics[k]:.4f} | {v1_metrics[k]:.4f} | {v2_metrics[k]:.4f} | {v2_metrics[k] - v1_metrics[k]:+.4f} |\n")

        f.write(f"\n## v2 修好的 query（v1 未命中、v2 命中）：{len(fixed)} 条\n\n")
        for q in fixed:
            f.write(f"- {q['qid']} | {q['query']}\n")

        f.write(f"\n## v2 改坏的 query（v1 命中、v2 未命中）：{len(broken)} 条\n\n")
        for q in broken:
            f.write(f"- {q['qid']} | {q['query']}\n")

        f.write(f"\n## v2 仍未命中：{len(v2_miss)} 条\n\n")
        for q in v2_miss:
            f.write(f"- {q['qid']} | {q['query']} | 期望: {q.get('relevant_job_ids', [])}\n")

        f.write("\n## 遗留问题\n\n")
        f.write("1. 排序仍不准：q004 广州产品经理，Top10 全广州但产品经理没进\n")
        f.write("2. 关键词被淹没：q050 硬件开发，Top10 里硬件只排第 1\n")
        f.write("3. 公司性质未过滤：q001 国企技术岗，私企仍在榜\n")
        f.write("→ 这些都是 D4 第二步 BM25 混合检索要解决的\n")

    print("\n已保存 eval/report_v2.md")


if __name__ == "__main__":
    main()