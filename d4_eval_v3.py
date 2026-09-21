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


def print_metrics_table(all_metrics):
    """all_metrics: [("v0 TF-IDF", m0), ("v1 BGE", m1), ...]"""
    names = [n for n, _ in all_metrics]
    header = f"{'指标':<12}" + "".join(f"{n:>14}" for n in names) + f"{'v3-v2':>10}"
    print(header)
    print("-" * len(header))
    keys = ["Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR"]
    for k in keys:
        line = f"{k:<12}"
        for _, m in all_metrics:
            line += f"{m[k]:>14.4f}"
        # 最后一个版本 - 倒数第二个版本
        if len(all_metrics) >= 2:
            delta = all_metrics[-1][1][k] - all_metrics[-2][1][k]
            line += f"{delta:>+10.4f}"
        print(line)


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

    print("加载 v1 (data/match_v1.csv)...")
    v1_results = load_match_csv("data/match_v1.csv")
    print(f"  {len(v1_results)} 条 query")

    print("加载 v2 (data/match_v2.csv)...")
    v2_results = load_match_csv("data/match_v2.csv")
    print(f"  {len(v2_results)} 条 query")

    print("加载 v3 (data/match_v3.csv)...")
    v3_results = load_match_csv("data/match_v3.csv")
    print(f"  {len(v3_results)} 条 query")

    print("\n===== 四版对比 =====")
    v0_m = evaluate(v0_results, queries)
    v1_m = evaluate(v1_results, queries)
    v2_m = evaluate(v2_results, queries)
    v3_m = evaluate(v3_results, queries)

    print_metrics_table([
        ("v0 TF-IDF", v0_m),
        ("v1 BGE", v1_m),
        ("v2 硬过滤", v2_m),
        ("v3 混合", v3_m),
    ])

    # ============ 差异分析 ============
    def diff(old_res, new_res):
        """返回 (fix: 老未命中→新命中, break: 老命中→新未命中)"""
        fix, brk = [], []
        for q in queries:
            rel = q.get("relevant_job_ids", [])
            old_hit = bool(set(old_res.get(q["qid"], [])[:10]) & set(rel))
            new_hit = bool(set(new_res.get(q["qid"], [])[:10]) & set(rel))
            if (not old_hit) and new_hit:
                fix.append(q)
            elif old_hit and (not new_hit):
                brk.append(q)
        return fix, brk

    fix_v2_v3, break_v2_v3 = diff(v2_results, v3_results)

    print(f"\n===== v2→v3 修好的 query（{len(fix_v2_v3)} 条）=====")
    for q in fix_v2_v3:
        print(f"  {q['qid']} | {q['query']}")

    print(f"\n===== v2→v3 改坏的 query（{len(break_v2_v3)} 条）=====")
    for q in break_v2_v3:
        print(f"  {q['qid']} | {q['query']}")

    # v3 仍未命中
    print("\n===== v3 仍未命中的 query =====")
    v3_miss = []
    for q in queries:
        rel = q.get("relevant_job_ids", [])
        pred = v3_results.get(q["qid"], [])
        if not (set(pred[:10]) & set(rel)):
            print(f"  {q['qid']} | {q['query']}")
            v3_miss.append(q)

    # ============ 保存报告 ============
    with open("eval/report_v3.md", "w", encoding="utf-8") as f:
        f.write("# W2 检索实验报告：v0 / v1 / v2 / v3\n\n")
        f.write("## 实验设置\n\n")
        f.write(f"- 查询数：{len(queries)}\n")
        f.write("- v0：`data/jd.jsonl`（规则版）+ TF-IDF\n")
        f.write("- v1：`data/jd_llm_fixed.jsonl`（LLM版）+ BGE-small-zh + 分块检索（Max 聚合）\n")
        f.write("- v2：v1 + 城市/学历元数据硬过滤\n")
        f.write("- v3：v2 + BM25 混合检索（RRF 融合，K=60，向量/BM25 权重各 1.0）\n\n")

        f.write("## 指标对比\n\n")
        f.write("| 指标 | v0 TF-IDF | v1 BGE | v2 硬过滤 | v3 混合 | v3-v2 |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for k in ["Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR"]:
            f.write(
                f"| {k} | {v0_m[k]:.4f} | {v1_m[k]:.4f} | "
                f"{v2_m[k]:.4f} | {v3_m[k]:.4f} | {v3_m[k] - v2_m[k]:+.4f} |\n"
            )

        f.write(f"\n## v2→v3 修好的 query（{len(fix_v2_v3)} 条）\n\n")
        for q in fix_v2_v3:
            f.write(f"- {q['qid']} | {q['query']}\n")

        f.write(f"\n## v2→v3 改坏的 query（{len(break_v2_v3)} 条）\n\n")
        for q in break_v2_v3:
            f.write(f"- {q['qid']} | {q['query']}\n")

        f.write(f"\n## v3 仍未命中：{len(v3_miss)} 条\n\n")
        for q in v3_miss:
            f.write(f"- {q['qid']} | {q['query']} | 期望: {q.get('relevant_job_ids', [])}\n")

        f.write("\n## 遗留问题\n\n")
        f.write("（根据实际结果补充）\n")

    print("\n已保存 eval/report_v3.md")


if __name__ == "__main__":
    main()