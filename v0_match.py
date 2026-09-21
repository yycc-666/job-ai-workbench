import json
import csv
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = set(
    "的 了 和 与 及 或 在 是 有 为 等 能够 要求 负责 相关 进行 以及 我们 公司 岗位 职位 工作 招聘 "
    "优先 熟悉 掌握 了解 具备 能力 经验 专业 以上 不限 全职 实习 完成 参与 协助 根据 使用".split()
)


def tokenize(text):
    text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9+#.]", " ", str(text))
    words = jieba.lcut(text)
    result = []
    for w in words:
        w = w.strip().lower()
        if w and w not in STOPWORDS:
            result.append(w)
    return result


def load_jobs(path="data/jd.jsonl"):
    jobs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            skills = rec.get("技能关键词", [])
            if isinstance(skills, list):
                skills = "、".join(skills)

            # 岗位名和技能关键词重复两次，提高权重
            rec["_text"] = " ".join([
                str(rec.get("岗位名", "")),
                str(rec.get("岗位名", "")),
                str(skills),
                str(skills),
                str(rec.get("城市", "")),
                str(rec.get("学历要求", "")),
                str(rec.get("职责原文", "")),
            ])
            jobs.append(rec)
    return jobs


def main():
    with open("resume.txt", "r", encoding="utf-8") as f:
        resume = f.read()

    jobs = load_jobs()
    print(f"读取简历 {len(resume)} 字，职位 {len(jobs)} 条")

    corpus = [resume] + [j["_text"] for j in jobs]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False
    )
    X = vectorizer.fit_transform(corpus)

    # 第 0 行是简历，后面是职位
    scores = cosine_similarity(X[0:1], X[1:]).flatten()
    ranked = sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True)

    # 保存全部打分，W2 对比用
    with open("data/match_v0_all.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["相似度", "岗位名", "公司", "城市", "学历要求", "技能关键词", "url"])
        for job, score in ranked:
            skills = job.get("技能关键词", [])
            if isinstance(skills, list):
                skills = "、".join(skills)
            writer.writerow([
                round(float(score), 4),
                job.get("岗位名", ""),
                job.get("公司", ""),
                job.get("城市", ""),
                job.get("学历要求", ""),
                skills,
                job.get("url", ""),
            ])

    top10 = ranked[:10]

    with open("data/match_v0_top10.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["排名", "相似度", "岗位名", "公司", "城市", "学历要求", "技能关键词", "url"])
        for i, (job, score) in enumerate(top10, 1):
            skills = job.get("技能关键词", [])
            if isinstance(skills, list):
                skills = "、".join(skills)
            writer.writerow([
                i,
                round(float(score), 4),
                job.get("岗位名", ""),
                job.get("公司", ""),
                job.get("城市", ""),
                job.get("学历要求", ""),
                skills,
                job.get("url", ""),
            ])

    print("\nTop 10：")
    for i, (job, score) in enumerate(top10, 1):
        print(f"{i}. {score:.4f} | {job.get('岗位名','')} | {job.get('公司','')} | {job.get('城市','')}")


if __name__ == "__main__":
    main()