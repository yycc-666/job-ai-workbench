import csv
import json

# 1. 原CSV 按 url 建索引
rows = {}
with open("whpu_jobs_detail.csv", "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        rows[row["url"]] = row

# 2. LLM 结果按 url 建索引
llm = {}
with open("data/jd_llm_fixed.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        llm[rec["url"]] = rec

# 3. 比对表中的 30 个 url（保持出现顺序）
urls = []
with open("data/d2_compare30.csv", "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["url"] and row["url"] not in urls:
            urls.append(row["url"])

out = []
for i, url in enumerate(urls, 1):
    row = rows[url]
    rec = llm[url]
    out.append("=" * 90)
    out.append(f"[{i:02d}] {url}")
    out.append("-" * 90)
    out.append(f"title    : {row['title']}")
    out.append(f"salary   : {row['salary']}")
    out.append(f"工作地址 : {row['工作地址']}")
    out.append(f"LLM城市  : {rec.get('城市', '')}")
    out.append(f"LLM学历  : {rec.get('学历要求', '')}")
    out.append(f"LLM岗位名: {rec.get('岗位名', '')}")
    out.append(f"LLM技能  : {'、'.join(rec.get('技能关键词', []))}")
    out.append("-" * 90)
    out.append(f"【职位详情原文】\n{row['职位详情']}")
    out.append("-" * 90)
    out.append(f"【LLM职责原文】\n{rec.get('职责原文', '')}")
    out.append("")

with open("data/d2_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))

print(f"已输出 data/d2_report.txt，共 {len(urls)} 个岗位")
