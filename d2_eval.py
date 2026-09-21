import csv
import json
import random

# 读LLM结果
with open("data/jd_llm_fixed.jsonl", "r", encoding="utf-8") as f:
    llm_records = {}
    for line in f:
        rec = json.loads(line)
        llm_records[rec["url"]] = rec

# 读原始CSV
rows = []
with open("whpu_jobs_detail.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

random.seed(42)
sample = random.sample(rows, 30)

# 输出对照表
with open("data/d2_compare30.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "字段", "原CSV", "LLM结果", "判定(对/错)"])
    for row in sample:
        url = row["url"]
        llm = llm_records.get(url, {})
        writer.writerow([url, "城市", row.get("工作地址", "")[:30], llm.get("城市", ""), ""])
        writer.writerow([url, "学历", row.get("salary", "")[-10:], llm.get("学历要求", ""), ""])
        writer.writerow([url, "岗位名", row.get("title", ""), llm.get("岗位名", ""), ""])
        writer.writerow([url, "技能", "", "、".join(llm.get("技能关键词", [])), ""])
        writer.writerow([url, "职责", row.get("职位详情", "")[:50], llm.get("职责原文", "")[:50], ""])
        writer.writerow([])

print("已输出 data/d2_compare30.csv，人工填最后一列")