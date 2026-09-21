import csv
import json
import re

# 1. 从原CSV读salary，抽学历
edu_map = {}
with open("whpu_jobs_detail.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        url = row.get("url", "")
        salary = row.get("salary", "")
        # salary 形如 "10500-14999广东省东莞市全职本科"
        m = re.search(r"(博士|硕士|本科|大专|中专|不限)", salary)
        if m:
            edu_map[url] = m.group(1)

print(f"从CSV抽到 {len(edu_map)} 条学历标签")

# 2. 读LLM结果，补学历
fixed = []
fix_count = 0
with open("data/jd_llm.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        url = rec.get("url", "")
        old_edu = rec.get("学历要求", "")
        if url in edu_map:
            new_edu = edu_map[url]
            if old_edu != new_edu:
                rec["学历要求"] = new_edu
                rec["学历来源"] = "salary标签补齐"
                fix_count += 1
        fixed.append(rec)

# 3. 写回
with open("data/jd_llm_fixed.jsonl", "w", encoding="utf-8") as f:
    for rec in fixed:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"补齐 {fix_count} 条学历，输出 data/jd_llm_fixed.jsonl")