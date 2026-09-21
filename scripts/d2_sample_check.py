import json
import random

with open("data/jd_llm.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

random.seed(42)
sample = random.sample(records, 30)

with open("data/jd_llm_sample30.csv", "w", newline="", encoding="utf-8-sig") as f:
    import csv
    writer = csv.DictWriter(f, fieldnames=["岗位名", "公司", "城市", "学历要求", "技能关键词", "职责原文", "url"])
    writer.writeheader()
    for rec in sample:
        rec["技能关键词"] = "、".join(rec.get("技能关键词", []))
        writer.writerow({k: rec.get(k, "") for k in writer.fieldnames})

print(f"已导出 30 条样本到 data/jd_llm_sample30.csv")
