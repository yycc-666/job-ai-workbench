import json
import csv
import random

with open("data/jd.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

for rec in records:
    rec["技能关键词"] = "、".join(rec["技能关键词"])

random.seed(42)
sample = random.sample(records, 20)

fields = ["岗位名", "公司", "城市", "学历要求", "技能关键词", "职责原文", "url"]
with open("data/sample_20.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(sample)

print(f"已保存 data/sample_20.csv（{len(sample)} 条）")