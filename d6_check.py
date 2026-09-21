import json

with open("data/interview_qs.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"总共 {len(lines)} 条\n")

TARGET = ["2384397", "2384848", "2385692", "2391103", "2385096"]

for line in lines:
    rec = json.loads(line)
    if rec["job_id"] in TARGET:
        print(f"=== {rec['岗位名']} ({rec['公司']}) ===")
        for q in rec["questions"]:
            print(f"[{q['category']}] {q['question']}")
        print()