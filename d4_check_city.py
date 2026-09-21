import json
import re

TARGET_IDS = ["2386836", "2386831", "2386823", "2386817", "2385729",
              "2387258", "2385692", "2386407", "2385443"]

def extract_job_id(url):
    m = re.search(r"/id/(\d+)", url or "")
    return m.group(1) if m else ""

with open("data/jd_llm_fixed.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        jid = extract_job_id(rec.get("url", ""))
        if jid in TARGET_IDS:
            print(f"{jid} | 岗位: {rec.get('岗位名', '')[:25]:<25} | 城市: '{rec.get('城市', '')}'")