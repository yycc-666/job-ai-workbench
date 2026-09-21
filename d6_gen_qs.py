import os
import json
import time
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

API_KEY = os.getenv("MIMO_API_KEY")
if not API_KEY:
    raise SystemExit("请先设置环境变量 MIMO_API_KEY")

API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = "mimo-v2.5-pro"

JD_FILE = "data/jd_llm_fixed.jsonl"
RESUME_FILE = "resume.txt"
OUTPUT = "data/interview_qs.jsonl"
FAILED_LOG = "data/interview_qs_failed.txt"

# 想生成题库的 job_id 列表。先跑 20 条试试
TARGET_JOB_IDS = [
    # AI 相关 16 条
    "2384397", "2384848", "2385692", "2385096", "2391103",
    "2384968", "2385095", "2384113", "2384112",
    "2390919", "2293596", "2384525", "2385849", "2384083",
    "2376253", "2369258",

    # 后端/Java 补 8 条（技术栈接近 AI 应用开发）
    "2388358", "2366060", "1857727", "2387531",
    "2384395", "2385850",
    "2387017", "2387008",

    # 保底 IT 数字化 6 条
    "2389420", "2384556", "2385844", "2385831", "2385851", "2384873",
]  # None = 全部；或者填 ["2391103", "2388358", ...]

MAX_WORKERS = 3  # LLM 接口慢，3 并发够了

SYSTEM_PROMPT = """你是一位有 10 年经验的技术面试官。你会根据候选人的简历和岗位 JD，预测候选人在面试中最可能被问到的 8 个问题。只输出 JSON，不输出解释、markdown 代码块或额外文字。"""

USER_PROMPT_TEMPLATE = """# 岗位 JD
岗位名：{岗位名}
公司：{公司}
城市：{城市}
技能关键词：{技能关键词}
职责原文：
{职责原文}

# 候选人简历
{简历}

# 任务
你是面试官，要预测候选人面试中最可能被问到的 8 个问题。

## 出题规则（严格遵守）
1. **技术基础题**：只看 JD 的技能关键词和职责，考察岗位要求的技能。
   - 如果 JD 要求「SolidWorks、Creo」，就问结构设计相关问题，不要问爬虫
   - 如果 JD 要求「Java、Spring」，就问 Java 后端问题，不要问数据清洗
   - **禁止把简历里的技能当考点**，除非它正好和 JD 技能重合
2. **项目深挖题**：从简历里的项目经历出题，考察候选人做过的东西。
3. **岗位匹配题**：问为什么投这个岗、如何理解岗位、技能如何迁移。
4. **行为面试题**：团队协作、抗压、失败经历等通用软技能。

## 四类题各 2 道，共 8 道

输出严格 JSON：
{{
  "questions": [
    {{"category": "技术基础", "question": "...", "why": "..."}},
    {{"category": "技术基础", "question": "...", "why": "..."}},
    {{"category": "项目深挖", "question": "...", "why": "..."}},
    {{"category": "项目深挖", "question": "...", "why": "..."}},
    {{"category": "岗位匹配", "question": "...", "why": "..."}},
    {{"category": "岗位匹配", "question": "...", "why": "..."}},
    {{"category": "行为面试", "question": "...", "why": "..."}},
    {{"category": "行为面试", "question": "...", "why": "..."}}
  ]
}}

只输出 JSON："""

write_lock = threading.Lock()


def extract_job_id(url):
    m = re.search(r"/id/(\d+)", url or "")
    return m.group(1) if m else ""


def parse_llm_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("返回里找不到 JSON")
    return json.loads(text[start:end + 1])


def call_llm(prompt, retry=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    for i in range(retry):
        try:
            r = requests.post(API_URL, headers=headers, json=data, timeout=90)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return parse_llm_json(content)
        except Exception as e:
            print(f"  重试 {i+1}/{retry}: {e}")
            time.sleep(2 * (i + 1))
    raise RuntimeError("LLM 调用失败")


def load_done_job_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec.get("job_id", ""))
                except Exception:
                    continue
    return done


def process_one(rec, resume, out_f):
    jid = extract_job_id(rec.get("url", ""))
    try:
        skills = rec.get("技能关键词", [])
        if isinstance(skills, list):
            skills = "、".join(skills)

        prompt = USER_PROMPT_TEMPLATE.format(
            岗位名=rec.get("岗位名", ""),
            公司=rec.get("公司", ""),
            城市=rec.get("城市", ""),
            技能关键词=skills,
            职责原文=rec.get("职责原文", "")[:3000],
            简历=resume[:2000],
        )

        result = call_llm(prompt)
        questions = result.get("questions", [])
        if not isinstance(questions, list):
            raise ValueError("questions 不是列表")

        out = {
            "job_id": jid,
            "岗位名": rec.get("岗位名", ""),
            "公司": rec.get("公司", ""),
            "城市": rec.get("城市", ""),
            "url": rec.get("url", ""),
            "questions": questions,
        }
        with write_lock:
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_f.flush()
        return jid, True, len(questions)
    except Exception as e:
        return jid, False, str(e)


def main():
    print("读取简历...")
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        resume = f.read()

    print("读取 JD 数据...")
    rows = []
    with open(JD_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            jid = extract_job_id(rec.get("url", ""))
            if TARGET_JOB_IDS and jid not in TARGET_JOB_IDS:
                continue
            rows.append(rec)
    print(f"  {len(rows)} 条 JD 待处理")

    done_ids = load_done_job_ids(OUTPUT)
    pending = [r for r in rows if extract_job_id(r.get("url", "")) not in done_ids]
    print(f"  已完成 {len(done_ids)}，待处理 {len(pending)}")

    if not pending:
        print("没有待处理的")
        return

    out_f = open(OUTPUT, "a", encoding="utf-8")
    failed = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, r, resume, out_f): r for r in pending}
        for future in as_completed(futures):
            jid, ok, info = future.result()
            done_count += 1
            if ok:
                print(f"[{done_count}/{len(pending)}] OK: {jid}, {info} 题")
            else:
                print(f"[{done_count}/{len(pending)}] 失败: {jid} | {info}")
                failed.append(jid)

    out_f.close()

    if failed:
        with open(FAILED_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        print(f"\n{len(failed)} 条失败，已记录到 {FAILED_LOG}，重跑可续")

    print(f"\n全部完成，输出 {OUTPUT}")


if __name__ == "__main__":
    main()