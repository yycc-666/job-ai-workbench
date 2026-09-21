import os
import csv
import json
import time
import re
import requests

API_KEY = os.getenv("MIMO_API_KEY")
if not API_KEY:
    raise SystemExit("请先设置环境变量 MIMO_API_KEY")

API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = "mimo-v2.5-pro"

INPUT_CSV = "whpu_jobs_detail.csv"
OUTPUT_JSONL = "data/jd_llm.jsonl"
FAILED_LOG = "data/jd_llm_failed.txt"

SYSTEM_PROMPT = """你是一个招聘信息结构化抽取引擎。用户给你一段招聘JD原文，你只输出一个JSON对象，不输出任何解释、markdown代码块、或额外文字。"""

USER_PROMPT_TEMPLATE = """请从下面的招聘JD中抽取字段，输出严格JSON（不要```json```包裹）：

{{
  "岗位名": "从标题或正文抽，简短",
  "公司": "公司全称",
  "城市": "只返回'XX市'，如'武汉市'；没有则空字符串",
  "学历要求": "只填 博士/硕士/本科/大专/不限 中的一个；找不到填'不限'",
  "技能关键词": ["从JD全文抽取的技术栈、证书、能力关键词，最多10个"],
  "职责原文": "从'岗位职责'或'工作职责'开始，到'任职要求'或'岗位要求'前结束；找不到则返回整段JD"
}}

抽取规则（必须遵守）：
1. 城市：只返回"XX市"格式。不要返回"中国XX省XX市XX区"，不要返回"XX省"。
2. 学历要求：只从"学历要求"字段或"本科及以上"这类明确表述中抽。不要从"博士年龄32周岁以下"这类年龄要求里误命中。
3. 技能关键词：从JD全文抽，包括技术语言、工具、证书、软技能。如"Python、Java、沟通能力、驾照"。最多10个。
4. 职责原文：找"岗位职责"/"工作职责"/"职位描述"到"任职要求"/"岗位要求"/"福利待遇"之间的内容。找不到标记就返回整段JD。
5. 所有字段都必须存在。找不到内容就填空字符串或空数组。

JD原文：
---
{jd_text}
---

只输出JSON："""


def build_jd_text(row):
    """把CSV一行拼成一段完整JD文本，喂给LLM"""
    parts = [
        f"标题：{row.get('title', '')}",
        f"公司：{row.get('公司名称', '') or row.get('company', '')}",
        f"工作地址：{row.get('工作地址', '')}",
        f"学历要求：{row.get('学历要求', '')}",
        f"需求专业：{row.get('需求专业', '')}",
        f"职位详情：{row.get('职位详情', '')}",
    ]
    return "\n".join(parts)


def parse_llm_json(text):
    """清洗LLM返回，抽出JSON"""
    text = text.strip()
    # 去掉可能的 ```json ``` 包裹
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("返回里找不到JSON")
    return json.loads(text[start:end + 1])


def call_llm(jd_text, retry=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(jd_text=jd_text[:6000])},
        ],
        "temperature": 0,
    }
    for i in range(retry):
        try:
            r = requests.post(API_URL, headers=headers, json=data, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return parse_llm_json(content)
        except Exception as e:
            print(f"  重试 {i+1}/{retry}：{e}")
            time.sleep(2 * (i + 1))
    raise RuntimeError("LLM调用失败")


def load_done_urls(path):
    """断点续跑：读已经处理完的url"""
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec.get("url", ""))
                except Exception:
                    continue
    return done


def main():
    # 读原始CSV
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"读取 {len(rows)} 条原始JD")

    # 断点续跑
    done_urls = load_done_urls(OUTPUT_JSONL)
    print(f"已完成 {len(done_urls)} 条，剩余 {len(rows) - len(done_urls)} 条")

    # 追加写入
    out_f = open(OUTPUT_JSONL, "a", encoding="utf-8")
    failed = []

    for i, row in enumerate(rows, 1):
        url = row.get("url", "")
        if url in done_urls:
            continue

        print(f"[{i}/{len(rows)}] {row.get('title', '')[:30]}...")
        try:
            jd_text = build_jd_text(row)
            result = call_llm(jd_text)
            result["url"] = url
            result["title_原始"] = row.get("title", "")
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"  -> OK：{result.get('岗位名', '')} | {result.get('城市', '')}")
        except Exception as e:
            print(f"  -> 失败：{e}")
            failed.append(url)

        time.sleep(0.5)  # 限频

    out_f.close()

    if failed:
        with open(FAILED_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        print(f"\n{len(failed)} 条失败，已记录到 {FAILED_LOG}")
    else:
        print(f"\n全部完成，输出 {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()