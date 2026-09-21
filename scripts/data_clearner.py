import csv
import json
import re
import os
from bs4 import BeautifulSoup


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def clean_text(text):
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_city(address):
    if not address:
        return ""
    if "省" in address:
        address = address.split("省", 1)[1]
    address = address.replace("中国", "", 1)
    match = re.search(r'([\u4e00-\u9fa5]{2,6}?市)', address)
    if match:
        return match.group(1)
    return ""


def extract_education(text):
    if not text:
        return ""
    match = re.search(r'学历(?:要求)?[：:]\s*(博士|硕士|本科|大专|中专)', text)
    if match:
        return match.group(1)
    match = re.search(r'(博士|硕士|本科|大专|中专)(?:及以上|以上)?学历', text)
    if match:
        return match.group(1)
    match = re.search(r'统招(博士|硕士|本科|大专|中专)', text)
    if match:
        return match.group(1)
    return ""


SKILL_KEYWORDS = [
    "Python", "Java", "JavaScript", "C++", "C#", "Go", "PHP", "SQL",
    "Excel", "Word", "PowerPoint", "Photoshop", "CAD", "MATLAB",
    "MySQL", "Oracle", "MongoDB", "Redis",
    "Spring", "Django", "React", "Vue",
    "英语四级", "英语六级", "普通话", "日语", "韩语",
    "CPA", "CFA", "教师资格证", "会计证", "驾照", "驾驶证",
    "沟通能力", "团队协作", "抗压能力", "学习能力", "组织协调",
    "数据分析", "项目管理", "客户服务", "市场营销", "文案写作",
    "新媒体运营", "办公软件", "逻辑思维",
]


def extract_skills(text):
    if not text:
        return []
    return [kw for kw in SKILL_KEYWORDS if kw in text]


def extract_responsibility(detail_text):
    if not detail_text:
        return ""

    start_patterns = [
        r'岗位职责[：:\n]', r'工作职责[：:\n]', r'工作内容[：:\n]',
        r'职位描述[：:\n]', r'主要职责[：:\n]'
    ]
    end_patterns = [
        r'任职要求', r'岗位要求', r'应聘条件', r'任职资格',
        r'福利待遇', r'投递方式', r'联系方式', r'应聘方式',
        r'简历投递', r'薪资待遇', r'招聘流程', r'应聘流程',
        r'薪酬福利', r'培养路径', r'应聘资料',
    ]

    start_idx = -1
    for p in start_patterns:
        m = re.search(p, detail_text)
        if m:
            start_idx = m.end()
            break

    if start_idx == -1:
        return detail_text

    end_idx = len(detail_text)
    for p in end_patterns:
        m = re.search(p, detail_text[start_idx:])
        if m:
            candidate = start_idx + m.start()
            if candidate < end_idx:
                end_idx = candidate

    return detail_text[start_idx:end_idx].strip()


def clean_and_extract(input_csv, output_jsonl):
    rows = read_csv(input_csv)
    print(f"读取 {len(rows)} 条原始数据")

    seen_urls = set()
    cleaned = []

    for row in rows:
        url = row.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        detail_raw = row.get("职位详情", "")
        detail_clean = clean_text(detail_raw)

        record = {
            "岗位名": clean_text(row.get("title", "")),
            "公司": clean_text(row.get("公司名称", "")),
            "城市": extract_city(row.get("工作地址", "")),
            "学历要求": extract_education(detail_raw),
            "技能关键词": extract_skills(detail_clean),
            "职责原文": extract_responsibility(detail_clean),
            "url": url,
        }
        cleaned.append(record)

    print(f"去重后 {len(cleaned)} 条数据")

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in cleaned:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"已保存到 {output_jsonl}")

    skill_counts = {}
    for rec in cleaned:
        for kw in rec["技能关键词"]:
            skill_counts[kw] = skill_counts.get(kw, 0) + 1
    print("\n技能关键词命中统计（Top 10）：")
    for kw, cnt in sorted(skill_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {kw}: {cnt} 次")


if __name__ == "__main__":
    clean_and_extract("whpu_jobs_detail.csv", "data/jd.jsonl")