import json
import re

INPUT = "data/jd_llm_fixed.jsonl"
OUTPUT = "data/jd_chunks.jsonl"

CHUNK_SIZE = 300      # 职责原文每块字数
CHUNK_OVERLAP = 50    # 块间重叠


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """把长文本按 size 字切，块间重叠 overlap 字"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    all_chunks = []

    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            url = rec.get("url", "")
            m = re.search(r"/id/(\d+)", url)
            job_id = m.group(1) if m else "unknown"

            meta = {
                "岗位名": rec.get("岗位名", ""),
                "公司": rec.get("公司", ""),
                "城市": rec.get("城市", ""),
                "学历要求": rec.get("学历要求", ""),
                "url": url,
            }

            # 块1：元信息
            meta_text = " ".join([
                rec.get("岗位名", ""),
                rec.get("公司", ""),
                rec.get("城市", ""),
                rec.get("学历要求", ""),
            ]).strip()
            if meta_text:
                all_chunks.append({
                    "chunk_id": f"jd_{job_id}_chunk0",
                    "job_id": job_id,
                    "chunk_type": "meta",
                    "text": meta_text,
                    "meta": meta,
                })

            # 块2：技能关键词
            skills = rec.get("技能关键词", [])
            if isinstance(skills, list):
                skills_text = "、".join(skills)
            else:
                skills_text = str(skills)
            if skills_text.strip():
                all_chunks.append({
                    "chunk_id": f"jd_{job_id}_chunk1",
                    "job_id": job_id,
                    "chunk_type": "skills",
                    "text": skills_text,
                    "meta": meta,
                })

            # 块3+：职责原文按长度切
            duty = rec.get("职责原文", "").strip()
            if duty:
                duty_chunks = split_text(duty)
                for i, dc in enumerate(duty_chunks):
                    all_chunks.append({
                        "chunk_id": f"jd_{job_id}_chunk{2+i}",
                        "job_id": job_id,
                        "chunk_type": "duty",
                        "text": dc,
                        "meta": meta,
                    })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 统计
    job_ids = set(c["job_id"] for c in all_chunks)
    type_count = {}
    for c in all_chunks:
        type_count[c["chunk_type"]] = type_count.get(c["chunk_type"], 0) + 1

    lengths = [len(c["text"]) for c in all_chunks]

    print(f"总 chunk 数：{len(all_chunks)}")
    print(f"涉及 JD 数：{len(job_ids)}")
    print(f"平均每 JD 块数：{len(all_chunks) / max(len(job_ids), 1):.1f}")
    print(f"按类型：{type_count}")
    print(f"chunk 长度：最短 {min(lengths)} / 最长 {max(lengths)} / 平均 {sum(lengths)//len(lengths)}")


if __name__ == "__main__":
    main()