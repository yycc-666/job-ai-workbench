#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布前脱敏：把手机号 / 邮箱 / 身份证号替换为占位符。

用法：
    python scripts/redact_pii.py            # 就地脱敏 data/ 下的数据文件
    python scripts/redact_pii.py --check    # 只扫描不修改，发现 PII 返回退出码 1
    python scripts/redact_pii.py 文件1 文件2  # 只处理指定文件

设计说明：
    - 只替换文本，不改动 JSON 结构；写回前会逐行校验仍是合法 JSON，不合法就拒绝写入
    - 幂等：已经脱敏过的文本再跑一次不会有任何变化
    - 纯标准库，无需安装任何依赖

踩坑记录（改这个文件前请先读完）：
    1. 邮箱正则必须带 (?<!\\\\) 负向后顾。JSON 里换行写成转义序列（反斜杠 + 字母 n），
       如果邮箱正好紧跟在它后面，正则会把转义用的那个字母 n 当成邮箱首字符一起吃掉，
       结果把「反斜杠 + n + 邮箱」改成了「反斜杠 + [邮箱已脱敏]」。
       而反斜杠后面跟方括号不是合法的 JSON 转义，整行 JSON 就解析失败了。
    2. 本文档里也不要写任何真实的邮箱、手机号示例——scripts/git-hooks/pre-commit
       会把它当成真实 PII 直接拦下提交（它分不清示例和真数据）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TARGETS = [
    "data/jd.jsonl",
    "data/jd_llm.jsonl",
    "data/jd_llm_fixed.jsonl",
    "data/jd_chunks.jsonl",
]

# 顺序很重要：邮箱要排在手机号前面，
# 否则「手机号 + @域名」这种邮箱会先被手机号规则切碎，只剩半截地址。
# (?<!\\\\) 保证不会把 JSON 转义序列的反斜杠前缀吃掉。
PATTERNS = [
    (re.compile(r"(?<!\\)[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[邮箱已脱敏]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号已脱敏]"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[身份证已脱敏]"),
]


def scan(text: str) -> list[str]:
    """返回文本中命中的全部个人身份信息（含重复）。"""
    hits: list[str] = []
    for pattern, _ in PATTERNS:
        hits.extend(m.group(0) for m in pattern.finditer(text))
    return hits


def redact(text: str) -> str:
    for pattern, placeholder in PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def jsonl_error(name: str, text: str) -> str | None:
    """脱敏结果如果是 .jsonl，逐行校验合法性；不合法返回错误说明。"""
    if not name.endswith(".jsonl"):
        return None
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            return f"{name} 第 {lineno} 行脱敏后不是合法 JSON：{exc}"
    return None


def show(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="发布前 PII 脱敏（手机号 / 邮箱 / 身份证）")
    parser.add_argument("--check", action="store_true", help="只检查，不修改文件")
    parser.add_argument("paths", nargs="*", help="要处理的文件（默认 data/ 下的数据文件）")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths] or [ROOT / t for t in DEFAULT_TARGETS]

    total = 0
    touched = 0
    failures = 0

    for path in paths:
        if not path.exists():
            print(f"跳过（不存在）：{show(path)}")
            continue

        text = path.read_text(encoding="utf-8")
        hits = scan(text)
        if not hits:
            print(f"干净：{show(path)}")
            continue

        total += len(hits)
        uniq = sorted(set(hits))
        print(f"命中 {len(hits)} 处（{len(uniq)} 个唯一值）：{show(path)}")
        if args.check:
            continue

        new_text = redact(text)
        error = jsonl_error(path.name, new_text)
        if error:
            failures += 1
            print(f"  [错误] {error}")
            print("  [跳过] 原文件保持不变，请检查正则规则")
            continue

        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            touched += 1
            print(f"  -> 已写入脱敏结果：{path.name}")

    print()
    if args.check:
        if total:
            print(f"[失败] 共发现 {total} 处个人身份信息，请先运行 python scripts/redact_pii.py")
            return 1
        print("[通过] 未发现手机号 / 邮箱 / 身份证号")
        return 0

    if failures:
        print(f"[失败] {failures} 个文件校验不通过，未写入")
        return 1

    print(f"[完成] 共处理 {total} 处，写入 {touched} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
