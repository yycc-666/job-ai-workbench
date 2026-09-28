#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提交前 PII 检查：扫描待提交文件，命中手机号 / 邮箱 / 身份证号就拒绝提交。

用法：
    python scripts/check_pii.py                # 检查 git 暂存区文件（pre-commit 钩子调用）
    python scripts/check_pii.py 文件1 文件2     # 检查指定文件

退出码：0 = 通过，1 = 发现 PII（阻止提交）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 复用脱敏脚本里的同一套正则，避免两处规则各改各的、逐渐漂移
from redact_pii import PATTERNS  # noqa: E402

BINARY_SUFFIXES = {
    ".npz", ".npy", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
    ".zip", ".gz", ".tar", ".woff", ".woff2", ".ttf", ".exe", ".dll",
}


def staged_files() -> list[Path]:
    """取暂存区新增/修改的文件。core.quotepath=false 保证中文文件名不被转义。"""
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    args = sys.argv[1:]
    files = [Path(a) for a in args] if args else staged_files()
    found_total = 0

    for path in files:
        if not path.exists() or path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for pattern, label in PATTERNS:
            found = sorted({m.group(0) for m in pattern.finditer(text)})
            if not found:
                continue
            found_total += len(found)
            preview = "、".join(found[:3]) + ("…" if len(found) > 3 else "")
            print(f"[PII] {path.relative_to(ROOT)} -> {label} {preview}")

    if found_total:
        print()
        print(f"拒绝提交：发现 {found_total} 处个人身份信息。")
        print("处理办法：python scripts/redact_pii.py，或把该文件写进 .gitignore")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
