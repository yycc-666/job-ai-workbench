# 安装 Git 钩子：让 scripts/git-hooks/ 下的钩子生效（提交前做 PII 检查）
# 用法：pwsh scripts/install_hooks.ps1
$ErrorActionPreference = "Stop"

$repo = git rev-parse --show-toplevel
if (-not $repo) { throw "当前目录不在 git 仓库里" }

git -C $repo config core.hooksPath scripts/git-hooks
Write-Output "已启用 core.hooksPath = scripts/git-hooks（钩子随仓库一起版本管理）"

Write-Output "自测当前仓库："
python "$repo/scripts/check_pii.py"
Write-Output "退出码 $LASTEXITCODE（0 = 没有 PII）"
