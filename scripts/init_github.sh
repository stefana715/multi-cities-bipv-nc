#!/bin/bash
# ============================================================================
# Paper 4 — GitHub 仓库初始化脚本
# ============================================================================
# 使用方法:
#   1. 先在 GitHub 上创建一个空仓库（不要勾选 README/gitignore/license）
#   2. 修改下面的 REPO_URL 为你的仓库地址
#   3. 在 paper4/ 目录下运行: bash scripts/init_github.sh
# ============================================================================

REPO_URL="https://github.com/YOUR_USERNAME/paper4-bipv-suitability.git"
# 或者用 SSH: git@github.com:YOUR_USERNAME/paper4-bipv-suitability.git

echo "=========================================="
echo "Paper 4 — 初始化 GitHub 仓库"
echo "=========================================="

# 检查是否已经是 git 仓库
if [ -d ".git" ]; then
    echo "⚠️  已经是 git 仓库，跳过 init"
else
    git init
    echo "✓ git init 完成"
fi

# 添加远程仓库
git remote add origin "$REPO_URL" 2>/dev/null || \
    echo "⚠️  远程仓库 origin 已存在，跳过"

# 设置主分支名称
git branch -M main

# 添加所有文件
git add -A

# 首次提交
git commit -m "feat: Paper 4 项目脚手架初始化

- 五城市 YAML 配置文件 (哈尔滨/北京/长沙/深圳/昆明)
- 备选城市配置 (长春/沈阳/西安/济南/贵阳)
- OSM 数据质量审计脚本 (Layer 3 筛选)
- PVGIS 数据下载脚本
- 项目目录结构与 README
- requirements.txt"

echo ""
echo "=========================================="
echo "准备推送，请确认 REPO_URL 已修改："
echo "  当前: $REPO_URL"
echo ""
echo "如果正确，运行："
echo "  git push -u origin main"
echo "=========================================="
