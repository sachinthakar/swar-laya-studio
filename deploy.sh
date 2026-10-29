#!/usr/bin/env bash
# deploy.sh — Swar-Laya Studio: commit + push to the branch Render deploys from.
#
# Render's static site (see render.yaml) auto-deploys whenever new commits
# land on the "render-deploy" branch of this repo's GitHub remote. Pushing
# to that branch IS the deploy -- this script is just a safe, repeatable way
# to do that push (switch to the branch, commit whatever's changed, push).
#
# Auth: this repo's "origin" remote already has a GitHub Personal Access
# Token baked into its URL (check with: git remote -v), so a plain
# `git push` authenticates with no separate login step. If that token is
# ever rotated or revoked, update the remote with:
#   git remote set-url origin https://<github-user>:<new-token>@github.com/<owner>/<repo>.git
# Never paste a fresh token into this file -- it would then be committed
# to the repo. Keep it only in the remote URL (which .git/config is not
# itself tracked by git) or in your own password manager.
#
# Optional: set RENDER_DEPLOY_HOOK_URL (from Render dashboard -> the
# swar-laya-studio service -> Settings -> Deploy Hook) to also force a
# redeploy with no new commit, e.g. after changing an env var in the
# Render dashboard rather than in code.
#
# Usage:
#   ./deploy.sh                       # auto commit message with timestamp
#   ./deploy.sh "fix raga dropdown"   # custom commit message
#   RENDER_DEPLOY_HOOK_URL="https://api.render.com/deploy/srv-xxxx?key=yyyy" ./deploy.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BRANCH="render-deploy"
MSG="${1:-Deploy $(date -u +'%Y-%m-%d %H:%M UTC')}"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$current_branch" != "$BRANCH" ]; then
    echo "[*] Switching from '$current_branch' to '$BRANCH'..."
    git checkout "$BRANCH"
fi

echo "[*] Staging changes..."
git add -A

if git diff --cached --quiet; then
    echo "[*] Nothing new to commit."
else
    git commit -m "$MSG"
fi

echo "[*] Pushing to origin/$BRANCH..."
git push origin "$BRANCH"

if [ -n "${RENDER_DEPLOY_HOOK_URL:-}" ]; then
    echo "[*] Triggering Render deploy hook..."
    curl -fsS -X POST "$RENDER_DEPLOY_HOOK_URL" && echo "[*] Deploy hook fired."
fi

echo "[*] Done. Render auto-deploys on push to '$BRANCH' -- check the Render"
echo "    dashboard (swar-laya-studio service -> Events) for build/deploy status."
