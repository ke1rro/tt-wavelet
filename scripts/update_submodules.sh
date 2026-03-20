#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/tt-wavelet}"
TARGET="$ROOT/tt-metal"
SNAPSHOT_DIR="$ROOT/.submodule_snapshots"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SNAPSHOT_FILE="$SNAPSHOT_DIR/ttmetal_before_main_$TIMESTAMP.tsv"

mkdir -p "$SNAPSHOT_DIR"

if [[ ! -d "$TARGET/.git" && ! -f "$TARGET/.git" ]]; then
    echo "tt-metal repo not found at: $TARGET"
    exit 1
fi

echo "=== Target repo: $TARGET"
cd "$TARGET"

echo "=== Sync/init submodules ==="
git submodule sync --recursive
git submodule update --init --recursive

echo "=== Saving snapshot to $SNAPSHOT_FILE ==="

save_repo_state() {
    local repo_path="$1"
    local rel_path="$2"

    local branch
    branch="$(git -C "$repo_path" symbolic-ref --short -q HEAD || true)"
    if [[ -z "$branch" ]]; then
        branch="DETACHED"
    fi

    local commit
    commit="$(git -C "$repo_path" rev-parse HEAD)"

    printf "%s\t%s\t%s\n" "$rel_path" "$branch" "$commit" >> "$SNAPSHOT_FILE"
}

: > "$SNAPSHOT_FILE"

save_repo_state "$TARGET" "."

git submodule foreach --recursive '
    branch="$(git symbolic-ref --short -q HEAD || true)"
    if [[ -z "$branch" ]]; then
        branch="DETACHED"
    fi
    commit="$(git rev-parse HEAD)"
    printf "%s\t%s\t%s\n" "$sm_path" "$branch" "$commit" >> "'"$SNAPSHOT_FILE"'"
'

echo "=== Switching root repo to main ==="
git fetch origin --prune
git checkout main
git pull --ff-only origin main

echo "=== Switching all submodules to main ==="
git submodule foreach --recursive '
    echo "--- [$sm_path]"
    git fetch origin --prune

    if git show-ref --verify --quiet refs/remotes/origin/main; then
        if git show-ref --verify --quiet refs/heads/main; then
            git checkout main
        else
            git checkout -b main --track origin/main
        fi
        git pull --ff-only origin main
    else
        echo "WARNING: origin/main not found for $sm_path, skipping"
    fi
'

echo "=== Final state ==="
echo "[root]"
git branch --show-current || true
git rev-parse --short HEAD

git submodule foreach --recursive '
    echo "[$sm_path] branch=$(git branch --show-current || true) commit=$(git rev-parse --short HEAD)"
'

echo
echo "Snapshot saved to:"
echo "$SNAPSHOT_FILE"