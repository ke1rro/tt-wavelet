#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/tt-wavelet}"
SNAPSHOT_FILE="${2:-}"
TARGET="$ROOT/tt-metal"

if [[ -z "$SNAPSHOT_FILE" ]]; then
    echo "Usage: $0 <root-path> <snapshot-file>"
    echo "Example:"
    echo "  $0 ~/tt-wavelet ~/tt-wavelet/.submodule_snapshots/ttmetal_before_main_20260320_130000.tsv"
    exit 1
fi

if [[ ! -f "$SNAPSHOT_FILE" ]]; then
    echo "Snapshot file not found: $SNAPSHOT_FILE"
    exit 1
fi

cd "$TARGET"

echo "=== Sync/init submodules ==="
git submodule sync --recursive
git submodule update --init --recursive

restore_repo_state() {
    local repo_path="$1"
    local branch="$2"
    local commit="$3"

    git -C "$repo_path" fetch --all --tags --prune >/dev/null 2>&1 || true

    if [[ "$branch" == "DETACHED" ]]; then
        git -C "$repo_path" checkout --detach "$commit"
    else
        if git -C "$repo_path" show-ref --verify --quiet "refs/heads/$branch"; then
            git -C "$repo_path" checkout "$branch"
        elif git -C "$repo_path" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
            git -C "$repo_path" checkout -b "$branch" --track "origin/$branch"
        else
            git -C "$repo_path" checkout --detach "$commit"
            return
        fi

        if git -C "$repo_path" merge-base --is-ancestor "$commit" HEAD >/dev/null 2>&1; then
            git -C "$repo_path" reset --hard "$commit"
        else
            git -C "$repo_path" checkout --detach "$commit"
        fi
    fi
}

echo "=== Restoring from snapshot: $SNAPSHOT_FILE ==="

while IFS=$'\t' read -r rel_path branch commit; do
    [[ -z "$rel_path" || -z "$branch" || -z "$commit" ]] && continue

    if [[ "$rel_path" == "." ]]; then
        repo_path="$TARGET"
    else
        repo_path="$TARGET/$rel_path"
    fi

    echo "--- Restoring $rel_path | branch=$branch | commit=$commit"
    restore_repo_state "$repo_path" "$branch" "$commit"
done < "$SNAPSHOT_FILE"

echo "=== Restore complete ==="
echo "[root]"
git branch --show-current || true
git rev-parse --short HEAD

git submodule foreach --recursive '
    echo "[$sm_path] branch=$(git branch --show-current || true) commit=$(git rev-parse --short HEAD)"
'