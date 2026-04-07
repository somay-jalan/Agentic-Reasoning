#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./extract_pr.sh <repo_dir> <commit_sha> <artifact_dir>
#
# Example:
#   ./extract_pr.sh sympy 127f2b43ec benchmark/artifacts/pr29394

REPO_DIR="${1:-}"
COMMIT_SHA="${2:-}"
ARTIFACT_DIR="${3:-}"

if [ -z "$REPO_DIR" ] || [ -z "$COMMIT_SHA" ] || [ -z "$ARTIFACT_DIR" ]; then
  echo "Usage: $0 <repo_dir> <commit_sha> <artifact_dir>"
  exit 1
fi

# Resolve to absolute paths BEFORE cd
REPO_DIR="$(realpath -m "$REPO_DIR")"
ARTIFACT_DIR="$(realpath -m "$ARTIFACT_DIR")"

mkdir -p "$ARTIFACT_DIR"

echo "============================================================"
echo "Resolved paths"
echo "============================================================"
echo "REPO_DIR     = $REPO_DIR"
echo "COMMIT_SHA   = $COMMIT_SHA"
echo "ARTIFACT_DIR = $ARTIFACT_DIR"

cd "$REPO_DIR"

echo
echo "============================================================"
echo "Inspecting commit: $COMMIT_SHA"
echo "============================================================"
git show --no-patch --pretty=fuller "$COMMIT_SHA"

echo
echo "============================================================"
echo "Detecting parents"
echo "============================================================"

PARENTS=($(git rev-list --parents -n 1 "$COMMIT_SHA"))
NUM_PARENTS=$((${#PARENTS[@]} - 1))

echo "Commit: ${PARENTS[0]}"
echo "Number of parents: $NUM_PARENTS"

if [ "$NUM_PARENTS" -eq 0 ]; then
  echo "ERROR: Commit has no parent (root commit)."
  exit 1
fi

BASE_PARENT="${PARENTS[1]}"
echo "Using first parent as BASE: $BASE_PARENT"

if [ "$NUM_PARENTS" -gt 1 ]; then
  echo "This is a MERGE commit."
else
  echo "This is a NORMAL commit."
fi

echo
echo "============================================================"
echo "Files changed between BASE and COMMIT"
echo "============================================================"

CHANGED_FILES=$(git diff --name-only "$BASE_PARENT" "$COMMIT_SHA" || true)

if [ -z "$CHANGED_FILES" ]; then
  echo "No files changed between $BASE_PARENT and $COMMIT_SHA"
  exit 1
fi

echo "$CHANGED_FILES"

echo
echo "============================================================"
echo "Collecting changed Python files"
echo "============================================================"

PY_FILES=$(echo "$CHANGED_FILES" | grep '\.py$' || true)

if [ -z "$PY_FILES" ]; then
  echo "No changed Python files found."
  exit 0
fi

echo "$PY_FILES"

echo
echo "============================================================"
echo "Extracting before/after versions"
echo "============================================================"

for FILE in $PY_FILES; do
  SAFE_NAME=$(echo "$FILE" | tr '/' '_' | sed 's/\.py$//')

  BEFORE_OUT="${ARTIFACT_DIR}/${SAFE_NAME}_before.py"
  AFTER_OUT="${ARTIFACT_DIR}/${SAFE_NAME}_after.py"
  DIFF_OUT="${ARTIFACT_DIR}/${SAFE_NAME}_fix.diff"

  echo
  echo "Processing: $FILE"
  echo "  BEFORE -> $BEFORE_OUT"
  echo "  AFTER  -> $AFTER_OUT"
  echo "  DIFF   -> $DIFF_OUT"

  # AFTER version (must exist in COMMIT)
  if git cat-file -e "${COMMIT_SHA}:${FILE}" 2>/dev/null; then
    git show "${COMMIT_SHA}:${FILE}" > "$AFTER_OUT"
  else
    echo "  [WARN] File does not exist in commit: $FILE"
    continue
  fi

  # BEFORE version (may not exist if newly added)
  if git cat-file -e "${BASE_PARENT}:${FILE}" 2>/dev/null; then
    git show "${BASE_PARENT}:${FILE}" > "$BEFORE_OUT"
  else
    echo "  [INFO] File did not exist before commit (new file)."
    : > "$BEFORE_OUT"
  fi

  # DIFF
  git diff "$BASE_PARENT" "$COMMIT_SHA" -- "$FILE" > "$DIFF_OUT"

  echo "  [OK] Wrote files for $FILE"
done

echo
echo "============================================================"
echo "Done."
echo "Artifacts written to: $ARTIFACT_DIR"
echo "============================================================"