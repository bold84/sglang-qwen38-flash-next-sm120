#!/usr/bin/env bash
# Reproduce the exact SGLang tree constructed in the candidate image.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
lock=stack.lock.json
containerfile=Containerfile

for tool in jq git; do command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 1; }; done
if command -v sha256sum >/dev/null; then
  sha256_file() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null; then
  sha256_file() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "missing SHA-256 tool" >&2; exit 1
fi

"$repo/scripts/validate-release.sh"
lock_value() { jq -r --arg arg "$1" '.pins[] | select(.arg == $arg) | .value' "$lock"; }

echo "== lock and Containerfile pins =="
while IFS=$'\t' read -r arg value; do
  grep -Fxq "ARG ${arg}=${value}" "$containerfile" || { echo "Containerfile pin mismatch: $arg" >&2; exit 1; }
  printf '  %s\n' "$arg"
done < <(jq -r '.pins[] | [.arg, .value] | @tsv' "$lock")
while IFS= read -r arg; do
  [[ -n "$(lock_value "$arg")" ]] || { echo "unrecorded Containerfile pin: $arg" >&2; exit 1; }
done < <(sed -n 's/^ARG \([A-Z0-9_]*\)=.*/\1/p' "$containerfile")

echo "== archived patch bytes =="
while IFS=$'\t' read -r path sha; do
  [[ -f "$path" ]] || { echo "missing patch: $path" >&2; exit 1; }
  [[ "$(sha256_file "$path")" == "$sha" ]] || { echo "SHA-256 mismatch: $path" >&2; exit 1; }
  grep -Fq "COPY ${path} " "$containerfile" || { echo "Containerfile does not copy $path" >&2; exit 1; }
  printf '  %s\n' "$path"
done < <(jq -r '.patches[] | [.path, .sha256] | @tsv' "$lock")
while IFS= read -r path; do
  [[ "$(jq -r --arg p "$path" '[.patches[] | select(.path == $p)] | length' "$lock")" == 1 ]] || {
    echo "unrecorded patch: $path" >&2; exit 1;
  }
done < <(find patches -type f | sort)

source_repo=$(jq -er '.verification.sglang_repository' "$lock")
source_head=$(lock_value QWEN38_SGLANG_MAIN_HEAD)
source_tree=$(lock_value QWEN38_SGLANG_MAIN_TREE)
effective_tree=$(lock_value QWEN38_SGLANG_EFFECTIVE_TREE)
pr_head=$(lock_value QWEN38_SGLANG_PR36497_HEAD)
integration_head=$(lock_value QWEN38_SGLANG_INTEGRATION_HEAD)
[[ "$(jq -er '.integration.sglang.upstream_pr_head' "$lock")" == "$pr_head" ]] || { echo "PR head mismatch" >&2; exit 1; }
[[ "$(jq -er '.integration.sglang.head' "$lock")" == "$integration_head" ]] || { echo "integration head mismatch" >&2; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git init -q "$work/sglang"
git -C "$work/sglang" remote add origin "$source_repo"
git -C "$work/sglang" fetch -q --depth=1 origin "$source_head"
git -C "$work/sglang" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$work/sglang" rev-parse HEAD)" == "$source_head" ]]
[[ "$(git -C "$work/sglang" rev-parse 'HEAD^{tree}')" == "$source_tree" ]]
while IFS= read -r path; do
  git -C "$work/sglang" apply --index --binary "$repo/$path"
  printf '  applied %s\n' "$path"
done < <(jq -r '[.patches[] | select(.repository == "sglang")] | sort_by(.order)[].path' "$lock")
[[ "$(git -C "$work/sglang" write-tree)" == "$effective_tree" ]] || { echo "effective tree mismatch" >&2; exit 1; }

echo "source construction reproduces effective tree ${effective_tree}"

echo "== FlashInfer main source =="
flashinfer_repo=$(jq -er '.verification.flashinfer_repository' "$lock")
flashinfer_head=$(lock_value QWEN38_FLASHINFER_MAIN_HEAD)
flashinfer_tree=$(lock_value QWEN38_FLASHINFER_MAIN_TREE)
[[ "$(jq -er '.integration.flashinfer.head' "$lock")" == "$flashinfer_head" ]] || {
  echo "FlashInfer integration head mismatch" >&2; exit 1;
}
[[ "$(jq -er '.integration.flashinfer.tree' "$lock")" == "$flashinfer_tree" ]] || {
  echo "FlashInfer integration tree mismatch" >&2; exit 1;
}
git init -q "$work/flashinfer"
git -C "$work/flashinfer" remote add origin "$flashinfer_repo"
git -C "$work/flashinfer" fetch -q --depth=1 origin "$flashinfer_head"
git -C "$work/flashinfer" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$work/flashinfer" rev-parse HEAD)" == "$flashinfer_head" ]]
[[ "$(git -C "$work/flashinfer" rev-parse 'HEAD^{tree}')" == "$flashinfer_tree" ]]
echo "FlashInfer source reproduces tree ${flashinfer_tree}"
