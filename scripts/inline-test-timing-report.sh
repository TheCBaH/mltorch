#!/usr/bin/env bash
# Times each inline-test partition (dune invokes one runner process per
# source file in an (inline_tests) library) individually, bypassing dune's
# own scheduling, and fails if any partition exceeds the threshold. See
# _ai_/lstm_scale_test_timing_notes.md.
set -euo pipefail

usage() {
  echo "usage: $0 [--no-build] <test-dir> <dune-lib-name> <mode: best|js> [threshold-seconds] [report-path]" >&2
  exit 1
}

build=1
if [[ "${1:-}" == "--no-build" ]]; then
  build=0
  shift
fi

[[ $# -ge 3 ]] || usage
dir=$1
lib=$2
mode=$3
threshold=${4:-10}
report=${5:-_build/inline_timing_report.$lib.$mode.tsv}

build_dir="_build/default/$dir"
inline_dir=".$lib.inline-tests"

# `-source-tree-root` is relative to $build_dir, one `..` per path component
# of $dir (e.g. `test` -> `..`, `test/native` -> `../..`).
IFS='/' read -ra dir_parts <<< "$dir"
source_tree_root=$(printf -- '../%.0s' "${dir_parts[@]}")
source_tree_root=${source_tree_root%/}

# Generous relative to the pass/fail threshold: a slow partition should be
# measured and reported as FAIL below, not merely cut off.
kill_after=$(( threshold * 6 > 60 ? threshold * 6 : 60 ))

case "$mode" in
  js)
    runner_rel="$inline_dir/inline-test-runner.bc.js"
    runner_cmd="node $runner_rel"
    list_partitions() { node "$runner_rel" inline-test-runner "$lib" -list-partitions; }
    ;;
  best)
    runner_rel="$inline_dir/inline-test-runner.exe"
    runner_cmd="$runner_rel"
    list_partitions() { "$runner_rel" inline-test-runner "$lib" -list-partitions; }
    ;;
  *) usage ;;
esac

# `-k 10`: kill with SIGKILL 10s after SIGTERM if the test ignores it.
# `timeout` exits 124 when it had to kill the command -- that's what marks
# a row TIMEOUT below, as opposed to a command that merely failed on its own.
run_partition() {
  timeout -k 10 "$kill_after" $runner_cmd inline-test-runner "$lib" \
    -partition "$1" -source-tree-root "$source_tree_root" -diff-cmd -
}
partition_repro_cmd() {
  echo "(cd $build_dir && $runner_cmd inline-test-runner $lib -partition $1 -source-tree-root $source_tree_root -diff-cmd -)"
}

if [[ "$build" -eq 1 ]]; then
  dune build --root . "$build_dir/$runner_rel"
fi

# `-list-partitions` asks the runner directly rather than reading a
# `partitions-<mode>` file: that file is only a side effect of dune's own
# sandboxed runtest action, not a directly buildable target, and a
# single-module library (e.g. test/model_explorer/me_session_test) has no
# such file at all since there's nothing to partition.
partitions=$(cd "$build_dir" && list_partitions)

mkdir -p "$(dirname "$report")"
: > "$report"

while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  start=$(date +%s%3N)
  set +e
  ( cd "$build_dir" && run_partition "$f" ) >/dev/null 2>&1
  ec=$?
  set -e
  end=$(date +%s%3N)
  printf '%d\t%s\t%d\t%s\n' "$((end - start))" "$f" "$ec" "$(partition_repro_cmd "$f")" >> "$report"
done <<< "$partitions"

sort -t $'\t' -k1,1rn -o "$report" "$report"

echo "inline test timing report ($lib/$mode, threshold=${threshold}s): $report" >&2
cut -f1-3 "$report" | column -t -s $'\t' >&2

awk -F'\t' -v threshold_ms=$((threshold * 1000)) '
  $3 == 124 {
    printf "TIMEOUT: %s (%s) killed after %s -- reproduce with: %s\n", $2, "'"$mode"'", "'"${kill_after}s"'", $4 > "/dev/stderr"
    bad = 1
    next
  }
  $3 != 0 {
    printf "FAIL: %s (%s) exited %d -- reproduce with: %s\n", $2, "'"$mode"'", $3, $4 > "/dev/stderr"
    bad = 1
    next
  }
  $1 > threshold_ms {
    printf "FAIL: %s (%s) took %dms, exceeding the %dms threshold -- reproduce with: %s\n", $2, "'"$mode"'", $1, threshold_ms, $4 > "/dev/stderr"
    bad = 1
  }
  END { exit(bad ? 1 : 0) }
' "$report"
