#!/usr/bin/env bash
# Download WOMD v1.3.1 scenario tfrecords → convert to shards → delete tfrecord.
# Keeps at most $PARALLEL tfrecords on disk at a time (one per worker).
# Resume-safe: any index whose output shard already exists is skipped.
#
# Usage:   bash scripts/prepare_shards.sh training validation testing
# Env:
#   SHARD_ROOT        destination root dir   (default: $HOME/shards)
#   TFRECORD_TMPDIR   scratch for tfrecords  (default: /tmp/womd_tfrecord)
#   PARALLEL          concurrent workers     (default: 8)
#
# Layout produced:
#   $SHARD_ROOT/training/train.<i>of1000.pt.zst
#   $SHARD_ROOT/validation/val.<i>of150.pt.zst
#   $SHARD_ROOT/test/test.<i>of150.pt.zst

set -u

ROOT="${SHARD_ROOT:-$HOME/shards}"
TMPDIR="${TFRECORD_TMPDIR:-/tmp/womd_tfrecord}"
PARALLEL="${PARALLEL:-8}"
BUCKET="gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$TMPDIR"

# Activate venv once in the parent so workers inherit it — avoids `uv run`
# startup overhead (~1-2 s) on every single file.
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

export REPO TMPDIR BUCKET

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
export -f log

worker() {
  # args: <split> <i> <total> <pad_src> <pad_total> <out>
  local split=$1 i=$2 total=$3 pad_src=$4 pad_total=$5 out=$6
  local src="$BUCKET/$split/${split}.tfrecord-${pad_src}-of-${pad_total}"
  local tmp="$TMPDIR/${split}.tfrecord-${pad_src}"

  # Another worker may have finished this index between gen_jobs and now.
  if [[ -f "$out" ]]; then
    return 0
  fi

  log "[$split $i/$total] download"
  if ! gsutil -q cp "$src" "$tmp"; then
    log "[$split $i/$total] DOWNLOAD FAILED"
    rm -f "$tmp"
    return 0
  fi

  log "[$split $i/$total] convert"
  if (cd "$REPO" && python -m data.convert "$tmp" "$out" >/dev/null 2>&1); then
    log "[$split $i/$total] ok"
  else
    log "[$split $i/$total] CONVERT FAILED"
    rm -f "$out"
  fi
  rm -f "$tmp"
}
export -f worker

# bucket-path -> (total, output-subdir, shard-prefix)
declare -A N=(       [training]=1000    [validation]=150   [testing]=150  )
declare -A OUTDIR=(  [training]=training [validation]=validation [testing]=test )
declare -A PREFIX=(  [training]=train    [validation]=val   [testing]=test )

gen_jobs() {
  local split total prefix outdir pad_total i pad_src out
  for split in "$@"; do
    total=${N[$split]}
    prefix=${PREFIX[$split]}
    outdir="$ROOT/${OUTDIR[$split]}"
    mkdir -p "$outdir"
    pad_total=$(printf "%05d" "$total")
    for i in $(seq 0 $((total - 1))); do
      out="$outdir/${prefix}.${i}of${total}.pt.zst"
      if [[ -f "$out" ]]; then
        continue
      fi
      pad_src=$(printf "%05d" "$i")
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$split" "$i" "$total" "$pad_src" "$pad_total" "$out"
    done
  done
}

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <split> [<split> ...]   (splits: training validation testing)" >&2
  exit 2
fi

for split in "$@"; do
  if [[ -z "${N[$split]:-}" ]]; then
    echo "unknown split: $split" >&2
    exit 2
  fi
done

log "starting with PARALLEL=$PARALLEL  ROOT=$ROOT"
gen_jobs "$@" | \
  xargs -P "$PARALLEL" -n 1 -d '\n' -I{} bash -c '
    IFS=$'"'"'\t'"'"' read -r split i total pad_src pad_total out <<< "$1"
    worker "$split" "$i" "$total" "$pad_src" "$pad_total" "$out"
  ' _ {}

log "all done."
