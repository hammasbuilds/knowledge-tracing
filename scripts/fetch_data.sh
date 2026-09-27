#!/usr/bin/env bash
# Download the public datasets into data/raw/ and verify them by sha256.
#
#   scripts/fetch_data.sh            # both datasets
#   scripts/fetch_data.sh assist09   # just one
#
# Both hosts throttle single connections hard (single-digit KB/s here), so each
# file is fetched as parallel byte ranges and stitched back together; an
# interrupted run resumes the ranges it already has.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

# dataset | file | url | bytes | sha256
# assist09: the ORIGINAL 2009-10 skill-builder release (duplicates included - we
# measure their effect) plus the publisher's corrected+collapsed release, which
# is only used to cross-check our own de-duplication.
DRIVE="https://drive.usercontent.google.com/download?export=download&confirm=t"
FILES=(
  "assist09|skill_builder_data_original.csv|$DRIVE&id=0B2X0QD6q79ZJUFU1cjYtdGhVNjg&resourcekey=0-OyI8ZWxtGSAzhodUIcMf_g|83201940|f22e3fb7872c1784ce93b0f9ebabbe0cbcac4f896fd8b4a11667b9715d77dbdc"
  "assist09|skill_builder_data_corrected_collapsed.csv|$DRIVE&id=1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE|64412812|162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da"
  "algebra05|algebra_2005_2006.zip|http://base.ustc.edu.cn/data/KDD_Cup_2010/algebra_2005_2006.zip|22438892|SHA_ALGEBRA"
)

fetch() {  # out url size sha
  local out=$1 url=$2 size=$3 sha=$4 chunk=2097152 par=32
  local dest="data/raw/$out" dir="data/raw/$out.parts"
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "ok   $out (already present)"; return 0
  fi
  mkdir -p "$dir"
  local n=$(( (size + chunk - 1) / chunk ))
  export url size chunk dir
  seq 0 $((n - 1)) | xargs -P "$par" -I{} bash -c '
    i={}; s=$(( i * chunk )); e=$(( s + chunk - 1 )); [ $e -ge $size ] && e=$(( size - 1 ))
    want=$(( e - s + 1 )); f="$dir/$(printf %06d $i)"
    for try in 1 2 3 4 5 6 7 8; do
      have=0; [ -f "$f" ] && have=$(stat -c%s "$f")
      [ "$have" -eq "$want" ] && exit 0
      if [ "$have" -gt 0 ] && [ "$have" -lt "$want" ]; then
        curl -s -m 600 -r $((s + have))-$e "$url" >> "$f" || true
      else
        rm -f "$f"; curl -s -m 600 -r $s-$e -o "$f" "$url" || true
      fi
    done
    echo "chunk $i incomplete after retries" >&2; exit 1'
  cat "$dir"/* > "$dest"
  local got; got=$(sha256sum "$dest" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then
    echo "FAIL $out: sha256 $got, expected $sha" >&2; exit 1
  fi
  rm -rf "$dir"
  echo "ok   $out"
}

wanted="${*:-assist09 algebra05}"
for w in $wanted; do
  case "$w" in assist09|algebra05) ;; *) echo "unknown dataset $w (assist09, algebra05)" >&2; exit 2;; esac
done
for entry in "${FILES[@]}"; do
  IFS='|' read -r ds out url size sha <<< "$entry"
  case " $wanted " in *" $ds "*) fetch "$out" "$url" "$size" "$sha" ;; esac
done
