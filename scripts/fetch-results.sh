#!/usr/bin/env bash
#
# Download real historical match results (with closing odds) from football-data.co.uk and
# concatenate them into one CSV for `wc-oracle backtest --data`. The data itself is not
# committed (it is football-data.co.uk's, and large); this script reproduces it.
#
#   bash scripts/fetch-results.sh [output.csv]
#   LEAGUE=SP1 bash scripts/fetch-results.sh   # other leagues: E0, D1, SP1, I1, F1, ...
#
# football-data.co.uk added betting columns over the years, so column positions differ
# between old and new seasons. We therefore key the combined file to the NEWEST season's
# header and include only seasons whose header matches it, written oldest-first so the
# backtest's row-order time split stays chronological.
set -euo pipefail

OUT="${1:-/tmp/real_results.csv}"
LEAGUE="${LEAGUE:-E0}"
# Oldest -> newest. Add more if you want a bigger sample.
SEASONS=(1617 1718 1819 1920 2021 2122 2223 2324)

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for s in "${SEASONS[@]}"; do
  url="https://www.football-data.co.uk/mmz4281/${s}/${LEAGUE}.csv"
  if curl -fsS -m 40 "$url" | tr -d '\r' > "$tmp/$s.csv"; then
    :
  else
    echo "skip ${s}: not available" >&2
    rm -f "$tmp/$s.csv"
  fi
done

# Reference header = the newest season we successfully downloaded.
ref=""
for ((i = ${#SEASONS[@]} - 1; i >= 0; i--)); do
  f="$tmp/${SEASONS[$i]}.csv"
  [ -s "$f" ] && { ref="$(head -1 "$f")"; break; }
done
[ -n "$ref" ] || { echo "no seasons downloaded" >&2; exit 1; }

printf '%s\n' "$ref" > "$OUT"
included=()
for s in "${SEASONS[@]}"; do            # oldest -> newest
  f="$tmp/$s.csv"
  [ -s "$f" ] || continue
  if [ "$(head -1 "$f")" = "$ref" ]; then
    tail -n +2 "$f" | sed '/^[[:space:]]*$/d' >> "$OUT"
    included+=("$s")
  else
    echo "skip ${s}: header differs from reference schema" >&2
  fi
done

echo "included seasons: ${included[*]}"
echo "wrote $(($(wc -l < "$OUT") - 1)) matches to $OUT"
