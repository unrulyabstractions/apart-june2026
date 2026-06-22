#!/usr/bin/env bash
# Poll the 16 geom_multi box logs; emit terminal-state transitions; exit when all done.
FLEET_DIR="$PWD/cloud/.fleet_geom_multi"
TOTAL=16
prev=""
for _ in $(seq 1 360); do  # up to ~3h at 30s
  done_n=0; lines=""
  for lg in "$FLEET_DIR"/*.log; do
    [ -e "$lg" ] || continue
    tag="$(basename "$lg" .log)"
    if grep -q "DONE + destroyed" "$lg" 2>/dev/null; then
      done_n=$((done_n+1)); st="DONE"
    elif grep -qE "never became running|SSH never came up|sync_up/at_setup failed|FATAL" "$lg" 2>/dev/null; then
      done_n=$((done_n+1)); st="FAILED"
    else
      continue
    fi
    lines="$lines$tag:$st\n"
  done
  # Emit only newly-terminal tags
  cur="$(printf "%b" "$lines" | sort)"
  newly="$(comm -13 <(printf "%s" "$prev") <(printf "%s" "$cur") 2>/dev/null)"
  [ -n "$newly" ] && printf "%s\n" "$newly"
  prev="$cur"
  if [ "$done_n" -ge "$TOTAL" ]; then
    echo "ALL_DONE $done_n/$TOTAL boxes terminal"
    exit 0
  fi
  sleep 30
done
echo "WATCH_TIMEOUT $done_n/$TOTAL terminal after 3h"
