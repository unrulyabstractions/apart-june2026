#!/usr/bin/env bash
# Billing safety net: destroy + VERIFY-gone every instance id recorded in
# .pending_destroy (boxes whose in-trap destroy could not be confirmed because the
# vast API was unreachable). Run whenever the API is reachable again. Ids confirmed
# gone are dropped; any that can't be verified stay in the file for the next run.
set -u; HERE="$(cd "$(dirname "$0")" && pwd)"; PEND="$HERE/.pending_destroy"
[ -s "$PEND" ] || { echo "[reap] nothing pending"; exit 0; }
remaining=""
while read -r id; do
  [ -z "$id" ] && continue
  echo "[reap] destroying $id"; printf 'y\n' | vastai destroy instance "$id" 2>&1 | head -2
  sleep 4
  gone="$(ID="$id" python3 -c '
import json,os,subprocess,sys
iid=int(os.environ["ID"])
p=subprocess.run(["vastai","show","instances-v1","--raw"],capture_output=True,text=True)
try: d=json.loads(p.stdout)
except Exception: print("unknown"); sys.exit()
rows=d if isinstance(d,list) else d.get("instances",d.get("results",[]))
print("yes" if any(isinstance(r,dict) and r.get("id")==iid for r in rows) else "no")')"
  if [ "$gone" = "no" ]; then echo "[reap] $id confirmed gone";
  else echo "[reap] $id still pending ($gone)"; remaining="$remaining$id\n"; fi
done < "$PEND"
printf "%b" "$remaining" | sed '/^$/d' > "$PEND"
echo "[reap] $(grep -c . "$PEND" 2>/dev/null || echo 0) ids still pending"
