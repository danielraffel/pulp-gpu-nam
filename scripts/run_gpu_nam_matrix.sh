#!/bin/sh
set -eu
EXE=${1:?usage: $0 /absolute/path/to/gpu-nam-gpu-cpu-diagnostic [output-dir]}
OUT=${2:-gpu-nam-matrix-receipt}
mkdir -p "$OUT"
case "$EXE" in /*) ;; *) echo 'executable must be absolute' >&2; exit 2;; esac
[ -x "$EXE" ] || { echo "not executable: $EXE" >&2; exit 2; }
sha256=$(shasum -a 256 "$EXE" | awk '{print $1}')
for b in 32 64 128; do for l in 1 2 4 8; do
  log="$OUT/matrix-${b}-${l}.log"
  set +e
  "$EXE" --block-size="$b" --lead-blocks="$l" >"$log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "{\"block_size\":$b,\"lead_blocks\":$l,\"exit_code\":$rc,\"log\":\"$log\"}" >> "$OUT/results.ndjson"
done; done
MODEL=${GPU_NAM_MODEL_PATH:-src/models/example.nam}
python3 - "$OUT" "$EXE" "$sha256" "$MODEL" "$(git rev-parse HEAD)" <<'PY'
import json,sys,glob,hashlib,os
out,exe,exe_sha,model,source_commit=sys.argv[1:]
rows=[json.loads(x) for x in open(os.path.join(out,'results.ndjson'))]
for r in rows:
 p=r['log']; r['log_sha256']=hashlib.sha256(open(p,'rb').read()).hexdigest(); r['status']=open(p).read().strip().splitlines()[-1]
mh=hashlib.sha256(open(model,'rb').read()).hexdigest() if os.path.exists(model) else None
receipt={'executable':exe,'executable_sha256':exe_sha,'source_commit':source_commit,'model':model,'model_sha256':mh,'transport_overlay_object_sha256':os.environ.get('GPU_NAM_TRANSPORT_OBJECT_SHA256'),'cases':rows,'cpu_process_time_baseline':'open: diagnostic records callback/fallback wall time; process CPU and CPU-only baseline remain to be added'}
open(os.path.join(out,'receipt.json'),'w').write(json.dumps(receipt,indent=2)+'\n')
PY
