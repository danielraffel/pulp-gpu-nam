#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
CASES=[(b,l) for b in (32,64,128) for l in (1,2,4,8)]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('executable'); ap.add_argument('output'); ap.add_argument('--model',default=None); ap.add_argument('--transport-sha',default=None); a=ap.parse_args()
 exe=Path(a.executable).resolve(); out=Path(a.output).resolve()
 if not exe.is_absolute() or not exe.is_file() or not os.access(exe,os.X_OK): return 2
 if out.exists() and any(out.iterdir()): print('output must be empty or absent',file=sys.stderr); return 2
 out.mkdir(parents=True,exist_ok=True)
 model=Path(a.model).resolve() if a.model else Path(__file__).resolve().parents[1]/'src/models/example.nam'
 if not model.is_file(): print('required model missing',file=sys.stderr); return 2
 before=sha(exe); rows=[]; unavailable=False
 for b,l in CASES:
  log=out/f'matrix-{b}-{l}.log'
  try: p=subprocess.run([str(exe),f'--block-size={b}',f'--lead-blocks={l}',f'--model-path={model}'],text=True,capture_output=True,timeout=30,cwd=Path(__file__).resolve().parents[1])
  except subprocess.TimeoutExpired as e: log.write_text((e.stdout or '')+(e.stderr or '')+'\ntimeout\n'); rows.append({'block_size':b,'lead_blocks':l,'exit_code':124,'status':'timeout','log':str(log),'log_sha256':sha(log)}); continue
  log.write_text(p.stdout+p.stderr)
  status=next((x for x in reversed((p.stdout+p.stderr).splitlines()) if x.startswith('diagnostic_status=')),'missing')
  unavailable |= status=='diagnostic_status=provider_unavailable'
  rows.append({'block_size':b,'lead_blocks':l,'exit_code':p.returncode,'status':status,'log':str(log),'log_sha256':sha(log)})
 after=sha(exe); failed=any(r['exit_code']!=0 or r['status']!='diagnostic_status=passed' for r in rows)
 rec={'executable':str(exe),'executable_sha256_before':before,'executable_sha256_after':after,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'model':str(model),'model_sha256':sha(model),'transport_overlay_object_sha256':a.transport_sha,'cases':rows,'case_count':len(rows),'cpu_process_time_baseline':'open'}
 (out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
 if before!=after or failed: return 3 if before!=after else (4 if unavailable else 1)
 return 0
if __name__=='__main__': sys.exit(main())
