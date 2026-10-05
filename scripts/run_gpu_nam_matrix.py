#!/usr/bin/env python3
import argparse,csv,hashlib,json,math,os,subprocess,sys,tempfile
from pathlib import Path
CASES=[(b,l) for b in (32,64,128) for l in (1,2,4,8)]
def cases_for_capacity(capacity):
 return [(b,l) for b,l in CASES if capacity is None or l < capacity]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def complete_sidecar(path, expected, lead):
 try:
  count=0; previous=-1
  with path.open(newline='') as stream:
   for row in csv.DictReader(stream):
    scheduled=int(row['scheduled_ns']); start=int(row['start_ns']); end=int(row['end_ns']); deadline=int(row['deadline_ns'])
    if int(row['block'])!=count or scheduled<=previous or deadline<=scheduled or end<start or row['selected'] not in ('priming','cpu_baseline','gpu_delivered','worker_output','cpu_fallback','silence','passthrough','invalid_rejected'): return False
    delivered=row['delivered_input_sequence']
    if count<lead:
     if delivered!='' or row['selected']!='priming': return False
    elif delivered!=str(count-lead) or row['selected']=='priming': return False
    previous=scheduled; count+=1
  return count==expected
 except (OSError,ValueError,KeyError,TypeError): return False

def sidecar_summary(path):
 values=[]; misses=0; dispositions={}
 with path.open(newline='') as stream:
  for row in csv.DictReader(stream):
   values.append(int(row['start_lateness_ns']))
   misses += int(row['deadline_missed'])
   selected=row['selected']; dispositions[selected]=dispositions.get(selected,0)+1
 if not values:
  return {'measured_blocks':0,'deadline_misses':0,'deadline_miss_rate':None,
          'start_lateness_ns':{},'terminal_dispositions':dispositions}
 values.sort()
 def nearest_rank(q):
  return values[max(0, min(len(values)-1, math.ceil(q*len(values))-1))]
 return {
  'measured_blocks':len(values),
  'deadline_misses':misses,
  'deadline_miss_rate':misses/len(values),
  'start_lateness_ns':{
   'p50':nearest_rank(.50), 'p99':nearest_rank(.99),
   'p99.9':nearest_rank(.999), 'p99.99':nearest_rank(.9999),
   'max':values[-1]},
  'terminal_dispositions':dispositions,
 }
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('executable'); ap.add_argument('output'); ap.add_argument('--model',default=None); ap.add_argument('--transport-sha',default=None)
 ap.add_argument('--stamped',action='store_true',help='run the stamped correctness consumer')
 ap.add_argument('--completion-policy',choices=('process-events','wait-any','timed-wait-any'),default=None)
 ap.add_argument('--worker-wait-ns',type=int,default=0)
 ap.add_argument('--paced',action='store_true')
 ap.add_argument('--cpu-baseline',action='store_true')
 ap.add_argument('--duration-seconds',type=int,default=None)
 ap.add_argument('--blocks',type=int,default=None)
 ap.add_argument('--capacity',type=int,choices=(2,4,8,16),default=None,
                 help='stamped transport slot capacity (default: executable default)')
 a=ap.parse_args()
 if (a.paced or a.cpu_baseline or a.duration_seconds is not None or a.blocks is not None) and not a.stamped: ap.error('paced options require --stamped')
 if (a.cpu_baseline or a.duration_seconds is not None or a.blocks is not None) and not a.paced: ap.error('paced settings require --paced')
 if a.duration_seconds is not None and a.blocks is not None: ap.error('choose duration or blocks')
 if a.duration_seconds is not None and not 1 <= a.duration_seconds <= 666: ap.error('duration must be 1..666 seconds')
 if a.blocks is not None and not 1 <= a.blocks <= 1000000: ap.error('blocks must be 1..1000000')
 if (a.completion_policy is not None or a.worker_wait_ns != 0) and not a.stamped: ap.error('completion options require --stamped')
 policy=a.completion_policy or 'process-events'
 if a.cpu_baseline and (policy!='process-events' or a.worker_wait_ns): ap.error('CPU baseline has no completion service')
 if not 0 <= a.worker_wait_ns <= 1000000 or (a.worker_wait_ns and policy != 'timed-wait-any'): ap.error('positive worker wait requires timed-wait-any; valid range is 0..1000000ns')
 exe=Path(a.executable).resolve(); out=Path(a.output).resolve()
 if not exe.is_absolute() or not exe.is_file() or not os.access(exe,os.X_OK): return 2
 if out.exists() and any(out.iterdir()): print('output must be empty or absent',file=sys.stderr); return 2
 out.mkdir(parents=True,exist_ok=True)
 model=Path(a.model).resolve() if a.model else Path(__file__).resolve().parents[1]/'src/models/example.nam'
 if not model.is_file(): print('required model missing',file=sys.stderr); return 2
 before=sha(exe); rows=[]; unavailable=False
 for b,l in cases_for_capacity(a.capacity):
  log=out/f'matrix-{b}-{l}.log'
  command=([str(exe),str(l),str(b),str(model),f'--completion-policy={policy}',f'--worker-wait-ns={a.worker_wait_ns}'] if a.stamped else [str(exe),f'--block-size={b}',f'--lead-blocks={l}',f'--model-path={model}'])
  if a.stamped and a.capacity is not None: command.append(f'--capacity={a.capacity}')
  sidecar=out/f'matrix-{b}-{l}.csv'
  if a.paced:
   command += ['--paced',f'--sidecar={sidecar}']
   if a.cpu_baseline: command.append('--cpu-baseline')
   if a.blocks is not None: command.append(f'--blocks={a.blocks}')
   else: command.append(f'--duration-seconds={a.duration_seconds or 10}')
  duration=(a.blocks*b/48000 if a.blocks is not None else a.duration_seconds or 10) if a.paced else 0
  try: p=subprocess.run(command,text=True,capture_output=True,timeout=max(30,duration*4+60) if a.paced else 30,cwd=Path(__file__).resolve().parents[1])
  except subprocess.TimeoutExpired as e:
   so=e.stdout or ''; se=e.stderr or ''
   if isinstance(so,bytes): so=so.decode(errors='replace')
   if isinstance(se,bytes): se=se.decode(errors='replace')
   log.write_text(so+se+'\ntimeout\n'); rows.append({'block_size':b,'lead_blocks':l,'exit_code':124,'status':'timeout','log':str(log),'log_sha256':sha(log)}); continue
  log.write_text(p.stdout+p.stderr)
  status=next((x for x in reversed((p.stdout+p.stderr).splitlines()) if x.startswith('diagnostic_status=')),'missing')
  unavailable |= status=='diagnostic_status=provider_unavailable'
  row={'block_size':b,'lead_blocks':l,'exit_code':p.returncode,'status':status,'log':str(log),'log_sha256':sha(log)}
  if a.paced:
   row.update({'sidecar':str(sidecar),'sidecar_sha256':sha(sidecar) if sidecar.is_file() else None})
   expected=(a.blocks if a.blocks is not None else ((a.duration_seconds or 10)*48000+b-1)//b)+l
   if not sidecar.is_file(): row['status']='missing_sidecar'
   elif not complete_sidecar(sidecar,expected,l): row['status']='incomplete_sidecar'
   elif row['status']=='diagnostic_status=passed': row['metrics']=sidecar_summary(sidecar)
  rows.append(row)
 after=sha(exe); failed=any(r['exit_code']!=0 or r['status']!='diagnostic_status=passed' for r in rows)
 rec={'schema':'pulp.gpu_nam.matrix.v2','executable':str(exe),'executable_sha256_before':before,'executable_sha256_after':after,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'model':str(model),'model_sha256':sha(model),'transport_overlay_object_sha256':a.transport_sha,'cases':rows,'case_count':len(rows),'requested_cases':len(cases_for_capacity(a.capacity)),'cpu_process_time_baseline':'open','consumer':'stamped' if a.stamped else 'legacy-diagnostic','completion_policy':policy if a.stamped else None,'worker_wait_ns':a.worker_wait_ns if a.stamped else None,'requested_capacity':a.capacity,'effective_capacity':a.capacity if a.capacity is not None else (16 if a.stamped else None),'paced':a.paced,'cpu_baseline':a.cpu_baseline,'duration_seconds':a.duration_seconds or (10 if a.paced and a.blocks is None else None),'input_blocks':a.blocks}
 (out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
 if before!=after or failed: return 3 if before!=after else (4 if unavailable else 1)
 return 0
if __name__=='__main__': sys.exit(main())
