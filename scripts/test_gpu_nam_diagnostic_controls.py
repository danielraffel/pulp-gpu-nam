import subprocess,sys
from pathlib import Path
exe=Path(sys.argv[1]).resolve(); model=Path(sys.argv[2]).resolve()
def run(extra):
 p=subprocess.run([str(exe),f'--model-path={model}','--block-size=32','--lead-blocks=4',*extra],capture_output=True,text=True,check=False,timeout=30)
 print('control=' + ('injected' if extra else 'positive'), 'exit_code=' + str(p.returncode))
 print(p.stdout, end=''); print(p.stderr, end='', file=sys.stderr)
 f=dict(x.split('=',1) for x in p.stdout.split() if '=' in x); return p,f
p,f=run([]); assert p.returncode==0 and f['diagnostic_status']=='passed' and f['direct_model_parity_failures']=='0' and f['parity_failures']=='0'
assert f['input_blocks']=='96' and f['measured_blocks']=='96' and f['drain_blocks']=='0' and int(f['process_cpu_ticks_per_second'])>0
p,f=run(['--inject-direct-output-error']); assert p.returncode==3 and f['diagnostic_status']=='failed' and int(f['direct_model_parity_failures'])>0 and f['parity_failures']=='0'
print('diagnostic control positive/injection checks passed')
