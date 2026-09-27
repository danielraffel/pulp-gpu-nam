#!/usr/bin/env python3
"""Manual silent-device child with an external whole-process watchdog."""
import argparse,hashlib,json,os,pathlib,signal,subprocess,sys
SDK='7858d7edb5a2bbb568c03d7e740043e8c897c0ee'
def run_child(command,timeout,log):
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:return process.wait(timeout=timeout),False
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()
        return process.returncode,True

def child_result(code,timed_out,receipt):
    if code!=0 or timed_out:return False,None
    try:child=json.loads(receipt.read_text())
    except (OSError,ValueError):return False,None
    if not isinstance(child,dict):return False,child
    passed=child.get('schema')=='gpu-nam.physical-lifecycle.v1' and child.get('epochs')==2 and all(
        child.get(k) is True for k in ['callback_owner_drain','stop_restart_close','plugin_destroy_deinit_dlclose'])
    passed=passed and child.get('physical_output')=='silence'
    return passed,child

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable',required=True,type=pathlib.Path);p.add_argument('--clap',required=True,type=pathlib.Path)
    p.add_argument('--sdk-provenance',required=True,type=pathlib.Path);p.add_argument('--device-id',required=True)
    p.add_argument('--mode',required=True,choices=['cpu','shared']);p.add_argument('--output-dir',required=True,type=pathlib.Path)
    p.add_argument('--timeout',type=float,default=45)
    a=p.parse_args();assert 1<=a.timeout<=120
    provenance=json.loads(a.sdk_provenance.read_text());assert provenance['source_git_sha']==SDK
    for f in [a.executable,a.clap]:assert f.is_file()
    # Exclusive fresh directory prevents an old success receipt satisfying this run.
    a.output_dir.mkdir(parents=True,exist_ok=False)
    prefix=a.output_dir/'capture';command=[str(a.executable.resolve()),str(a.clap.resolve()),a.device_id,a.mode,str(prefix.resolve()),'--normal-lifecycle']
    with (a.output_dir/'child.log').open('w') as log:code,timed_out=run_child(command,a.timeout,log)
    evidence={'schema':'gpu-nam.physical-lifecycle-parent.v1','command':command,'exit_code':code,'timeout':timed_out,
              'sdk_source':SDK,'mode':a.mode,'device_id':a.device_id,'passed':False,
              'inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.executable,a.clap,a.sdk_provenance]}}
    receipt=prefix.with_name(prefix.name+'-lifecycle.json')
    evidence['passed'],child=child_result(code,timed_out,receipt)
    if child is not None:
        evidence['child']=child
        evidence['passed']=evidence['passed'] and child.get('sdk_sha')==SDK and child.get('mode')==a.mode and child.get('device_id')==a.device_id
        source=child.get('source_sha','')
        evidence['passed']=evidence['passed'] and len(source)==40 and all(c in '0123456789abcdef' for c in source)
    stage=prefix.with_name(prefix.name+'-stage.txt')
    evidence['last_child_stage']=stage.read_text().strip() if stage.is_file() else 'unavailable'

    for epoch in range(2):
        for suffix in ('callbacks.csv','audio.csv','metadata.txt'):
            if not (a.output_dir/f'capture-epoch{epoch}-{suffix}').is_file():evidence['passed']=False
        meta=a.output_dir/f'capture-epoch{epoch}-metadata.txt'
        if meta.is_file():
            fields=dict(line.split('=',1) for line in meta.read_text().splitlines() if '=' in line)
            evidence.setdefault('epoch_metadata',[]).append(fields)
            evidence['passed']=evidence['passed'] and fields.get('engine')==a.mode and fields.get('device_id')==a.device_id and fields.get('epoch')==str(epoch)
            evidence['passed']=evidence['passed'] and fields.get('callback_entries')==fields.get('callback_exits') and int(fields.get('callback_entries','0'))>0
    evidence['outputs']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output_dir.iterdir() if f.is_file()}
    (a.output_dir/'receipt.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return 0 if evidence['passed'] else 1
if __name__=='__main__':sys.exit(main())
