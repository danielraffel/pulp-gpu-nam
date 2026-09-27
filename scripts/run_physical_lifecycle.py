#!/usr/bin/env python3
"""Manual silent-device child with an external whole-process watchdog."""
import argparse,hashlib,json,math,os,pathlib,signal,subprocess,sys
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
    if not isinstance(child,dict):return False,None
    passed=child.get('schema')=='gpu-nam.physical-lifecycle.v1' and child.get('epochs')==2 and all(
        child.get(k) is True for k in ['callback_owner_drain','stop_restart_close','plugin_destroy_deinit_dlclose'])
    passed=passed and child.get('physical_output')=='silence'
    return passed,child

def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def epoch_result(path,mode,device,epoch,duration=None):
    try:
        pairs=[line.split('=',1) for line in path.read_text().splitlines() if '=' in line]
        fields=dict(pairs)
        if len(fields)!=len(pairs):return False,fields
        entries=int(fields.get('callback_entries','0'));exits=int(fields.get('callback_exits','-1'))
        valid=fields.get('engine')==mode and fields.get('device_id')==device and fields.get('epoch')==str(epoch)
        if duration is not None:
            target=((duration*48000+511)//512)*512
            valid=valid and fields.get('requested_duration_seconds')==str(duration) and fields.get('target_frames')==str(target)
            valid=valid and fields.get('frames_captured')==str(target) and fields.get('frames_compared')==str(target)
            valid=valid and fields.get('row_overflow')=='0' and fields.get('frame_overflow')=='0'
        return valid and entries==exits and entries>0,fields
    except (OSError,ValueError,UnicodeError):return False,None

def validate_timing(duration,timeout):
    if duration is not None and (type(duration) is not int or not 1<=duration<=60):
        raise ValueError("duration must be an integer from 1 through 60 seconds")
    minimum=2*duration+15 if duration is not None else 1
    if not math.isfinite(timeout) or not minimum<=timeout<=150:
        raise ValueError(f"timeout must be finite, at least {minimum}s and at most 150s")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable',required=True,type=pathlib.Path);p.add_argument('--clap',required=True,type=pathlib.Path)
    p.add_argument('--expected-product-sha',required=True);p.add_argument('--sdk-provenance',required=True,type=pathlib.Path);p.add_argument('--device-id',required=True)
    p.add_argument('--mode',required=True,choices=['cpu','shared']);p.add_argument('--output-dir',required=True,type=pathlib.Path)
    p.add_argument('--duration-seconds',type=int,default=None,help='Per epoch, 1..60; omitted preserves 96256-frame capture')
    p.add_argument('--timeout',type=float,default=None)
    a=p.parse_args()
    if a.timeout is None:a.timeout=max(45,2*a.duration_seconds+15) if a.duration_seconds is not None else 45
    try:validate_timing(a.duration_seconds,a.timeout)
    except ValueError as error:p.error(str(error))
    assert len(a.expected_product_sha)==40 and all(c in '0123456789abcdef' for c in a.expected_product_sha)
    provenance=json.loads(a.sdk_provenance.read_text());assert provenance['source_git_sha']==SDK
    for f in [a.executable,a.clap]:assert f.is_file()
    # Exclusive fresh directory prevents an old success receipt satisfying this run.
    a.output_dir.mkdir(parents=True,exist_ok=False)
    before_hashes={str(f):file_hash(f) for f in [a.executable,a.clap,a.sdk_provenance]}
    prefix=a.output_dir/'capture';command=[str(a.executable.resolve()),str(a.clap.resolve()),a.device_id,a.mode,str(prefix.resolve()),'--normal-lifecycle']
    if a.duration_seconds is not None:command+=['--duration-seconds',str(a.duration_seconds)]
    with (a.output_dir/'child.log').open('w') as log:code,timed_out=run_child(command,a.timeout,log)
    evidence={'schema':'gpu-nam.physical-lifecycle-parent.v1','command':command,'exit_code':code,'timeout':timed_out,
              'duration_seconds_per_epoch':a.duration_seconds,'watchdog_seconds':a.timeout,
              'sdk_source':SDK,'mode':a.mode,'device_id':a.device_id,'passed':False,
              'inputs_before':before_hashes,'expected_product_sha':a.expected_product_sha}
    receipt=prefix.with_name(prefix.name+'-lifecycle.json')
    evidence['passed'],child=child_result(code,timed_out,receipt)
    if child is not None:
        evidence['child']=child
        evidence['passed']=evidence['passed'] and child.get('sdk_sha')==SDK and child.get('mode')==a.mode and child.get('device_id')==a.device_id
        source=child.get('source_sha','')
        evidence['passed']=evidence['passed'] and source==a.expected_product_sha
    stage=prefix.with_name(prefix.name+'-stage.txt')
    evidence['last_child_stage']=stage.read_text().strip() if stage.is_file() else 'unavailable'

    for epoch in range(2):
        for suffix in ('callbacks.csv','audio.csv','metadata.txt'):
            if not (a.output_dir/f'capture-epoch{epoch}-{suffix}').is_file():evidence['passed']=False
        meta=a.output_dir/f'capture-epoch{epoch}-metadata.txt'
        valid,fields=epoch_result(meta,a.mode,a.device_id,epoch,a.duration_seconds)
        evidence.setdefault('epoch_metadata',[]).append(fields)
        evidence['passed']=evidence['passed'] and valid
    try:evidence['inputs_after']={str(f):file_hash(f) for f in [a.executable,a.clap,a.sdk_provenance]}
    except OSError:evidence['inputs_after']={}
    evidence['inputs_unchanged']=evidence['inputs_after']==before_hashes
    evidence['passed']=evidence['passed'] and evidence['inputs_unchanged']
    evidence['outputs']={f.name:file_hash(f) for f in a.output_dir.iterdir() if f.is_file()}
    (a.output_dir/'receipt.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return 0 if evidence['passed'] else 1
if __name__=='__main__':sys.exit(main())
