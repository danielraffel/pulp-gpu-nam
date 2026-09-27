import pathlib,sys,tempfile,json
from run_physical_lifecycle import run_child,child_result,epoch_result,file_hash
checks=0
def check(v):
    global checks
    checks+=1
    assert v
with tempfile.TemporaryDirectory() as temporary:
    root=pathlib.Path(temporary)
    with (root/'child.log').open('w') as log:
        check(run_child([sys.executable,'-c','pass'],2,log)==(0,False))
        check(run_child([sys.executable,'-c','raise SystemExit(78)'],2,log)==(78,False))
        code,timeout=run_child([sys.executable,'-c','import time;time.sleep(30)'],.05,log)
        check(timeout and code<0)
    receipt=root/'child.json'
    good={'schema':'gpu-nam.physical-lifecycle.v1','epochs':2,'callback_owner_drain':True,'stop_restart_close':True,'plugin_destroy_deinit_dlclose':True,'physical_output':'silence'}
    check(child_result(0,False,receipt)[0] is False)
    for malformed in ['', '{', 'null', '[]']:
        receipt.write_text(malformed);check(child_result(0,False,receipt)==(False,None))
    receipt.write_text(json.dumps(good));check(child_result(0,False,receipt)[0])
    check(not child_result(78,False,receipt)[0]);check(not child_result(0,True,receipt)[0])
    for k in good:
        bad=dict(good);bad.pop(k);receipt.write_text(json.dumps(bad));check(not child_result(0,False,receipt)[0])
    meta=root/'metadata.txt'
    good_meta='engine=shared\ndevice_id=42\nepoch=0\ncallback_entries=8\ncallback_exits=8\n'
    meta.write_text(good_meta);check(epoch_result(meta,'shared','42',0)[0])
    for text in [good_meta.replace('callback_exits=8','callback_exits=7'),good_meta.replace('callback_entries=8','callback_entries=oops'),good_meta+'engine=cpu\n',good_meta.replace('device_id=42','device_id=43'),good_meta.replace('epoch=0','epoch=1'),'']:
        meta.write_text(text);check(not epoch_result(meta,'shared','42',0)[0])
    before=file_hash(meta);meta.write_text('changed');check(file_hash(meta)!=before)
print(f'{checks} watchdog and receipt controls passed')
