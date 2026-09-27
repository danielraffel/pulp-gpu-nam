import pathlib,sys,tempfile,json
from run_physical_lifecycle import run_child,child_result
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
        receipt.write_text(malformed);check(child_result(0,False,receipt)[0] is False)
    receipt.write_text(json.dumps(good));check(child_result(0,False,receipt)[0])
    check(not child_result(78,False,receipt)[0]);check(not child_result(0,True,receipt)[0])
    for k in good:
        bad=dict(good);bad.pop(k);receipt.write_text(json.dumps(bad));check(not child_result(0,False,receipt)[0])
print(f'{checks} watchdog and receipt controls passed')
