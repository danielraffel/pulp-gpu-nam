#!/usr/bin/env python3
import json,os,tempfile,subprocess,unittest
import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; RUN=HERE/'run_gpu_nam_matrix.sh'
MODEL=HERE.parent/'src/models/example.nam'
class T(unittest.TestCase):
 def fake(self,body):
  d=Path(tempfile.mkdtemp()); p=d/'fake'; p.write_text('#!/bin/sh\n'+body); p.chmod(0o755); return d,p
 def runr(self,p,out,model=MODEL): return subprocess.run([str(RUN),str(p),str(out),'--model',str(model)],capture_output=True,text=True)
 def test_all_pass(self):
  d,p=self.fake('echo diagnostic_status=passed\n'); r=self.runr(p,d/'o'); self.assertEqual(r.returncode,0); self.assertEqual(len(json.loads((d/'o/receipt.json').read_text())['cases']),12)
 def test_nonzero(self):
  d,p=self.fake('echo diagnostic_status=failed\nexit 7\n'); r=self.runr(p,d/'o'); self.assertNotEqual(r.returncode,0); self.assertTrue((d/'o/receipt.json').exists())
 def test_unavailable(self):
  d,p=self.fake('echo diagnostic_status=provider_unavailable\nexit 2\n'); r=self.runr(p,d/'o'); self.assertNotEqual(r.returncode,0)
 def test_nonempty(self):
  d,p=self.fake('echo diagnostic_status=passed\n'); o=d/'o'; o.mkdir(); (o/'keep').write_text('x'); r=self.runr(p,o); self.assertNotEqual(r.returncode,0); self.assertEqual((o/'keep').read_text(),'x')
 def test_missing_model(self):
  d,p=self.fake('echo diagnostic_status=passed\n'); r=self.runr(p,d/'o',d/'missing'); self.assertNotEqual(r.returncode,0)
 def test_mutation(self):
  d,p=self.fake('echo diagnostic_status=passed\ncat >> "$0" <<EOF\n#mutated\nEOF\n'); r=self.runr(p,d/'o'); self.assertNotEqual(r.returncode,0)
 def test_timeout_bytes_writes_receipt(self):
  d,p=self.fake('echo diagnostic_status=passed\n'); out=d/'o'
  spec=importlib.util.spec_from_file_location('runner',HERE/'run_gpu_nam_matrix.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  old=m.subprocess.run
  def fake_run(cmd,*args,**kwargs):
   if Path(cmd[0]).name == p.name: raise subprocess.TimeoutExpired(cmd,1,output=b'partial',stderr=b'err')
   return old(cmd,*args,**kwargs)
  m.subprocess.run=fake_run; old_argv=sys.argv; sys.argv=['runner',str(p),str(out),'--model',str(MODEL)]
  try: rc=m.main()
  finally: sys.argv=old_argv; m.subprocess.run=old
  self.assertNotEqual(rc,0); rec=json.loads((out/'receipt.json').read_text()); self.assertEqual(rec['cases'][0]['exit_code'],124); self.assertIn('partialerr',(out/'matrix-32-1.log').read_text())
 def test_model_path_forwarded_and_cwd_independent(self):
  d,p=self.fake('echo "$@" >> "$FAKE_ARGS"\necho diagnostic_status=passed\n'); args=d/'args'; os.environ['FAKE_ARGS']=str(args)
  try:
   r=subprocess.run([str(RUN),str(p),str(d/'o'),'--model',str(MODEL)],cwd='/',capture_output=True,text=True)
  finally: os.environ.pop('FAKE_ARGS',None)
  self.assertEqual(r.returncode,0); text=args.read_text(); self.assertIn('--model-path='+str(MODEL.resolve()),text)
  rec=json.loads((d/'o/receipt.json').read_text()); self.assertEqual(rec['source_commit'],subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parent,text=True).strip()); self.assertEqual(rec['model_sha256'],__import__('hashlib').sha256(MODEL.read_bytes()).hexdigest())
 def test_stamped_options_forwarded(self):
  d,p=self.fake('echo "$@" >> "$FAKE_ARGS"\necho diagnostic_status=passed\n'); args=d/'args'; env=dict(os.environ,FAKE_ARGS=str(args))
  r=subprocess.run([str(RUN),str(p),str(d/'o'),'--model',str(MODEL),'--stamped','--completion-policy=timed-wait-any','--worker-wait-ns=100000'],env=env,capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr); lines=args.read_text().splitlines(); self.assertEqual(len(lines),12)
  self.assertIn('1 32 '+str(MODEL.resolve()),lines[0]); self.assertIn('--completion-policy=timed-wait-any --worker-wait-ns=100000',lines[0])
  rec=json.loads((d/'o/receipt.json').read_text()); self.assertEqual(rec['completion_policy'],'timed-wait-any'); self.assertEqual(rec['worker_wait_ns'],100000)
 def test_completion_options_rejected_before_execution(self):
  for options in (['--completion-policy=wait-any'],['--stamped','--worker-wait-ns=1'],['--stamped','--completion-policy=wait-any','--worker-wait-ns=1'],['--stamped','--completion-policy=timed-wait-any','--worker-wait-ns=1000001'],['--stamped','--worker-wait-ns=-1'],['--stamped','--completion-policy=invalid']):
   with self.subTest(options=options):
    d,p=self.fake('exit 99\n')
    r=subprocess.run([str(RUN),str(p),str(d/'o'),*options],capture_output=True,text=True)
    self.assertEqual(r.returncode,2); self.assertFalse((d/'o').exists())
 def test_paced_sidecar_required(self):
  d,p=self.fake('echo diagnostic_status=passed\n')
  r=subprocess.run([str(RUN),str(p),str(d/'o'),'--stamped','--paced','--blocks=2'],capture_output=True,text=True)
  self.assertNotEqual(r.returncode,0); rec=json.loads((d/'o/receipt.json').read_text()); self.assertEqual(rec['cases'][0]['status'],'missing_sidecar')
 def test_paced_flags_and_sidecar_bound(self):
  d,p=self.fake('for a in "$@"; do case "$a" in --sidecar=*) f=${a#--sidecar=};; esac; done\necho block,delivered_input_sequence,scheduled_ns,start_ns,end_ns,deadline_ns,selected > "$f"\ni=0; n=$((3+$1)); while [ $i -lt $n ]; do if [ $i -lt $1 ]; then delivered=; selected=priming; else delivered=$((i-$1)); selected=cpu_baseline; fi; echo "$i,$delivered,$i,$i,$i,$((i+1)),$selected" >> "$f"; i=$((i+1)); done\necho diagnostic_status=passed\n')
  r=subprocess.run([str(RUN),str(p),str(d/'o'),'--stamped','--paced','--cpu-baseline','--blocks=3'],capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr); rec=json.loads((d/'o/receipt.json').read_text()); self.assertTrue(rec['paced']); self.assertTrue(rec['cpu_baseline']); self.assertEqual(rec['input_blocks'],3); self.assertEqual(len(rec['cases'][0]['sidecar_sha256']),64)
 def test_invalid_paced_flags(self):
  for opts in (['--paced'],['--stamped','--cpu-baseline'],['--stamped','--paced','--blocks=0'],['--stamped','--paced','--blocks=1000001'],['--stamped','--paced','--blocks=3','--duration-seconds=2']):
   d,p=self.fake('exit 99\n');r=subprocess.run([str(RUN),str(p),str(d/'o'),*opts],capture_output=True,text=True)
   self.assertEqual(r.returncode,2); self.assertFalse((d/'o').exists())
 def test_incomplete_or_out_of_order_sidecar_rejected(self):
  spec=importlib.util.spec_from_file_location('runner_sidecar',HERE/'run_gpu_nam_matrix.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  d=Path(tempfile.mkdtemp()); p=d/'rows.csv'; header='block,delivered_input_sequence,scheduled_ns,start_ns,end_ns,deadline_ns,selected\n'
  p.write_text(header+'0,,10,11,12,20,priming\n1,0,20,21,22,30,gpu_delivered\n')
  self.assertTrue(m.complete_sidecar(p,2,1)); self.assertFalse(m.complete_sidecar(p,3,1))
  p.write_text(header+'0,,10,11,12,20,priming\n0,0,20,21,22,30,gpu_delivered\n'); self.assertFalse(m.complete_sidecar(p,2,1))
  p.write_text(header+'0,,10,11,12,20,priming\n1,1,20,21,22,30,gpu_delivered\n'); self.assertFalse(m.complete_sidecar(p,2,1))
  p.write_text(header+'0,0,10,11,12,20,priming\n1,0,20,21,22,30,gpu_delivered\n'); self.assertFalse(m.complete_sidecar(p,2,1))
 def test_exact_transport_dispositions_are_preserved(self):
  spec=importlib.util.spec_from_file_location('runner_exact_selection',HERE/'run_gpu_nam_matrix.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'rows.csv'; header='block,delivered_input_sequence,scheduled_ns,start_ns,end_ns,deadline_ns,selected\n'
   for selected in ('gpu_delivered','worker_output','cpu_fallback','silence','passthrough','invalid_rejected','cpu_baseline'):
    p.write_text(header+'0,,10,11,12,20,priming\n1,0,20,21,22,30,'+selected+'\n')
    self.assertTrue(m.complete_sidecar(p,2,1),selected)
   for selected in ('accounting_error','unknown','gpu_forward_failed'):
    p.write_text(header+'0,,10,11,12,20,priming\n1,0,20,21,22,30,'+selected+'\n')
    self.assertFalse(m.complete_sidecar(p,2,1),selected)
if __name__=='__main__': unittest.main()
