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
if __name__=='__main__': unittest.main()
