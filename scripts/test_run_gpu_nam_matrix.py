#!/usr/bin/env python3
import json,os,tempfile,subprocess,unittest
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
if __name__=='__main__': unittest.main()
