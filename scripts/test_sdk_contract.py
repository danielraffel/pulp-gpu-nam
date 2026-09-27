#!/usr/bin/env python3
"""Configure-policy controls; no SDK or GPU is required."""
import subprocess,tempfile,unittest
from pathlib import Path
MODULE=Path(__file__).resolve().parents[1]/'cmake/GpuNamSdkContract.cmake'
SHA='a'*40
class SdkContractTests(unittest.TestCase):
 def check(self,values,ok,reason=''):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'contract.cmake'
   f.write_text(''.join(f'set({k} "{v}")\n' for k,v in values.items())+f'include("{MODULE}")\ngpu_nam_validate_sdk_contract()\n')
   p=subprocess.run(['cmake','-P',str(f)],capture_output=True,text=True)
   self.assertEqual(p.returncode==0,ok,p.stdout+p.stderr)
   if reason:self.assertIn(reason," ".join(p.stderr.split()))
 def test_default_compatibility(self):self.check({},True)
 def test_matching_source(self):self.check(dict(GPU_NAM_USE_INSTALLED_PULP='ON',GPU_NAM_EXPECTED_PULP_SOURCE_SHA=SHA,PULP_SDK_SOURCE_GIT_SHA=SHA),True)
 def test_missing_source(self):self.check(dict(GPU_NAM_USE_INSTALLED_PULP='ON',GPU_NAM_EXPECTED_PULP_SOURCE_SHA=SHA),False,'does not match')
 def test_wrong_source(self):self.check(dict(GPU_NAM_USE_INSTALLED_PULP='ON',GPU_NAM_EXPECTED_PULP_SOURCE_SHA=SHA,PULP_SDK_SOURCE_GIT_SHA='b'*40),False,'does not match')
 def test_invalid_source(self):self.check(dict(GPU_NAM_EXPECTED_PULP_SOURCE_SHA='short'),False,'full lowercase')
 def test_source_mode_refused(self):self.check(dict(GPU_NAM_EXPECTED_PULP_SOURCE_SHA=SHA),False,'installed-SDK mode')
 def test_stamped_provider_absent(self):self.check(dict(GPU_NAM_USE_INSTALLED_PULP='ON',GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET='ON'),False,'Dawn shared-I/O provider')
 def test_stamped_provider_present(self):self.check(dict(GPU_NAM_USE_INSTALLED_PULP='ON',GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET='ON',PULP_GPU_AUDIO_HAS_DAWN_SHARED_IO='ON'),True)
if __name__=='__main__':unittest.main()
