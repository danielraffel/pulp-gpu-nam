import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('compare', Path(__file__).with_name('compare_silent_device_captures.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CaptureComparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cpu, self.gpu = (Path(self.temp.name) / name for name in ('cpu', 'gpu'))
        for prefix, engine in ((self.cpu, 'cpu'), (self.gpu, 'shared')):
            Path(str(prefix)+'-metadata.txt').write_text(
                'engine='+engine+'\ndevice_id=fixture\nactual_rate=48000\ninput_channels=0\n'
                'physical_output=silence\nhardware_host_timestamp=unavailable\npdc=1024\n'
                'frames_compared=2\nframes_captured=1024\nlifetime=process_retained_until_exit\n'
                'teardown_proven=false\ncallbacks=1\nxrun_source=AudioDevice::xrun_count\n'
                'xrun_listener_availability=unavailable\ncpu_fallback=0\nother=0\n'
                'priming='+('1' if engine == 'shared' else '0')+'\n'
                'gpu_selected='+('1' if engine == 'shared' else '0')+'\n')
            Path(str(prefix)+'-audio.csv').write_text('sample,left,right\n0,0.5,-0.25\n1,0.1,-0.2\n')
            Path(str(prefix)+'-callbacks.csv').write_text(
                'callback,sample_position,frames,observed_entry_ns,observed_exit_ns,clap_status\n0,0,1024,100,200,1\n')

    def test_matching_capture(self):
        self.assertTrue(module.compare(self.cpu, self.gpu)['passed'])

    def test_corruption_is_not_accepted(self):
        Path(str(self.gpu)+'-audio.csv').write_text('sample,left,right\n0,0.75,-0.25\n1,0.1,-0.2\n')
        self.assertFalse(module.compare(self.cpu, self.gpu)['passed'])

    def test_missing_gpu_is_not_accepted(self):
        p = Path(str(self.gpu)+'-metadata.txt')
        p.write_text(p.read_text().replace('gpu_selected=1', 'gpu_selected=0'))
        with self.assertRaises(ValueError): module.compare(self.cpu, self.gpu)

    def test_wrong_device_is_not_accepted(self):
        p = Path(str(self.gpu)+'-metadata.txt')
        p.write_text(p.read_text().replace('device_id=fixture', 'device_id=other'))
        with self.assertRaises(ValueError): module.compare(self.cpu, self.gpu)

    def test_partition_difference_is_explicit(self):
        Path(str(self.gpu)+'-callbacks.csv').write_text(
            'callback,sample_position,frames,observed_entry_ns,observed_exit_ns,clap_status\n'
            '0,0,512,100,200,1\n1,512,512,300,400,1\n')
        p = Path(str(self.gpu)+'-metadata.txt')
        p.write_text(p.read_text().replace('callbacks=1', 'callbacks=2'))
        result = module.compare(self.cpu, self.gpu)
        self.assertTrue(result['passed'])
        self.assertFalse(result['same_observed_partitions'])

    def test_matching_wrong_contract_is_rejected(self):
        for key, wrong in (('actual_rate', '44100'), ('pdc', '512'),
                           ('input_channels', '2'), ('physical_output', 'plugin_audio'),
                           ('hardware_host_timestamp', 'assumed'),
                           ('lifetime', 'unloaded'), ('teardown_proven', 'true'),
                           ('xrun_listener_availability', 'available')):
            originals = []
            for prefix in (self.cpu, self.gpu):
                p = Path(str(prefix)+'-metadata.txt')
                original = p.read_text()
                originals.append((p, original))
                p.write_text('\n'.join(key+'='+wrong if line.startswith(key+'=') else line
                                      for line in original.splitlines())+'\n')
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.compare(self.cpu, self.gpu)
            for p, original in originals:
                p.write_text(original)

    def test_malformed_delivery_and_callback_counts_are_rejected(self):
        cases = ((self.cpu, 'cpu_fallback', '1'), (self.cpu, 'priming', '1'),
                 (self.cpu, 'other', '1'), (self.gpu, 'gpu_selected', '-1'),
                 (self.gpu, 'cpu_fallback', '-1'), (self.gpu, 'cpu_fallback', '1'),
                 (self.gpu, 'priming', '0'), (self.gpu, 'other', '1'),
                 (self.gpu, 'callbacks', '2'))
        for prefix, key, wrong in cases:
            p = Path(str(prefix)+'-metadata.txt')
            original = p.read_text()
            p.write_text('\n'.join(key+'='+wrong if line.startswith(key+'=') else line
                                  for line in original.splitlines())+'\n')
            with self.subTest(key=key, value=wrong), self.assertRaises(ValueError):
                module.compare(self.cpu, self.gpu)
            p.write_text(original)

    def test_gap_is_not_accepted(self):
        p = Path(str(self.gpu)+'-audio.csv')
        p.write_text(p.read_text().replace('1,0.1', '2,0.1'))
        with self.assertRaises(ValueError): module.compare(self.cpu, self.gpu)


if __name__ == '__main__':
    unittest.main()
