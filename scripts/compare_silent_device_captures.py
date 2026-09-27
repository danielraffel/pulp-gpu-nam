#!/usr/bin/env python3
"""Compare real callback captures offline. No audio device or plugin is opened."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def load(prefix):
    prefix = Path(prefix)
    paths = [Path(str(prefix) + suffix) for suffix in
             ('-metadata.txt', '-audio.csv', '-callbacks.csv')]
    meta = dict(line.split('=', 1) for line in paths[0].read_text().splitlines())
    with paths[1].open() as stream:
        audio = list(csv.DictReader(stream))
    values = []
    for i, row in enumerate(audio):
        if int(row['sample']) != i:
            raise ValueError('noncontiguous audio samples')
        pair = (float(row['left']), float(row['right']))
        if not all(math.isfinite(v) for v in pair):
            raise ValueError('nonfinite captured output')
        values.append(pair)
    with paths[2].open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(meta['callbacks']):
        raise ValueError('reported callback count differs from rows')
    if not rows:
        raise ValueError('no device callbacks')
    cursor = int(rows[0]['sample_position'])
    for i, row in enumerate(rows):
        n = int(row['frames'])
        if (int(row['callback']) != i or int(row['sample_position']) != cursor
                or cursor < 0 or not 0 < n <= 4096 or not 1 <= int(row['clap_status']) <= 4
                or int(row['observed_entry_ns']) < 0
                or int(row['observed_exit_ns']) < int(row['observed_entry_ns'])):
            raise ValueError('invalid callback row')
        cursor += n
    if cursor - int(rows[0]['sample_position']) != int(meta['frames_captured']):
        raise ValueError('callback accounting mismatch')
    if len(values) != int(meta['frames_compared']) or len(values) > int(meta['frames_captured']):
        raise ValueError('capture length mismatch')
    return meta, values, rows, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def compare(cpu_prefix, shared_prefix):
    cpu, ca, cr, ch = load(cpu_prefix)
    gpu, ga, gr, gh = load(shared_prefix)
    for key in ('device_id', 'actual_rate', 'input_channels', 'physical_output',
                'hardware_host_timestamp', 'pdc', 'frames_compared', 'lifetime', 'teardown_proven'):
        if cpu[key] != gpu[key]:
            raise ValueError('unmatched ' + key)
    pinned = {'actual_rate': '48000', 'input_channels': '0',
              'physical_output': 'silence', 'hardware_host_timestamp': 'unavailable',
              'pdc': '1024', 'lifetime': 'process_retained_until_exit',
              'teardown_proven': 'false', 'xrun_source': 'AudioDevice::xrun_count',
              'xrun_listener_availability': 'unavailable'}
    count_keys = ('gpu_selected', 'cpu_fallback', 'priming', 'other')
    for meta, engine in ((cpu, 'cpu'), (gpu, 'shared')):
        if meta['engine'] != engine or not meta['device_id']:
            raise ValueError('missing engine or explicit device identity')
        for key, expected in pinned.items():
            if meta[key] != expected:
                raise ValueError('invalid pinned contract: ' + key)
        counts = [int(meta[key]) for key in count_keys]
        if any(value < 0 for value in counts):
            raise ValueError('negative delivery count')
        if engine == 'cpu' and any(counts):
            raise ValueError('CPU control claimed transport selections')
        if engine == 'shared' and (counts[0] <= 0 or counts[2] != 1 or counts[3] != 0
                                   or sum(counts) != int(meta['frames_captured']) // 512):
            raise ValueError('invalid shared delivery accounting')
    if len(ca) != len(ga) or not ca:
        raise ValueError('unmatched capture size')
    peak = max(abs(a-b) for c, g in zip(ca, ga) for a, b in zip(c, g))
    energy = sum(v*v for pair in ca for v in pair)
    def timing(rows):
        durations = sorted(int(r['observed_exit_ns'])-int(r['observed_entry_ns']) for r in rows)
        return {'callbacks': len(rows), 'observed_frame_sizes': sorted({int(r['frames']) for r in rows}),
                'process_observation_ns_p50': durations[len(durations)//2],
                'process_observation_ns_max': durations[-1]}
    return {'schema': 'gpu-nam.silent-device-comparison.v1',
            'passed': peak < 1e-4 and energy > 1e-6,
            'max_absolute_error': peak, 'cpu_energy': energy,
            'same_observed_partitions': [r['frames'] for r in cr] == [r['frames'] for r in gr],
            'cpu': timing(cr), 'shared': timing(gr),
            'alignment': 'same synthesized input sample index; no fitted offset',
            'deadline_claim': False, 'teardown_tested': False,
            'audio_workgroup_membership_measured': False,
            'sha256': {**ch, **gh}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cpu_prefix')
    parser.add_argument('shared_prefix')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = compare(args.cpu_prefix, args.shared_prefix)
    except (ValueError, KeyError, OSError) as error:
        result = {'passed': False, 'error': str(error)}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
