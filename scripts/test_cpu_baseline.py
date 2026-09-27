#!/usr/bin/env python3
"""Run the built CPU diagnostic; absence or a wrong delay is never a pass."""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'src/models/example.nam')
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    model = args.model.resolve(strict=True)
    checked = 0

    def run(options, expected):
        nonlocal checked
        result = subprocess.run([str(executable), f'--model-path={model}', *options],
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == expected, (options, result.returncode, result.stdout, result.stderr)
        checked += 1
        return result

    for block in (32, 64, 128):
        for lead in (1, 2, 4, 8):
            result = run([f'--block-size={block}', f'--lead-blocks={lead}'], 0)
            fields = dict(part.split('=', 1) for part in result.stdout.split())
            assert fields['diagnostic_status'] == 'passed'
            assert fields['output_finite'] == '1'
            assert float(fields['max_error']) <= 1e-6
            assert int(fields['measured_samples']) == (96 + lead) * block
            assert float(fields['process_cpu_seconds']) >= 0
            assert int(fields['process_cpu_ticks_per_second']) > 0
            assert int(fields['elapsed_process_ns']) >= 0
            assert int(fields['wall_ns']) >= int(fields['elapsed_process_ns'])
    for option in ('--block-size=0', '--lead-blocks=3', '--block-size=32junk',
                   '--block-size=4294967328', '--lead-blocks=-1', '--unknown=1',
                   f'--model-path={model.parent / "not-a-model.nam"}'):
        run([option], 2)
    wrong = run(['--lead-blocks=4', '--verify-delay-blocks=3'], 3)
    assert 'diagnostic_status=failed' in wrong.stdout
    assert 'output_finite=1' in wrong.stdout
    print(f'{checked} CPU baseline invocation checks passed, including wrong-delay rejection')


if __name__ == '__main__':
    main()
