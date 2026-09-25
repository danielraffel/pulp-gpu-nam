#!/usr/bin/env bash
# Validate GPU NAM against an exact installed Pulp SDK.
#
# This checks the public API exported by Pulp #8848 and the GPU-NAM consumer
# contract. It deliberately does not construct a WaveNet shared provider: the
# generic shared-I/O session in #8848 is private, and GPU-NAM must remain on its
# staged path until a Pulp-owned WaveNet provider exists.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: validate-installed-pulp-gpu-audio.sh --sdk-prefix DIR --sdk-commit SHA [options]

Options:
  --sdk-prefix DIR    Installed Pulp SDK prefix to consume.
  --sdk-commit SHA    Exact Pulp source/SDK receipt identifier (required).
  --build-dir DIR     Disposable CMake build directory.
  --receipt FILE      JSON receipt path (default: BUILD_DIR/gpu-nam-installed-sdk-validation.json).
  --jobs N            Build parallelism (default: 4).
  -h, --help          Show this help.

The script requires the public gpu_audio_program.hpp, gpu_audio_capability.hpp,
and gpu_convolver.hpp headers, and verifies the #8848 ProviderPolicy::SharedRequired
API. It runs only compile/configure/consumer capability checks. It never claims
that GPU NAM has a shared WaveNet execution provider.
USAGE
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
sdk_prefix=""
sdk_commit="${PULP_SDK_COMMIT:-}"
build_dir=""
receipt=""
jobs="${GPU_NAM_BUILD_JOBS:-4}"

while (($#)); do
    case "$1" in
        --sdk-prefix) [[ $# -ge 2 ]] || { echo "--sdk-prefix requires a value" >&2; exit 2; }; sdk_prefix=$2; shift 2 ;;
        --sdk-commit) [[ $# -ge 2 ]] || { echo "--sdk-commit requires a value" >&2; exit 2; }; sdk_commit=$2; shift 2 ;;
        --build-dir) [[ $# -ge 2 ]] || { echo "--build-dir requires a value" >&2; exit 2; }; build_dir=$2; shift 2 ;;
        --receipt) [[ $# -ge 2 ]] || { echo "--receipt requires a value" >&2; exit 2; }; receipt=$2; shift 2 ;;
        --jobs) [[ $# -ge 2 ]] || { echo "--jobs requires a value" >&2; exit 2; }; jobs=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$sdk_prefix" ]] || { echo "--sdk-prefix is required" >&2; exit 2; }
[[ -n "$sdk_commit" ]] || { echo "--sdk-commit is required (or set PULP_SDK_COMMIT)" >&2; exit 2; }
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
[[ -d "$sdk_prefix" ]] || { echo "SDK prefix does not exist: $sdk_prefix" >&2; exit 2; }
sdk_prefix=$(cd "$sdk_prefix" && pwd -P)

if [[ -z "$build_dir" ]]; then build_dir="$repo_root/build-installed-pulp-gpu-audio"; fi
build_dir=$(mkdir -p "$build_dir" && cd "$build_dir" && pwd -P)
if [[ -z "$receipt" ]]; then receipt="$build_dir/gpu-nam-installed-sdk-validation.json"; fi
receipt=$(mkdir -p "$(dirname "$receipt")" && cd "$(dirname "$receipt")" && pwd -P)/$(basename "$receipt")

pulp_config=$(find "$sdk_prefix" -type f -name PulpConfig.cmake -print -quit)
[[ -n "$pulp_config" ]] || { echo "PulpConfig.cmake not found below SDK prefix: $sdk_prefix" >&2; exit 1; }

for header in pulp/gpu_audio/gpu_audio_program.hpp pulp/gpu_audio/gpu_audio_capability.hpp pulp/gpu_audio/gpu_convolver.hpp; do
    [[ -f "$sdk_prefix/include/$header" ]] || { echo "required installed SDK header is missing: $sdk_prefix/include/$header" >&2; exit 1; }
done
grep -q "SharedRequired" "$sdk_prefix/include/pulp/gpu_audio/gpu_convolver.hpp" || { echo "installed gpu_convolver.hpp has no ProviderPolicy::SharedRequired (#8848)" >&2; exit 1; }
grep -q "GpuAudioProgramDescriptor" "$sdk_prefix/include/pulp/gpu_audio/gpu_audio_program.hpp" || { echo "installed gpu_audio_program.hpp has no typed program descriptor (#8843)" >&2; exit 1; }

sha256_file() {
    shasum -a 256 "$1" | awk '{print $1}'
}
pulp_config_sha256=$(sha256_file "$pulp_config")
program_header_sha256=$(sha256_file "$sdk_prefix/include/pulp/gpu_audio/gpu_audio_program.hpp")
convolver_header_sha256=$(sha256_file "$sdk_prefix/include/pulp/gpu_audio/gpu_convolver.hpp")

probe_src=$(mktemp "${TMPDIR:-/tmp}/gpu-nam-sdk-probe.XXXXXX.cpp")
trap 'rm -f "$probe_src"' EXIT
cat >"$probe_src" <<'PROBE'
#include <pulp/gpu_audio/gpu_audio_capability.hpp>
#include <pulp/gpu_audio/gpu_audio_program.hpp>
#include <pulp/gpu_audio/gpu_convolver.hpp>
static_assert(static_cast<unsigned>(pulp::gpu_audio::GpuConvolver::ProviderPolicy::SharedRequired) != static_cast<unsigned>(pulp::gpu_audio::GpuConvolver::ProviderPolicy::Auto));
static_assert(static_cast<unsigned>(pulp::gpu_audio::GpuAudioProgramKind::Neural) != 0);
int main() { return 0; }
PROBE

echo "[1/4] syntax-checking installed Pulp #8848 public GPU-audio API"
"${CXX:-c++}" -std=c++20 -fsyntax-only -Wall -Wextra -Werror -I"$sdk_prefix/include" "$probe_src"

echo "[2/4] configuring GPU NAM against installed SDK"
cmake -S "$repo_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DGPU_NAM_USE_INSTALLED_PULP=ON -DCMAKE_PREFIX_PATH="$sdk_prefix" -DGPU_NAM_BUILD_TESTS=ON -DGPU_NAM_BUILD_GPU_AUDIO_CAPABILITY_PROBE=ON

echo "[3/4] building consumer contract targets"
cmake --build "$build_dir" --target gpu-nam-prepared-program-test gpu-nam-gpu-audio-capability-probe --parallel "$jobs"
echo "[4/4] running consumer contract tests"
ctest --test-dir "$build_dir" --output-on-failure -R 'gpu-nam-(prepared-program|gpu-audio-capability-probe)'

source_head=$(git -C "$repo_root" rev-parse HEAD)
python3 - "$receipt" "$source_head" "$sdk_commit" "$sdk_prefix" "$pulp_config" "$pulp_config_sha256" "$program_header_sha256" "$convolver_header_sha256" "$build_dir" "$jobs" <<'PY'
import json
import pathlib
import sys
(
    receipt,
    source_head,
    sdk_commit,
    sdk_prefix,
    pulp_config,
    pulp_config_sha256,
    program_header_sha256,
    convolver_header_sha256,
    build_dir,
    jobs,
) = sys.argv[1:]
data = {
    "schema": "gpu-nam.installed-sdk-gpu-audio-validation.v1",
    "source_head": source_head,
    "pulp_sdk_commit": sdk_commit,
    "sdk_prefix": sdk_prefix,
    "pulp_config": pulp_config,
    "artifacts": {
        "PulpConfig.cmake": pulp_config_sha256,
        "gpu_audio_program.hpp": program_header_sha256,
        "gpu_convolver.hpp": convolver_header_sha256,
    },
    "build_dir": build_dir,
    "build_jobs": int(jobs),
    "checks": {
        "typed_program_api": "pass",
        "shared_required_provider_policy_api": "pass",
        "gpu_nam_prepared_program_consumer": "pass",
        "gpu_nam_capability_report_consumer": "pass",
        "gpu_nam_shared_wavenet_execution": "blocked_provider_not_implemented",
    },
    "interpretation": "The installed SDK exposes typed program and SharedRequired convolver contracts. GPU-NAM consumes them for validation; this receipt makes no shared WaveNet execution claim.",
}
pathlib.Path(receipt).write_text(json.dumps(data, indent=2) + "\n")
print(f"receipt={receipt}")
PY
