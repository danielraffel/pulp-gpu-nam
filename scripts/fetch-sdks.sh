#!/usr/bin/env bash
# Fetch the optional VST3 and AudioUnit SDKs that the VST3 and AU plugin formats
# need. CLAP and the Standalone app build without any external SDK; run this only
# if you want the VST3 and/or AU formats.
#
# Both SDKs are permissively licensed (VST3 SDK: dual GPL/proprietary — Pulp uses
# it under the proprietary/Steinberg terms for MIT-compatible distribution; AU:
# Apache-2.0) and are developer-supplied — they are NOT vendored in this repo.
# They clone into the Pulp submodule's external/ dir, where Pulp's build expects
# them.
#
# Usage:  ./scripts/fetch-sdks.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ext="$here/pulp/external"

if [[ ! -d "$here/pulp/.git" && ! -f "$here/pulp/.git" ]]; then
  echo "error: the Pulp submodule isn't checked out. Run:" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi
mkdir -p "$ext"

# Pin the same refs Pulp itself uses (see pulp/tools/deps/manifest.json).
VST3_REF="v3.7.12_build_20"

if [[ -d "$ext/vst3sdk/.git" ]]; then
  echo "vst3sdk already present ($ext/vst3sdk)"
else
  echo "Cloning VST3 SDK ($VST3_REF) …"
  git clone --depth 1 --branch "$VST3_REF" --recurse-submodules \
    https://github.com/steinbergmedia/vst3sdk.git "$ext/vst3sdk"
fi

if [[ -d "$ext/AudioUnitSDK/.git" ]]; then
  echo "AudioUnitSDK already present ($ext/AudioUnitSDK)"
else
  echo "Cloning AudioUnitSDK …"
  git clone --depth 1 https://github.com/apple/AudioUnitSDK.git "$ext/AudioUnitSDK"
fi

echo
echo "Done. Reconfigure to pick up the new formats:"
echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
