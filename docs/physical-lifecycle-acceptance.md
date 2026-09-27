# Manual silent physical lifecycle acceptance

This is a diagnostic extension, disabled by default. It is not a plugin
performance claim or a native-resource leak test. First complete ordinary
installed CLAP acceptance. Then build the existing
`gpu-nam-silent-device-probe` target with installed SDK mode, native host probe,
silent device probe and `GPU_NAM_NORMAL_PHYSICAL_LIFECYCLE=ON`. This narrow
experiment requires exact SDK source
`7858d7edb5a2bbb568c03d7e740043e8c897c0ee` and a clean committed product tree.
The executable embeds the source and SDK revisions. Do not supply private
CoreAudio headers or source overlays.

Enumerate devices with the probe's `--list-devices` first. Running without
arguments never opens hardware. With explicit device authorization and an
owned host window, use the external watchdog:

```sh
python3 -B scripts/run_physical_lifecycle.py \
  --executable /absolute/path/gpu-nam-silent-device-probe \
  --clap /absolute/path/GpuNam.clap/Contents/MacOS/GpuNam \
  --expected-product-sha EXACT_REVIEWED_PRODUCT_SHA \
  --sdk-provenance /absolute/sdk/prefix/sdk-provenance.json \
  --device-id ENUMERATED_ID --mode shared \
  --output-dir /absolute/new/physical-shared
```

Run `--mode cpu` separately with another fresh output directory. Physical
output is always silence; plugin output goes only to capture files. The parent
creates the directory exclusively and the child rejects previously used
prefixes, so a failed repeat cannot inherit an old success file. The parent
kills the entire child process group on timeout and retains exit status, last
stage, logs and file hashes. Exit 78 or missing/incomplete receipts fail.

Normal mode executes two freshly activated epochs on the same open device.
The callback stops CLAP processing and signals completion. Control then calls
the installed SDK's stop barrier before reading the snapshot or deactivating.
On this pinned CoreAudio implementation, callback admission closes and drains
before native stop is attempted. `done` alone is not that barrier. The second
epoch reactivates the plugin and restarts the device with fresh history.
Afterward the diagnostic closes and destroys the device/system, destroys the
inactive plugin, deinitializes its entry and checks `dlclose` before returning
normally. The shared `Loaded` helper and ordinary CLAP acceptance are unchanged.

Each epoch preserves callback/audio CSVs and metadata including selected GPU,
fallback and priming counts. The final child lifecycle receipt is written only
after successful teardown. The parent binds it to mode/device, embedded source
and SDK revisions, input hashes before launch and unchanged hashes afterward, both epoch records
and process exit. The expected product SHA is an explicit required argument.
Malformed epoch metadata fails into the parent receipt.
Source revisions identify the configured source; archive exact compile/link
receipts separately when accepting an installed build.

Failures retain reachable owners until process exit. A hung teardown is a
watchdog failure, never permission to free owners while the call runs. Native
stop/close error stages and GPU retirement totals are unavailable through this
installed public API. CoreAudio intentionally keeps closed refcon tombstones
for late native entries. Successful stop/restart/close observations therefore
do not establish native leak freedom or reclamation of those tombstones.

The pure C++ controls exercise stop-before-snapshot-before-deactivate ordering
and refusal on stop/query failure. Python controls exercise whole-child timeout,
nonzero exits and malformed/missing lifecycle evidence. They do not prove
physical device close/unload behavior. That remains an explicit manual gate.
