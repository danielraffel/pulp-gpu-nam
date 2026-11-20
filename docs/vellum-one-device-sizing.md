# GPU NAM on one Vellum Dawn device

Status: sizing for validation member 4, plan item 2.1.7, Lane B6.

This document is a migration plan. It changes no product code. The goal is for
GPU NAM's WaveNet inference and its Vellum editor to use the one Dawn device
created by the installed Vellum GPU plug-in host.

## Current ownership

Today GPU NAM has two relevant paths.

- `src/gpu_nam_cloud_node.hpp` owns `shared_gpu_`, a `std::unique_ptr<render::GpuCompute>`. `prepare()` calls `render::GpuCompute::create()` and `initialize_standalone()`. The object is shared by the per-channel `nam::GpuNam` plans, so stereo already uses one compute device for audio, but that device is created by the plug-in and is independent of the editor.
- `src/gpu_nam.hpp` contains the same ownership split. `GpuNam::prepare()` creates an owned standalone device, while `prepare_with()` accepts a caller-owned device and assigns an instance slot. The latter is the seam needed for host injection.
- `src/gpu_nam_processor.hpp` creates the audio `GpuAudioTransport` and the `GpuNamCloudNode` in `build_gpu_stack()`. It currently has no Vellum host-device token to pass into that node. A failed standalone preparation routes to the existing CPU engine.
- `src/gpu_nam_view.cpp` constructs `GpuNamUi`. `src/gpu_nam_ui.hpp` draws with the canvas supplied by the view host and calls `set_requires_gpu_host(true)`. The editor therefore uses the host's render surface for painting, but the processor cannot currently prove that its compute object uses that surface's device.
- `src/gpu_nam_staged_diagnostic.hpp` and several GPU tests also call `GpuCompute::create()` plus `initialize_standalone()`. Those are diagnostic or headless paths, not proof of the installed plug-in host contract.

The installed Vellum host creates the `GpuSurface` and its Dawn instance, adapter,
device, and queue. Vellum's Skia/Dawn provider receives those handles for the
editor. The migration must make the audio path adopt that same surface/device.

## Target runtime contract

The consumer should link the installed `VellumGpuRuntime::DeviceApi` target and
use the public `gpu_runtime` headers. The relevant API is:

```cpp
std::unique_ptr<gpu_runtime::GpuCompute> compute =
    gpu_runtime::GpuCompute::create();
compute->initialize_from_surface(host_surface);
```

`initialize_from_surface(gpu_runtime::GpuSurface&)` is the native shared-device
entry point. It must replace `initialize_standalone()` in the plug-in path.
`GpuCompute` then exposes the existing WaveNet preparation and forward methods,
including the `(block_size, instance)` plan slots that keep stereo channel
history independent while sharing device, queue, and pipelines.

The host handoff needs one typed, lifetime-safe way to obtain the current
`GpuSurface` from the Vellum plug-in editor host and make it available to the
non-real-time GPU-stack builder. The handoff must be invalidated before the
surface is destroyed or recreated. It must not pass a raw Dawn or Skia type
through GPU NAM. `GpuSurface::dawn_device_handle()` remains an opaque diagnostic
identity only. No consumer code should call `dawnProcSetProcs` or create a
second Dawn instance.

The Vellum runtime's `GpuSurface` and `GpuCompute` are single-owner-strand
objects. GPU NAM must keep preparation, WaveNet submissions, readback polling,
and device-loss publication on the host-approved worker strand. The audio
callback continues to use only the existing bounded transport and CPU fallback.

## Files and expected changes

The following is the complete consumer change set currently sized. The first
PR in this lane is this document only.

| File | Planned change | Main risk |
| --- | --- | --- |
| `CMakeLists.txt` | Require the installed Vellum GPU runtime in installed-SDK mode and link the consumer to `VellumGpuRuntime::DeviceApi`/`VellumGpuRuntime::Device`; keep source-tree compatibility only behind an explicit development option. | A package can configure against headers from one runtime and a library from another. Add an exact version/lock check and inspect the final link closure for a private Dawn copy. |
| `cmake/GpuNamSdkContract.cmake` | Extend the contract from the old Pulp render target to the Vellum runtime target and record the Vellum runtime/package identity used by the build. | A green configure can still select the old Pulp GPU implementation. Fail when the Vellum target or required WaveNet API is absent. |
| `src/gpu_nam.hpp` | Include `<gpu_runtime/gpu_compute.hpp>`, use `gpu_runtime::GpuCompute`, preserve `prepare_with()` and instance-slot semantics, and make standalone preparation diagnostic-only. | Ownership or `unique_ptr` changes can invalidate plan pointers. Add lifetime tests and keep `prepare_with()` non-owning. |
| `src/gpu_nam_cloud_node.hpp` | Remove standalone device creation from the host plug-in path. Accept a host-provided `gpu_runtime::GpuCompute` or `GpuSurface`, initialize from the shared surface, then prepare one plan per channel. Preserve prewarm, CPU shadow, and `MissPolicy::CpuFallback`. | A surface can disappear during editor close or device loss. Preparation must fail closed, publish CPU fallback, and never touch the audio callback. |
| `src/gpu_nam_processor.hpp` | Add the host-runtime handoff to `build_gpu_stack()`. Acquire the host surface on the approved worker strand, build the node against it, and release the handoff before retirement. Keep model reload and engine switching behavior unchanged. | Rebuilds can race editor teardown. The current stack-retirement rule must cover the surface reference and compute object, and a stale handoff must be rejected by generation. |
| `src/gpu_nam_staged_diagnostic.hpp` | Convert the diagnostic node to dependency injection. Its standalone mode may remain for headless benchmarking, but the plug-in-host acceptance mode must use `initialize_from_surface()`. | Accidentally treating a headless standalone result as host proof. Label and gate the modes separately. |
| `src/gpu_nam_view.cpp` | Pass the host-runtime identity into the processor/view bridge if the installed host exposes it only at view creation. No renderer or Dawn calls belong here. | Passing a view-local object into the audio stack can create a dangling reference. Use a shared host lease with explicit teardown and generation. |
| `src/gpu_nam_ui.hpp` | Keep canvas drawing and interaction unchanged. If the Vellum host API requires a declaration for the runtime lease, add only that typed bridge call and retain `set_requires_gpu_host(true)`. | UI repaint and compute submission could contend on an undocumented thread. The UI must remain a consumer of the host canvas, never a device owner. |
| `src/test_gpu_nam.cpp` and `src/test_gpu_nam_real.cpp` | Replace host-path standalone setup with an injected shared surface/device fixture. Continue CPU/GPU parity, stereo independence, prewarm, and device-loss/fallback assertions. | Tests may silently skip when no device exists. The installed-host fixture must distinguish unavailable hardware from a failed second-device assertion. |
| `src/test_gpu_nam_plugin.cpp` and `src/test_gpu_nam_clap_host.cpp` | Add an installed-host editor/audio fixture that opens the real plug-in view, runs a GPU block, and records the runtime device identity for both consumers. | A reported FX or headless render is insufficient. The test must observe a measurable editor surface and actual GPU delivery. |
| `src/test_gpu_nam_one_device.cpp` (new) | Add positive sharing and negative second-device tests described below. | A test-only fake can pass while production still links a private Dawn. Pair it with binary/link and installed-consumer checks. |
| `src/gpu_nam_gpu_cpu_diagnostic.cpp` and `src/gpu_audio_capability_probe.cpp` | Report the Vellum runtime identity and whether compute was adopted from the host surface. Keep reports fail-closed before preparation. | Metadata could be mistaken for execution proof. Reports must remain supplementary to the real host test. |
| `tools/validation/installed_sdk_gpu_audio_receipt.py` and its tests | Require the Vellum package, exact runtime identity, one-device receipt, and a negative second-device result. | A stale SDK or old executable can produce a false receipt. Keep the empty-build-dir and exact-source/package hashes. |
| `docs/installed-sdk-gpu-audio.md` | Replace the old “does not prove private shared-memory provider” wording with the Vellum one-device acceptance scope and its limits. | Documentation can claim support before editor/audio evidence exists. Keep source, package, runtime, and product proof as separate gates. |

No change is expected in the NAM model files, assets, `include/gpu_nam/forge_adapter.hpp`,
or the CPU-only architecture tests.

## Test plan

### API and source checks

1. Configure against an installed Vellum SDK and assert that the consumer links
   `VellumGpuRuntime::DeviceApi` and the Vellum device implementation. Fail if
   `pulp::render::GpuCompute` or `initialize_standalone()` is reachable from the
   plug-in target.
2. Run the runtime's device-sharing verifier with the editor surface. It must
   report one device, successful compute-buffer creation, concurrent submission,
   and no corruption.
3. Run a binary/link census. The plug-in must not contain a second bundled Dawn,
   a second `libwgpu_native`, or a private device implementation.

### Positive one-device acceptance

1. The installed Vellum plug-in host creates one `GpuSurface` and opens the GPU
   NAM editor. Record its runtime/device generation and adapter identity.
2. Build the GPU audio stack from that host surface. Prepare both stereo NAM
   instances with distinct instance slots on the adopted `GpuCompute`.
3. Render editor frames while processing a sequence of audio blocks. Assert that
   the editor remains measurable, GPU delivery occurs, CPU parity stays within
   the existing tolerance, and no audio callback performs allocation or waits on
   Dawn.
4. Destroy and recreate the editor surface. The old generation must be rejected,
   the audio stack must fall back safely, and a rebuild may adopt only the new
   generation.

### Planted second-device negative control

The host fixture must install a device-creation guard before opening the plug-in.
The guard counts Dawn device/adapter creation and rejects any request after the
host's first device. Run GPU NAM with the editor open and request a GPU engine
rebuild. The test fails if the plug-in calls `initialize_standalone()`, creates a
second `GpuSurface`, or otherwise obtains a new device. The expected result is a
single host device, successful compute adoption from that device, and no second
creation attempt. A second explicit `GpuCompute::initialize_standalone()` probe
must return failure under the guard and leave the current stack on CPU fallback.

This negative control is required in addition to a static census because a
runtime-loaded or hidden duplicate can evade symbol inspection. It must run in
the installed consumer lane with the exact Vellum package used for the positive
proof.

## PR-sized estimate

1. **SDK seam and build contract, 1 PR.** Add the Vellum targets, exact package
   identity checks, and a compile-only consumer fixture. No runtime behavior
   changes.
2. **Injected compute ownership, 1 PR.** Port `gpu_nam.hpp` and
   `gpu_nam_cloud_node.hpp` to `initialize_from_surface()`. Preserve the current
   CPU fallback and add unit coverage for non-owning lifetime and stereo slots.
3. **Plugin-host handoff, 1 PR.** Wire the Vellum surface lease through
   `gpu_nam_processor.hpp` and the view bridge. Add generation and teardown
   handling without changing UI painting.
4. **One-device acceptance, 1 PR.** Add the real editor/audio fixture, runtime
   device-sharing receipt, and planted second-device negative test.
5. **Installed packaging and documentation, 1 PR.** Update the receipt tooling,
   SDK validation docs, link census, and final macOS installed-consumer proof.

The estimate is five reviewable PRs after this sizing PR. Product support is
complete only when the installed positive and negative host tests, exact package
identity, binary census, and visible editor/audio proof all pass.

## Evidence boundary

The existing standalone GPU tests and historical GPU NAM receipts establish that
NAM inference can use one compute device for its stereo channels. They do not
prove that inference and the Vellum editor share the installed host device. That
claim requires the host fixture and the planted second-device failure above.
