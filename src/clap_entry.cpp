#include "gpu_nam_processor.hpp"
#include <pulp/format/clap_entry.hpp>
#if defined(GPU_NAM_NATIVE_HOST_PROBE)
#include "gpu_nam_host_probe_entry.hpp"
PULP_CLAP_PLUGIN(pulp::examples::host_probe::create)
#else
PULP_CLAP_PLUGIN(pulp::examples::create_gpu_nam)
#endif
