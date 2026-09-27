#pragma once
#include "gpu_nam_completion_options.hpp"
#include "gpu_nam_paced_options.hpp"
#include "nam_model.hpp"
namespace pulp::examples {
int run_stamped_paced(const nam::NamModel&, unsigned frames, unsigned lead,
                     GpuNamCompletionOptions, const GpuNamPacedOptions&);
}
