#include "gpu_nam_delivery_status.hpp"
#include <iostream>
using namespace pulp::examples::nam;
namespace {
struct OlderTransport {
    struct Stats { unsigned produced_blocks=999, miss_blocks=50; };
    Stats stats() const { return {}; }
};
struct PartialTransport {
    struct Snapshot { unsigned gpu_blocks=999; };
    Snapshot delivery_snapshot() const { return {}; }
};
struct CurrentTransport {
    struct Snapshot {
        std::uint64_t gpu_blocks=0,worker_output_blocks=0,cpu_fallback_blocks=0,
                      silence_blocks=0,passthrough_blocks=0,priming_blocks=0,invalid_blocks=0;
    } value;
    Snapshot delivery_snapshot() const { return value; }
};
}
int main() {
    unsigned checks=0;
    auto check=[&](bool value){++checks;if(!value){std::cerr<<"failed check "<<checks<<'\n';return false;}return true;};
    auto unavailable=read_gpu_nam_delivery_status(OlderTransport{});
    if(!check(!unavailable.available) || !check(unavailable.gpu_selected==0) ||
       !check(gpu_nam_delivery_label(true,unavailable)=="GPU path: delivery counts unavailable") ||
       !check(!read_gpu_nam_delivery_status(PartialTransport{}).available))return 1;
    CurrentTransport transport;
    transport.value={7,0,3,1,2,4,5};
    auto current=read_gpu_nam_delivery_status(transport);
    if(!check(current.available) || !check(current.gpu_selected==7) || !check(current.cpu_fallback==3) ||
       !check(current.silence==1) || !check(current.passthrough==2) || !check(current.priming==4) ||
       !check(current.invalid==5) || !check(gpu_nam_delivery_label(true,current)=="GPU selected 7; CPU fallback 3"))return 1;
    transport.value={0,42,9,0,0,0,0};
    const auto worker=read_gpu_nam_delivery_status(transport);
    if(!check(worker.gpu_selected==0) || !check(worker.worker_selected==42) ||
       !check(gpu_nam_delivery_label(true,worker)=="Worker output 42; CPU fallback 9"))return 1;
    transport.value={0,0,15,0,0,1,0};
    const auto fallback=read_gpu_nam_delivery_status(transport);
    if(!check(gpu_nam_delivery_label(true,fallback)=="GPU selected 0; CPU fallback 15") ||
       !check(gpu_nam_delivery_label(false,current)=="CPU engine active") ||
       !check(gpu_nam_delivery_label(true,current).find("RT")==std::string::npos))return 1;
    transport.value={2,8,1,0,0,0,0};
    if(!check(gpu_nam_delivery_label(true,read_gpu_nam_delivery_status(transport))==
              "GPU selected 2; CPU fallback 1; worker output 8"))return 1;
    std::cout<<checks<<" delivery source/label checks passed\n";
}
