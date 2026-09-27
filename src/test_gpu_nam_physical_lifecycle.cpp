#include "gpu_nam_physical_lifecycle.hpp"
#include <stdexcept>
#include <string>
#include <iostream>
using namespace gpu_nam_diagnostic;
int main() {
    unsigned checks=0;
    auto check=[&](bool value){++checks;if(!value)throw std::runtime_error("lifecycle control failed");};
    for(unsigned epoch=0;epoch<2;++epoch) {
        std::string events;bool callback_active=true,prepared=true;
        auto result=stop_epoch([&]{events+='S';callback_active=false;return true;},
            [&]{check(!callback_active&&prepared);events+='Q';return true;},
            [&]{check(!callback_active&&prepared);events+='D';prepared=false;});
        check(result==EpochStopResult::Complete);check(events=="SQD");check(!prepared);
    }
    std::string events;
    auto stop_failed=stop_epoch([&]{events+='S';return false;},[&]{events+='Q';return true;},[&]{events+='D';});
    check(stop_failed==EpochStopResult::AdmissionNotDrained);check(events=="S");
    events.clear();
    auto query_failed=stop_epoch([&]{events+='S';return true;},[&]{events+='Q';return false;},[&]{events+='D';});
    check(query_failed==EpochStopResult::SnapshotRejected);check(events=="SQ");
    events.clear();bool threw=false;
    try {stop_epoch([&]{events+='S';return true;},[&]()->bool{events+='Q';throw std::runtime_error("snapshot failure");},[&]{events+='D';});}
    catch(const std::runtime_error&){threw=true;}
    check(threw);check(events=="SQ");
    std::cout<<checks<<" lifecycle sequencing controls passed\n";
}
