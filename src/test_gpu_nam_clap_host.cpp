#include "gpu_nam_host_probe.hpp"
#include <clap/clap.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
thread_local bool audio_thread=false;
bool is_main(const clap_host_t*) { return !audio_thread; }
bool is_audio(const clap_host_t*) { return audio_thread; }
const clap_host_thread_check_t thread_check{is_main,is_audio};
const void* extension(const clap_host_t*,const char* id) {
    return std::strcmp(id,CLAP_EXT_THREAD_CHECK)==0 ? &thread_check : nullptr;
}
void noop(const clap_host_t*) {}
void require(bool condition,const char* reason) { if(!condition) throw std::runtime_error(reason); }
struct Events {
    std::vector<clap_event_param_value_t> values;
    clap_input_events_t input{this,
        [](const clap_input_events_t* e){return std::uint32_t(static_cast<Events*>(e->ctx)->values.size());},
        [](const clap_input_events_t* e,std::uint32_t i)->const clap_event_header_t* {
            const auto& values=static_cast<Events*>(e->ctx)->values;
            return i<values.size() ? &values[i].header : nullptr;
        }};
    void add(clap_id id,double value) {
        clap_event_param_value_t event{};
        event.header={sizeof(event),0,CLAP_CORE_EVENT_SPACE_ID,CLAP_EVENT_PARAM_VALUE,0};
        event.param_id=id;event.note_id=-1;event.port_index=-1;event.channel=-1;event.key=-1;event.value=value;
        values.push_back(event);
    }
};
const clap_output_events_t discard{nullptr,[](const clap_output_events_t*,const clap_event_header_t*){return true;}};
struct Loaded {
    void* library=nullptr;
    const clap_plugin_entry_t* entry=nullptr;
    const clap_plugin_t* plugin=nullptr;
    bool initialized=false,active=false,processing=false;
    ~Loaded() {
        if(plugin) {
            if(processing){audio_thread=true;plugin->stop_processing(plugin);audio_thread=false;}
            if(active)plugin->deactivate(plugin);
            plugin->destroy(plugin);
        }
        if(initialized)entry->deinit();
        if(library)dlclose(library);
    }
};
}
int main(int argc,char** argv) {
    if(argc!=2)return 64;
    try {
        clap_host_t host{CLAP_VERSION,nullptr,"GPU NAM acceptance","Pulp","","1",extension,noop,noop,noop};
        Loaded loaded;
        loaded.library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
        require(loaded.library,"CLAP load failed");
        loaded.entry=static_cast<const clap_plugin_entry_t*>(dlsym(loaded.library,"clap_entry"));
        require(loaded.entry && loaded.entry->init(argv[1]),"CLAP entry init failed");loaded.initialized=true;
        const auto query=reinterpret_cast<GpuNamHostProbeQuery>(dlsym(loaded.library,"gpu_nam_host_probe_v1"));
        require(query,"diagnostic query absent: configure GPU_NAM_NATIVE_HOST_PROBE=ON; no GPU proof");
        const auto* factory=static_cast<const clap_plugin_factory_t*>(loaded.entry->get_factory(CLAP_PLUGIN_FACTORY_ID));
        require(factory && factory->get_plugin_count(factory)==1,"expected exactly one descriptor");
        const auto* descriptor=factory->get_plugin_descriptor(factory,0);
        loaded.plugin=factory->create_plugin(factory,&host,descriptor->id);
        require(loaded.plugin && loaded.plugin->init(loaded.plugin),"plugin init failed");
        const auto* params=static_cast<const clap_plugin_params_t*>(loaded.plugin->get_extension(loaded.plugin,CLAP_EXT_PARAMS));
        const auto* latency=static_cast<const clap_plugin_latency_t*>(loaded.plugin->get_extension(loaded.plugin,CLAP_EXT_LATENCY));
        require(params && latency,"params or latency extension unavailable");
        bool engine_found=false;
        for(std::uint32_t i=0;i<params->count(loaded.plugin);++i) {
            clap_param_info_t info{};require(params->get_info(loaded.plugin,i,&info),"parameter enumeration failed");
            if(info.id==4){engine_found=true;require(!(info.flags&CLAP_PARAM_IS_AUTOMATABLE),"stamped Engine must be preparation-bound");}
        }
        require(engine_found,"Engine parameter missing");
        require(query(nullptr)==1,"null snapshot query did not fail closed");
        GpuNamHostProbeSnapshot malformed;
        malformed.size=sizeof(malformed)-1;
        require(query(&malformed)==1,"malformed snapshot size did not fail closed");
        malformed.size=sizeof(malformed);malformed.version=2;
        require(query(&malformed)==1,"unsupported snapshot version did not fail closed");
        // Negative control: a query must never pick the last-created instance.
        const auto* second=factory->create_plugin(factory,&host,descriptor->id);
        require(second,"second-instance control setup failed");
        if(!second->init(second)){second->destroy(second);throw std::runtime_error("second-instance init failed");}
        GpuNamHostProbeSnapshot ambiguous;
        const auto ambiguous_result=query(&ambiguous);
        second->destroy(second);
        require(ambiguous_result==3,"multi-instance query did not fail closed");
        constexpr unsigned frames=128,blocks=96;
        std::array<std::vector<float>,3> captures;
        bool query_missing=false;
        for(unsigned epoch=0;epoch<3;++epoch) {
            const unsigned engine=epoch==0 ? 0 : 1;
            Events edits;edits.add(4,engine);edits.add(10,0);edits.add(11,0);edits.add(12,0);
            // CLAP explicitly permits params.flush while inactive. This write
            // precedes activate, unlike an automation event at sample zero.
            params->flush(loaded.plugin,&edits.input,&discard);
            double requested=-1;require(params->get_value(loaded.plugin,4,&requested)&&requested==engine,"inactive Engine write not applied");
            require(loaded.plugin->activate(loaded.plugin,48000,frames,frames),"activate failed");loaded.active=true;
            require(latency->get(loaded.plugin)==1024,"unexpected plugin PDC (expected512reblock+512lead)");
            audio_thread=true;
            require(loaded.plugin->start_processing(loaded.plugin),"start processing failed");loaded.processing=true;
            std::array<std::array<float,frames>,2> input{},output{};
            float* ip[]{input[0].data(),input[1].data()};float* op[]{output[0].data(),output[1].data()};
            clap_audio_buffer_t in{ip,nullptr,2,0,0},out{op,nullptr,2,0,0};
            Events empty;
            auto& capture=captures[epoch];capture.resize(blocks*frames*2);
            const auto origin=std::chrono::steady_clock::now();
            for(unsigned b=0;b<blocks;++b) {
                std::this_thread::sleep_until(origin+std::chrono::nanoseconds(std::uint64_t(b)*frames*1000000000ULL/48000));
                for(unsigned ch=0;ch<2;++ch)for(unsigned i=0;i<frames;++i)
                    input[ch][i]=.2f*std::sin(float(b*frames+i)*(.013f+.009f*ch));
                clap_process_t process{};process.steady_time=b*frames;process.frames_count=frames;
                process.audio_inputs=&in;process.audio_outputs=&out;process.audio_inputs_count=1;process.audio_outputs_count=1;
                process.in_events=&empty.input;process.out_events=&discard;
                require(loaded.plugin->process(loaded.plugin,&process)!=CLAP_PROCESS_ERROR,"CLAP process failed");
                for(unsigned ch=0;ch<2;++ch)for(unsigned i=0;i<frames;++i) {
                    require(std::isfinite(output[ch][i]),"nonfinite native plugin output");
                    capture[(b*frames+i)*2+ch]=output[ch][i];
                }
            }
            loaded.plugin->stop_processing(loaded.plugin);loaded.processing=false;audio_thread=false;
            // Query while the callback is stopped and the prepared state exists.
            GpuNamHostProbeSnapshot snapshot;
            const auto query_result=query(&snapshot);
            query_missing|=query_result==2;
            require(query_result==0 || query_result==2,"delivery snapshot query failed");
            require(snapshot.requested_engine==int(engine) && snapshot.prepared_engine==int(engine),"prepared Engine differs from inactive request");
            require(snapshot.active==engine,"effective engine differs from inactive request");
            require(snapshot.latency_samples==1024 && latency->get(loaded.plugin)==1024,"PDC changed within epoch");
            if(query_result==0 && engine==1) {
                require(snapshot.gpu_delivered>0,"no accepted GPU output: fallback is not GPU proof");
                require(snapshot.other==0 && snapshot.priming==1,"unexpected terminal disposition or priming count");
                require(snapshot.gpu_delivered+snapshot.cpu_fallback+snapshot.priming==blocks*frames/512,"delivery accounting incomplete");
            }
            if(query_result==0 && engine==0)
                require(snapshot.gpu_delivered==0 && snapshot.cpu_fallback==0 &&
                        snapshot.priming==0 && snapshot.other==0,
                        "CPU epoch reported transport selections");
            double energy=0,max_error=0;
            for(std::size_t i=0;i<capture.size();++i){energy+=double(capture[i])*capture[i];if(epoch)max_error=std::max(max_error,std::abs(double(capture[i])-captures[0][i]));}
            require(energy>1e-6,"silent native plugin output");
            if(epoch)require(max_error<1e-4,"native GPU/fallback audio differs from CPU reference");
            std::cout<<"epoch="<<epoch<<" engine="<<engine<<" pdc=1024 max_error="<<max_error
                     <<" delivery_query="<<query_result;
            if(query_result==0)std::cout<<" gpu_delivered="<<snapshot.gpu_delivered
                <<" cpu_fallback="<<snapshot.cpu_fallback<<" priming="<<snapshot.priming
                <<" other="<<snapshot.other;
            std::cout<<'\n';
            loaded.plugin->deactivate(loaded.plugin);loaded.active=false;
        }
        if(query_missing){std::cerr<<"SDK delivery query not wired: audio/lifecycle evidence only; GPU acceptance incomplete\n";return 78;}
        std::cout<<"native_clap_gpu_acceptance=passed\n";return 0;
    }catch(const std::exception& e){audio_thread=false;std::cerr<<e.what()<<'\n';return 1;}
}
