#pragma once
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
namespace {
struct LoadedApi {
    GpuNamHostProbeQuery query;
    const clap_plugin_factory_t* factory;
    const clap_plugin_descriptor_t* descriptor;
    const clap_plugin_params_t* params;
    const clap_plugin_latency_t* latency;
};
LoadedApi load_plugin(Loaded& loaded, const char* path, const clap_host_t& host) {
        loaded.library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
        require(loaded.library,"CLAP load failed");
        loaded.entry=static_cast<const clap_plugin_entry_t*>(dlsym(loaded.library,"clap_entry"));
        require(loaded.entry && loaded.entry->init(path),"CLAP entry init failed");loaded.initialized=true;
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
    return {query, factory, descriptor, params, latency};
}
}
