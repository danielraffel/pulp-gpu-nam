#include "gpu_nam_clap_host_common.hpp"
int main(int argc,char** argv) {
    if(argc!=2)return 64;
    try {
        clap_host_t host{CLAP_VERSION,nullptr,"GPU NAM acceptance","Pulp","","1",extension,noop,noop,noop};
        Loaded loaded;
        const auto api=load_plugin(loaded,argv[1],host);
        const auto query=api.query;
        const auto* factory=api.factory;
        const auto* descriptor=api.descriptor;
        const auto* params=api.params;
        const auto* latency=api.latency;
        bool engine_found=false;
        for(std::uint32_t i=0;i<params->count(loaded.plugin);++i) {
            clap_param_info_t info{};require(params->get_info(loaded.plugin,i,&info),"parameter enumeration failed");
            if(info.id==4){
                engine_found=true;
#if !defined(GPU_NAM_HOST_SHARED_SESSION)
                require(!(info.flags&CLAP_PARAM_IS_AUTOMATABLE),"stamped Engine must be preparation-bound");
#endif
            }
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
