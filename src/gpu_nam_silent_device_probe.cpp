// Manual process-retained diagnostic. Device/plugin/callback storage intentionally
// survive until _Exit: the SDK's void stop() cannot prove error-aware quiescence.
#include "gpu_nam_clap_host_common.hpp"
#include <pulp/audio/device.hpp>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>

namespace {
constexpr unsigned max_frames=4096, target_frames=96256, max_rows=16384;
struct Row { std::uint64_t sample_position=0, entry_ns=0, exit_ns=0; unsigned frames=0; int status=0; };
struct Owner {
    clap_host_t host{CLAP_VERSION,nullptr,"GPU NAM silent device diagnostic","Pulp","","1",extension,noop,noop,noop};
    Loaded loaded;
    LoadedApi api{};
    std::unique_ptr<pulp::audio::AudioSystem> system;
    std::unique_ptr<pulp::audio::AudioDevice> device;
    std::array<std::vector<float>,2> input, capture;
    std::array<std::array<float,max_frames>,2> scratch{};
    std::vector<Row> rows=std::vector<Row>(max_rows);
    Events empty;
    std::atomic<bool> done{false};
    unsigned position=0,row_count=0;
    int error=0;
    bool began=false;
    std::uint64_t first_sample_position=0;
    static std::uint64_t now() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void finish(int code) noexcept {
        error=code;
        if(loaded.processing){loaded.plugin->stop_processing(loaded.plugin);loaded.processing=false;}
        // An ownership protocol for data only, NOT a callback-return barrier.
        // Everything reachable from this callback remains alive until _Exit.
        done.store(true,std::memory_order_release);
    }
    void callback(pulp::audio::BufferView<float>& hardware,const pulp::audio::CallbackContext& ctx) noexcept {
        hardware.clear(); // Never route scratch/plugin output to the real device.
        if(done.load(std::memory_order_acquire))return;
        audio_thread=true;
        if(ctx.buffer_size<=0 || unsigned(ctx.buffer_size)>max_frames ||
           ctx.sample_rate!=48000 || hardware.num_samples()!=unsigned(ctx.buffer_size)) {
            finish(1);audio_thread=false;return;
        }
        const unsigned n=unsigned(ctx.buffer_size);
        if(!began) {
            began=true;first_sample_position=ctx.sample_position;
            if(!loaded.plugin->start_processing(loaded.plugin)){finish(2);audio_thread=false;return;}
            loaded.processing=true;
        }
        if(ctx.sample_position!=first_sample_position+position || row_count>=max_rows ||
           position+n>target_frames+max_frames) {finish(3);audio_thread=false;return;}
        auto& row=rows[row_count++];row.sample_position=ctx.sample_position;row.frames=n;row.entry_ns=now();
        float* ip[]{input[0].data()+position,input[1].data()+position};
        float* op[]{scratch[0].data(),scratch[1].data()};
        clap_audio_buffer_t in{ip,nullptr,2,0,0},out{op,nullptr,2,0,0};
        clap_process_t process{};process.steady_time=position;process.frames_count=n;
        process.audio_inputs=&in;process.audio_outputs=&out;process.audio_inputs_count=1;process.audio_outputs_count=1;
        process.in_events=&empty.input;process.out_events=&discard;
        row.status=loaded.plugin->process(loaded.plugin,&process);
        row.exit_ns=now();
        if(row.status==CLAP_PROCESS_ERROR){finish(4);audio_thread=false;return;}
        for(unsigned ch=0;ch<2;++ch)for(unsigned i=0;i<n;++i) {
            const float v=scratch[ch][i];
            if(!std::isfinite(v)){finish(5);audio_thread=false;return;}
            capture[ch][position+i]=v;
        }
        position+=n;
        if(position>=target_frames)finish(0);
        audio_thread=false;
    }
};
[[noreturn]] void retained_exit(int code) {
    std::cout.flush();std::cerr.flush();std::_Exit(code);
}
}
int main(int argc,char** argv) {
    if(argc==2 && std::strcmp(argv[1],"--list-devices")==0) {
        const auto system=pulp::audio::create_audio_system();
        if(!system)return 1;
        for(const auto& d:system->enumerate_devices())
            std::cout<<d.id<<'\t'<<d.name<<'\t'<<d.max_output_channels<<'\n';
        return 0;
    }
    // No arguments never opens hardware. Require a concrete enumerated device ID.
    if(argc!=5){std::cerr<<"usage: gpu-nam-silent-device-probe <clap-binary> <device-id> <cpu|shared> <output-prefix>\n";return 64;}
    Owner* owner=nullptr;
    try {
        const std::string mode=argv[3];require(mode=="cpu" || mode=="shared","engine must be cpu or shared");
        require(std::strlen(argv[2])>0,"explicit output device ID required");
        owner=new Owner; // Deliberately never destroyed, including failure paths.
        owner->api=load_plugin(owner->loaded,argv[1],owner->host);
        owner->system=pulp::audio::create_audio_system();require(bool(owner->system),"audio system unavailable");
        const auto devices=owner->system->enumerate_devices();
        const auto found=std::find_if(devices.begin(),devices.end(),[&](const auto& d){return d.id==argv[2];});
        require(found!=devices.end() && found->max_output_channels>=2,"explicit stereo output device unavailable");
        owner->device=owner->system->create_device(found->id);require(bool(owner->device),"device creation failed");
        pulp::audio::DeviceConfig config;config.device_id=found->id;config.sample_rate=48000;config.buffer_size=128;
        config.input_channels=0;config.output_channels=2;
        require(owner->device->open(config),"device open failed");
        require(owner->device->sample_rate()==48000,"actual sample rate differs; no implicit resampling");
        const auto actual_info=owner->device->info();
        require(actual_info.id==found->id,"opened device identity differs from requested");
        const unsigned engine=mode=="shared"?1:0;
        Events edits;edits.add(4,engine);edits.add(10,0);edits.add(11,0);edits.add(12,0);
        owner->api.params->flush(owner->loaded.plugin,&edits.input,&discard);
        double requested=-1;require(owner->api.params->get_value(owner->loaded.plugin,4,&requested)&&requested==engine,"inactive Engine write failed");
        require(owner->loaded.plugin->activate(owner->loaded.plugin,48000,1,max_frames),"CLAP activate failed");owner->loaded.active=true;
        require(owner->api.latency->get(owner->loaded.plugin)==1024,"unexpected PDC");
        for(unsigned ch=0;ch<2;++ch){
            owner->input[ch].resize(target_frames+max_frames);owner->capture[ch].resize(target_frames+max_frames);
            for(unsigned i=0;i<target_frames+max_frames;++i)owner->input[ch][i]=.2f*std::sin(float(i)*(.013f+.009f*ch));
        }
        require(owner->device->start([owner](const auto&,auto& out,const auto& ctx){owner->callback(out,ctx);}),"device start failed");
        const auto timeout=std::chrono::steady_clock::now()+std::chrono::seconds(15);
        while(!owner->done.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<timeout)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        require(owner->done.load(std::memory_order_acquire),"device callback timeout; retained graph until process exit");
        require(owner->error==0,"callback failed; retained graph until process exit");
        GpuNamHostProbeSnapshot snapshot;
        require(owner->api.query(&snapshot)==0,"public delivery query unavailable");
        require(snapshot.requested_engine==int(engine)&&snapshot.prepared_engine==int(engine)&&snapshot.active==engine,"engine mismatch");
        require(snapshot.latency_samples==1024 && owner->api.latency->get(owner->loaded.plugin)==1024,"PDC drift");
        if(engine)require(snapshot.gpu_delivered>0 && snapshot.other==0 && snapshot.priming==1 &&
            snapshot.gpu_delivered+snapshot.cpu_fallback+snapshot.priming==owner->position/512,
            "no selected GPU output or incomplete delivery accounting");
        else require(snapshot.gpu_delivered==0 && snapshot.cpu_fallback==0 && snapshot.priming==0 && snapshot.other==0,"CPU mode claimed transport selection");
        const std::string prefix=argv[4];
        std::ofstream rows(prefix+"-callbacks.csv"),audio(prefix+"-audio.csv"),meta(prefix+"-metadata.txt");
        require(bool(rows)&&bool(audio)&&bool(meta),"output files unavailable");
        rows<<"callback,sample_position,frames,observed_entry_ns,observed_exit_ns,clap_status\n";
        for(unsigned i=0;i<owner->row_count;++i){const auto&r=owner->rows[i];rows<<i<<','<<r.sample_position<<','<<r.frames<<','<<r.entry_ns<<','<<r.exit_ns<<','<<r.status<<'\n';}
        audio<<std::setprecision(9)<<"sample,left,right\n";
        double energy=0;
        for(unsigned i=0;i<target_frames;++i){audio<<i<<','<<owner->capture[0][i]<<','<<owner->capture[1][i]<<'\n';energy+=double(owner->capture[0][i])*owner->capture[0][i]+double(owner->capture[1][i])*owner->capture[1][i];}
        meta<<"engine="<<mode<<"\ndevice_id="<<actual_info.id<<"\ndevice_name="<<actual_info.name
            <<"\nrequested_rate=48000\nactual_rate="<<owner->device->sample_rate()<<"\nrequested_frames=128\nopened_frames="<<owner->device->buffer_size()
            <<"\ninput_channels=0\nphysical_output=silence\nhardware_host_timestamp=unavailable\ncallback_clock=steady_clock_observation_only"
            <<"\nauxiliary_audio_workgroup_join=not_attempted\nlifetime=process_retained_until_exit\nteardown_proven=false\nframes_captured="<<owner->position
            <<"\nframes_compared="<<target_frames<<"\ncallbacks="<<owner->row_count<<"\npdc=1024\ngpu_selected="<<snapshot.gpu_delivered
            <<"\ncpu_fallback="<<snapshot.cpu_fallback<<"\npriming="<<snapshot.priming<<"\nother="<<snapshot.other
            <<"\ncallback_allocations=not_measured\nxruns="<<owner->device->xrun_count()<<"\nenergy="<<energy<<'\n';
        rows.close();audio.close();meta.close();require(bool(rows)&&bool(audio)&&bool(meta),"output write failed");
        require(energy>1e-6,"silent captured plugin output");
        std::cout<<"silent_device_capture=passed engine="<<mode<<" gpu_selected="<<snapshot.gpu_delivered<<" cpu_fallback="<<snapshot.cpu_fallback<<" callback_teardown=unproven\n";
        retained_exit(0);
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';retained_exit(1);}
}
