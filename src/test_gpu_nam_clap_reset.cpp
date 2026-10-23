// Headless CLAP host check for a reset after silence: a host that keeps the plugin
// active and processing silence while stopped, then calls clap_plugin::reset() at
// the start of a render, must get the same output as a fresh activation, with the
// default noise gate on. Usage: gpu-nam-clap-reset-test <GpuNam.clap binary>.
#include <clap/clap.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

thread_local bool audio_thread = false;
bool is_main(const clap_host_t*) { return !audio_thread; }
bool is_audio(const clap_host_t*) { return audio_thread; }
const clap_host_thread_check_t thread_check{is_main, is_audio};
const void* extension(const clap_host_t*, const char* id) {
    return std::strcmp(id, CLAP_EXT_THREAD_CHECK) == 0 ? &thread_check : nullptr;
}
void noop(const clap_host_t*) {}
void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

struct Events {
    std::vector<clap_event_param_value_t> values;
    clap_input_events_t input{
        this,
        [](const clap_input_events_t* e) {
            return static_cast<std::uint32_t>(static_cast<Events*>(e->ctx)->values.size());
        },
        [](const clap_input_events_t* e, std::uint32_t i) -> const clap_event_header_t* {
            const auto& v = static_cast<Events*>(e->ctx)->values;
            return i < v.size() ? &v[i].header : nullptr;
        }};
    void add(clap_id id, double value) {
        clap_event_param_value_t ev{};
        ev.header = {sizeof(ev), 0, CLAP_CORE_EVENT_SPACE_ID, CLAP_EVENT_PARAM_VALUE, 0};
        ev.param_id = id;
        ev.note_id = -1;
        ev.port_index = -1;
        ev.channel = -1;
        ev.key = -1;
        ev.value = value;
        values.push_back(ev);
    }
};
const clap_output_events_t discard{
    nullptr, [](const clap_output_events_t*, const clap_event_header_t*) { return true; }};

constexpr double kRate = 48000.0;
constexpr std::uint32_t kFrames = 128;
constexpr clap_id kEngineParam = 4;   // GpuNamProcessor::kEngine

struct Session {
    const clap_plugin_t* plugin;
    std::vector<float> in_l = std::vector<float>(kFrames), in_r = std::vector<float>(kFrames);
    std::vector<float> out_l = std::vector<float>(kFrames), out_r = std::vector<float>(kFrames);
    std::int64_t steady = 0;

    void block(float amp, std::uint64_t phase0, std::vector<float>* capture) {
        for (std::uint32_t i = 0; i < kFrames; ++i)
            in_l[i] = in_r[i] = amp * std::sin(0.06f * static_cast<float>(phase0 + i));
        float* ip[] = {in_l.data(), in_r.data()};
        float* op[] = {out_l.data(), out_r.data()};
        clap_audio_buffer_t in{ip, nullptr, 2, 0, 0}, out{op, nullptr, 2, 0, 0};
        Events empty;
        clap_process_t p{};
        p.steady_time = steady;
        p.frames_count = kFrames;
        p.audio_inputs = &in;
        p.audio_outputs = &out;
        p.audio_inputs_count = 1;
        p.audio_outputs_count = 1;
        p.in_events = &empty.input;
        p.out_events = &discard;
        audio_thread = true;
        const auto status = plugin->process(plugin, &p);
        audio_thread = false;
        require(status != CLAP_PROCESS_ERROR, "CLAP process failed");
        steady += kFrames;
        if (capture) capture->insert(capture->end(), out_l.begin(), out_l.end());
    }

    // Activate, run `silence` silent blocks, optionally reset, then render a
    // -12 dBFS tone for `tone` blocks; returns channel 0 of the tone section.
    std::vector<float> render(int silence, bool reset, int tone) {
        require(plugin->activate(plugin, kRate, kFrames, kFrames), "activate failed");
        audio_thread = true;
        require(plugin->start_processing(plugin), "start_processing failed");
        audio_thread = false;
        for (int b = 0; b < silence; ++b) block(0.0f, 0, nullptr);
        if (reset) {
            audio_thread = true;
            plugin->reset(plugin);
            audio_thread = false;
        }
        std::vector<float> out;
        for (int b = 0; b < tone; ++b) block(0.25f, static_cast<std::uint64_t>(b) * kFrames, &out);
        audio_thread = true;
        plugin->stop_processing(plugin);
        audio_thread = false;
        plugin->deactivate(plugin);
        return out;
    }
};

double energy(const std::vector<float>& v, std::size_t from, std::size_t n) {
    double e = 0.0;
    for (std::size_t i = from; i < from + n && i < v.size(); ++i) e += static_cast<double>(v[i]) * v[i];
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 64;
    void* lib = nullptr;
    const clap_plugin_entry_t* entry = nullptr;
    const clap_plugin_t* plugin = nullptr;
    int rc = 1;
    try {
        clap_host_t host{CLAP_VERSION, nullptr, "GPU NAM reset check", "Pulp", "", "1",
                         extension, noop, noop, noop};
        lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        require(lib, "CLAP load failed");
        entry = static_cast<const clap_plugin_entry_t*>(dlsym(lib, "clap_entry"));
        require(entry && entry->init(argv[1]), "CLAP entry init failed");
        const auto* factory = static_cast<const clap_plugin_factory_t*>(
            entry->get_factory(CLAP_PLUGIN_FACTORY_ID));
        require(factory && factory->get_plugin_count(factory) == 1, "expected one descriptor");
        const auto* desc = factory->get_plugin_descriptor(factory, 0);
        plugin = factory->create_plugin(factory, &host, desc->id);
        require(plugin && plugin->init(plugin), "plugin init failed");
        const auto* params = static_cast<const clap_plugin_params_t*>(
            plugin->get_extension(plugin, CLAP_EXT_PARAMS));
        const auto* latency = static_cast<const clap_plugin_latency_t*>(
            plugin->get_extension(plugin, CLAP_EXT_LATENCY));
        require(params && latency, "params or latency extension unavailable");
        Events edits;
        edits.add(kEngineParam, 0.0);   // CPU engine; every other parameter at its default
        params->flush(plugin, &edits.input, &discard);

        Session s{plugin};
        constexpr int kTone = 48, kSilence = 400;   // ~0.13 s of tone after ~1 s of silence
        const auto fresh = s.render(0, false, kTone);
        const auto lat = static_cast<std::size_t>(latency->get(plugin));
        const auto stale = s.render(kSilence, false, kTone);
        const auto reset = s.render(kSilence, true, kTone);
        require(fresh.size() > lat + 1024, "latency leaves no tone to compare");

        const double e_fresh = energy(fresh, lat, 1024), e_stale = energy(stale, lat, 1024);
        double max_err = 0.0;
        for (std::size_t i = 0; i < fresh.size(); ++i)
            max_err = std::max(max_err, std::abs(static_cast<double>(reset[i]) - fresh[i]));
        std::cout << "latency=" << lat << " first_1024_energy fresh=" << e_fresh << " stale=" << e_stale
                  << " max|reset-fresh|=" << max_err << '\n';
        require(e_fresh > 1e-3, "fresh activation produced no output");
        // Control: without the reset the silence closed the gate.
        require(e_stale < e_fresh * 0.01, "control: silence did not close the gate");
        require(max_err == 0.0, "output after reset differs from a fresh activation");
        std::cout << "clap_reset_after_silence=passed\n";
        rc = 0;
    } catch (const std::exception& e) {
        audio_thread = false;
        std::cerr << e.what() << '\n';
    }
    if (plugin) plugin->destroy(plugin);
    if (entry) entry->deinit();
    if (lib) dlclose(lib);
    return rc;
}
