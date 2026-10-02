#include "gpu_nam_paced_options.hpp"
#include "gpu_nam_paced_delivery.hpp"
#include "gpu_nam_paced_error.hpp"
#include <string>
#include <string_view>
#include <iostream>
#include <limits>
int main() {
    using namespace pulp::examples;
    unsigned count=0;
    auto check=[&](bool good){++count;return good;};
    GpuNamPacedOptions o;
    if (!check(o.valid(32,2))) return 1;
    o.enabled=true;
    if (!check(!o.valid(32,2))) return 2;
    o.sidecar="receipt.csv";
    if (!check(o.valid(32,2)&&o.blocks(32)==15000)) return 3;
    for (const auto arg : {"--blocks=0","--blocks=-1","--blocks=4x","--duration-seconds=0","--sidecar="}) {
        GpuNamPacedOptions invalid;
        if (!check(!invalid.parse(arg))) return 4;
    }
    if (!check(o.parse("--blocks=1000000")&&o.valid(32,2)&&!o.valid(128,2))) return 5;
    if (!check(o.parse("--trace") && o.trace)) return 5;
    if (!check(o.parse("--capacity=8") && o.capacity == 8 && o.capacity_explicit &&
               o.valid(32, 4) && !o.valid(32, 8))) return 5;
    for (const auto arg : {"--capacity=0", "--capacity=3", "--capacity=16", "--capacity=abc"}) {
        GpuNamPacedOptions invalid_capacity;
        if (!check(!invalid_capacity.parse(arg))) return 5;
    }
    if (!check(o.parse("--duration-seconds=10")&&!o.valid(32,2))) return 6;
    for (auto frames : {32u,64u,128u,512u}) {
        if (!check(paced_offset_ns(48000,frames)==std::uint64_t(frames)*1'000'000'000)) return 7;
        if (!check(paced_offset_ns(3,frames)==std::uint64_t(frames)*62500)) return 8;
    }
    if (!check(paced_offset_ns(1,32)==666666&&paced_offset_ns(3,32)==2000000)) return 9;
    if (!check(paced_sample_matches(.2f,.2f)&&!paced_sample_matches(.45f,.2f)&&
               !paced_sample_matches(std::numeric_limits<float>::quiet_NaN(),.2f)&&
               !paced_sample_matches(.2f,std::numeric_limits<float>::infinity()))) return 10;
    GpuNamPacedOptions staged;
    if (!check(staged.parse("--staged-gpu") && !staged.valid(128,4))) return 11;
    staged.enabled=true; staged.sidecar="staged.csv";
    if (!check(staged.valid(128,4))) return 12;
    if (!check(staged.parse("--inject-forward-failure") && staged.valid(128,4))) return 13;
    if (!check(staged.parse("--force-fallback") && !staged.valid(128,4))) return 14;
    staged.inject_forward_failure=false;
    if (!check(staged.valid(128,4))) return 15;
    staged.cpu_only=true;
    if (!check(!staged.valid(128,4))) return 16;
    const PacedDeliveryCounts empty{};
    PacedDeliveryCounts totals{};
    for (std::size_t i=0;i<empty.size();++i) {
        auto after=totals; ++after[i];
        const auto selected=paced_delivery_selection(totals,after);
        if (!check(selected==static_cast<PacedSelection>(i))) return 17;
        if (!check(paced_record_delivery(totals,selected)&&totals==after)) return 18;
    }
    if (!check(paced_delivery_reconciles(empty,totals,totals,7))) return 19;
    if (!check(!paced_delivery_reconciles(empty,totals,totals,6))) return 20;
    if (!check(!paced_delivery_reconciles(empty,totals,totals,8))) return 21;
    if (!check(paced_delivery_selection(empty,empty)==PacedSelection::AccountingError)) return 22;
    auto two=empty; two[0]=2;
    if (!check(paced_delivery_selection(empty,two)==PacedSelection::AccountingError)) return 23;
    two[0]=1; two[2]=1;
    if (!check(paced_delivery_selection(empty,two)==PacedSelection::AccountingError)) return 24;
    auto reset=totals; reset[0]=0;
    if (!check(paced_delivery_selection(totals,reset)==PacedSelection::AccountingError)) return 25;
    auto wrapped=empty; wrapped[0]=std::numeric_limits<std::uint64_t>::max();
    if (!check(paced_delivery_selection(wrapped,empty)==PacedSelection::AccountingError)) return 26;
    if (!check(!paced_record_delivery(totals,PacedSelection::AccountingError)&&
               !paced_record_delivery(totals,PacedSelection::CpuBaseline))) return 27;
    auto wrong=totals; --wrong[0]; ++wrong[1];
    if (!check(!paced_delivery_reconciles(empty,totals,wrong,7))) return 28;
    if (!check(!paced_delivery_reconciles(totals,empty,empty,0))) return 29;
    if (!check(paced_delivery_reconciles(totals,totals,empty,0))) return 30;
    if (!check(std::string_view(paced_selection_name(PacedSelection::WorkerOutput))=="worker_output"&&
               std::string_view(paced_selection_name(PacedSelection::Silence))=="silence"&&
               std::string_view(paced_selection_name(PacedSelection::CpuBaseline))=="cpu_baseline")) return 31;
    struct Snapshot {
        std::uint64_t gpu_blocks=1,worker_output_blocks=2,cpu_fallback_blocks=3,
            silence_blocks=4,passthrough_blocks=5,priming_blocks=6,invalid_blocks=7;
    };
    if (!check(paced_delivery_counts(Snapshot{})==PacedDeliveryCounts{1,2,3,4,5,6,7})) return 32;
    auto csv_fields=[](const PacedErrorSummary& error) {
        FILE* file=std::tmpfile();
        if (!file) return std::string("tmpfile_failed");
        error.write_csv_fields(file); std::rewind(file);
        char buffer[512]{}; const auto size=std::fread(buffer,1,sizeof(buffer),file);
        std::fclose(file); return std::string(buffer,size);
    };
    PacedErrorSummary clean;
    clean.compare(.25f,.25f,0,0);
    if (!check(clean.mismatches==0 && csv_fields(clean)==",0,0,0,,,,,none")) return 33;
    PacedErrorSummary finite;
    finite.compare(.5f,.25f,1,17);
    finite.compare(.75f,.25f,0,32);
    if (!check(finite.mismatches==2 && finite.nonfinite_mismatches==0 &&
               finite.max_finite_abs_error==.5 && finite.first_channel==1 && finite.first_frame==17 &&
               csv_fields(finite)==",2,0,0.5,1,17,0.5,0.25,finite")) return 34;
    PacedErrorSummary nonfinite;
    nonfinite.compare(std::numeric_limits<float>::quiet_NaN(),.25f,1,3);
    nonfinite.compare(.5f,std::numeric_limits<float>::infinity(),0,4);
    nonfinite.compare(.75f,.25f,0,5);
    if (!check(nonfinite.mismatches==3 && nonfinite.nonfinite_mismatches==2 &&
               nonfinite.max_finite_abs_error==.5 && nonfinite.first_channel==1 &&
               nonfinite.first_frame==3 && std::string_view(nonfinite.first_kind())=="actual_nonfinite" &&
               csv_fields(nonfinite).find(",1,3,nan,0.25,actual_nonfinite")!=std::string::npos)) return 35;
    PacedErrorSummary expected_nonfinite,both_nonfinite;
    expected_nonfinite.compare(0,std::numeric_limits<float>::infinity(),0,0);
    both_nonfinite.compare(std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),0,0);
    if (!check(expected_nonfinite.mismatches==1 && std::string_view(expected_nonfinite.first_kind())=="expected_nonfinite" &&
               both_nonfinite.mismatches==1 && std::string_view(both_nonfinite.first_kind())=="both_nonfinite")) return 36;
    std::cout<<count<<" paced option/schedule/oracle/delivery controls passed\n";
}
