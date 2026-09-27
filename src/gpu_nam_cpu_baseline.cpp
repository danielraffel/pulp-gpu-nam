#include "nam_model.hpp"
#include <chrono>
#include <ctime>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
int main(int argc,char** argv){
  std::string path=GPU_NAM_MODEL_PATH; unsigned block=32, lead=1; try { if(argc>1) path=argv[1]; if(argc>2) block=std::stoul(argv[2]); if(argc>3) lead=std::stoul(argv[3]); } catch (...) { std::cerr<<"invalid arguments\n"; return 2; } if((block!=32&&block!=64&&block!=128)||(lead!=1&&lead!=2&&lead!=4&&lead!=8)){std::cerr<<"unsupported block/lead\n";return 2;} constexpr unsigned blocks=96;
  pulp::examples::nam::NamModel m; std::string e; if(!pulp::examples::nam::load_nam(path,m,&e)){std::cerr<<"load_error="<<e<<'\n';return 2;} m.prewarm_block_aligned(block);
  std::vector<float> in(block),out(block); unsigned long long elapsed_ns=0; std::clock_t c0=std::clock(); auto wall0=std::chrono::steady_clock::now(); bool finite=true;
  for(unsigned b=0;b<blocks;++b){for(unsigned i=0;i<block;++i){auto s=float(b*block+i);in[i]=.07f*std::sin(.013f*s)+.02f*std::cos(.037f*s);} auto t=std::chrono::steady_clock::now(); m.process(in.data(),out.data(),block); elapsed_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t).count(); for(float x:out) finite &= std::isfinite(x);}
  auto process_ticks=std::clock()-c0;
  auto wall=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-wall0).count();
  std::cout<<"model="<<path<<" blocks="<<blocks<<" block_size="<<block<<" lead_blocks="<<lead<<" elapsed_process_ns="<<elapsed_ns<<" process_cpu_ticks="<<process_ticks<<" process_cpu_seconds="<<(double(process_ticks)/CLOCKS_PER_SEC)<<" wall_ns="<<wall<<" output_finite="<<(finite?1:0)<<"\n"; return finite?0:3;
}
