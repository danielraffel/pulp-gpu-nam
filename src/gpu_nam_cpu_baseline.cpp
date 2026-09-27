#include "nam_model.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
int main(int argc,char** argv){
  std::string path=GPU_NAM_MODEL_PATH; if(argc>1) path=argv[1]; unsigned block=argc>2?std::stoul(argv[2]):32; constexpr unsigned blocks=96;
  pulp::examples::nam::NamModel m; std::string e; if(!pulp::examples::nam::load_nam(path,m,&e)){std::cerr<<"load_error="<<e<<'\n';return 2;} m.prewarm_block_aligned(block);
  std::vector<float> in(block),out(block); unsigned long long cpu_ns=0; auto wall0=std::chrono::steady_clock::now();
  for(unsigned b=0;b<blocks;++b){for(unsigned i=0;i<block;++i){auto s=float(b*block+i);in[i]=.07f*std::sin(.013f*s)+.02f*std::cos(.037f*s);} auto t=std::chrono::steady_clock::now(); m.process(in.data(),out.data(),block); cpu_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t).count();}
  auto wall=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-wall0).count();
  std::cout<<"model="<<path<<" blocks="<<blocks<<" block_size="<<block<<" cpu_process_ns="<<cpu_ns<<" wall_ns="<<wall<<" output_finite="<<(std::isfinite(out[0])?1:0)<<"\n"; return 0;
}
