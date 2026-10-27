// Test-only pinned original CPU QK capture. Never linked into Zerogravity.
// Native raw dot tensors; scale is independently read from softmax op metadata.
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
static FILE *capture=nullptr;
static int position=0;
static unsigned captured=0;
static std::map<int,std::vector<float>> dots;
static int layer_of(const ggml_tensor *t) {
    if(!t) return -1;
    for(int layer : {3,7,11,15,19,23})
        if(std::string(t->name)=="kq-"+std::to_string(layer)) return layer;
    return -1;
}
static void fail(const char *s) { std::fprintf(stderr,"REFERENCE_QK_ERROR %s\n",s); std::exit(2); }
static bool observe(ggml_tensor *t,bool ask,void*) {
    if(!t) return false;
    const int layer=layer_of(t);
    if(layer>=0 && t->op==GGML_OP_MUL_MAT) {
        if(ask) return true;
        if(t->type!=GGML_TYPE_F32 || t->ne[0]<position+1 || t->ne[1]!=1 || t->ne[2]!=8 || t->ne[3]!=1 || dots.count(layer)) fail("raw QK geometry/order");
        std::vector<unsigned char> bytes(ggml_nbytes(t));
        ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
        std::vector<float> values;
        for(int h=0;h<8;++h) for(int p=0;p<=position;++p) {
            const size_t offset=h*t->nb[2]+p*t->nb[0];
            if(offset+sizeof(float)>bytes.size()) fail("QK tensor stride");
            float v; std::memcpy(&v,bytes.data()+offset,sizeof(v));
            if(!std::isfinite(v)) fail("nonfinite QK");
            values.push_back(v);
        }
        dots.emplace(layer,std::move(values));
        return true;
    }
    const int src=t->op==GGML_OP_SOFT_MAX ? layer_of(t->src[0]) : -1;
    if(src<0) return false;
    if(ask) return true;
    if(!dots.count(src)) fail("softmax scale before raw QK");
    float scale; std::memcpy(&scale,t->op_params,sizeof(scale));
    if(!std::isfinite(scale) || scale<=0) fail("source softmax scale");
    std::fprintf(capture,"{\"position\":%d,\"layer\":%d,\"query_heads\":8,\"scale\":%.9g,\"dots\":[",position,src,scale);
    const auto &values=dots.at(src);
    for(size_t i=0;i<values.size();++i) {
        if(i) std::fputc(',',capture);
        if(values[i]==0.0f) std::fputs(std::signbit(values[i])?"-0.0":"0.0",capture);
        else std::fprintf(capture,"%.9g",values[i]);
    }
    std::fputs("]}\n",capture); ++captured;
    return true;
}
int main(int argc,char **argv) {
    if(argc!=3) return 2;
    llama_backend_init(); auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
    auto *model=llama_model_load_from_file(argv[1],mp); if(!model) return 2;
    auto p=llama_context_default_params(); p.n_ctx=128; p.n_batch=128; p.n_ubatch=128;
    p.n_threads=8; p.n_threads_batch=8; p.type_k=GGML_TYPE_F32; p.type_v=GGML_TYPE_F32;
    p.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; p.cb_eval=observe; p.no_perf=true;
    auto *ctx=llama_init_from_model(model,p); if(!ctx) return 2;
    capture=std::fopen(argv[2],"wb"); if(!capture) return 2;
    for(llama_token token : {11,353,17,42}) {
        captured=0; dots.clear(); auto batch=llama_batch_get_one(&token,1);
        if(llama_decode(ctx,batch)!=0 || captured!=6 || dots.size()!=6) fail("incomplete causal QK capture");
        ++position;
    }
    std::fclose(capture); llama_free(ctx); llama_model_free(model); llama_backend_free();
    std::fprintf(stderr,"INDEPENDENT_QK_CAPTURED=24 raw_dots_native=1 scaled_scores_reconstructed=1\n");
}
