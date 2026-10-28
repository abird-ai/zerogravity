// Test-only independent attention RoPE source capture, never linked to Zerogravity.
// Decode four raw IDs individually on the pinned external CPU library.
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <vector>
static FILE *capture=nullptr;
static int position=0;
static unsigned captured=0;
static bool observe(ggml_tensor *t, bool ask, void*) {
    if (!t || t->op!=GGML_OP_ROPE || captured>=2) return false;
    // First two RoPE outputs are Q/K of the first full-attention block (3).
    if(t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t)) return false;
    const auto n=ggml_nelements(t);
    if(n!=2048 && n!=512) return false;
    if(ask) return true;
    std::vector<float> values(static_cast<size_t>(n));
    ggml_backend_tensor_get(t,values.data(),0,values.size()*sizeof(float));
    std::fprintf(capture,"{\"position\":%d,\"kind\":\"%s\",\"values\":[",position,n==2048?"query":"key");
    for(size_t i=0;i<values.size();++i) {
        if(i) std::fputc(',',capture);
        if(!std::isfinite(values[i])) return false;
        if(values[i]==0.0f) std::fputs(std::signbit(values[i])?"-0.0":"0.0",capture);
        else std::fprintf(capture,"%.9g",values[i]);
    }
    std::fputs("]}\n",capture);
    ++captured;
    return true;
}
int main(int argc,char **argv) {
    if(argc!=3) return 2;
    llama_backend_init();
    auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
    auto *model=llama_model_load_from_file(argv[1],mp);
    if(!model) return 2;
    auto p=llama_context_default_params();
    p.n_ctx=128; p.n_batch=128; p.n_ubatch=128;
    p.n_threads=8; p.n_threads_batch=8;
    p.type_k=GGML_TYPE_F32; p.type_v=GGML_TYPE_F32;
    p.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.cb_eval=observe; p.no_perf=true;
    auto *ctx=llama_init_from_model(model,p);
    if(!ctx) return 2;
    capture=std::fopen(argv[2],"wb"); if(!capture) return 2;
    for(llama_token token : {11,353,17,42}) {
        captured=0;
        auto batch=llama_batch_get_one(&token,1);
        if(llama_decode(ctx,batch)!=0 || captured!=2) return 2;
        ++position;
    }
    std::fclose(capture);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    std::fprintf(stderr,"INDEPENDENT_ROPE_CAPTURED=8\n");
}
