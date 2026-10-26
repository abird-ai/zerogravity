// Test-only independent capture of 24 Qwen3.5 trained layer stages.
// Linked against a separately version-pinned llama.cpp, never Zerogravity.
// argv: <GGUF> <raw prompt UTF8> <output JSONL> [physical_prefill_batch]
//
// For long inputs, only the final prompt token is decoded alone so the
// callback tensors are one-token vectors. Earlier prefix batches execute
// independently on the pinned CPU reference and must match raw tokenizer
// IDs; this is a source-numerical diagnostic, not acceptance substitution.
// CPU only, F32 KV, FlashAttention disabled, one raw token per decode.
// Note: post-FFN residual has no named callback in the source graph; the
// converter reconstructs it from independent source FFN/attention outputs.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static FILE *capture=nullptr;
static int current_position=-1, last_position=-1;
static unsigned captured=0;
static bool starts(const std::string&v,const char *prefix) {
    return v.rfind(prefix,0)==0;
}
static bool selected(const std::string&n) {
    return n=="model.input_embed" || n=="result_norm" ||
        starts(n,"attn_norm-") || starts(n,"linear_attn_out-") ||
        starts(n,"attn_output-") || starts(n,"attn_residual-") ||
        starts(n,"attn_post_norm-") || starts(n,"ffn_out-");
}
static bool observe(struct ggml_tensor *t,bool ask,void*) {
    if (!t || current_position!=last_position) return false;
    std::string name(t->name);
    if (!selected(name))return false;
    int64_t n=1;
    for(int i=0;i<4;i++){
        if(t->ne[i]<=0 || n>131072/t->ne[i])return false;
        n*=t->ne[i];
    }
    if (t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t))return false;
    if (ask)return true;
    std::vector<float>values(static_cast<size_t>(n));
    ggml_backend_tensor_get(t,values.data(),0,values.size()*sizeof(float));
    std::fprintf(capture,"{\"name\":\"%s\",\"position\":%d,\"values\":[",name.c_str(),current_position);
    for(size_t i=0;i<values.size();i++){
        if(i)std::fputc(',',capture);
        if(!std::isfinite(values[i])) {std::fputs("null",capture);continue;}
        std::fprintf(capture,"%.9g",values[i]);
    }
    std::fputs("]}\n",capture);
    captured++;
    return true;
}
static int fail(const char *error) {
    std::fprintf(stderr,"REFERENCE_LAYER_ERROR: %s\n",error);
    return 2;
}
int main(int argc,char **argv) {
    if(argc!=4 && argc!=5)return fail("usage: capture GGUF raw-prompt output.jsonl [prefix_batch]");
    const int prefix_batch=argc==5 ? std::atoi(argv[4]) : 1;
    if(prefix_batch<1 || prefix_batch>512) return fail("prefix batch out of bounds");
    llama_backend_init();
    auto modelparams=llama_model_default_params();
    modelparams.n_gpu_layers=0;
    auto *model=llama_model_load_from_file(argv[1],modelparams);
    if(!model)return fail("model load");
    const auto *vocab=llama_model_get_vocab(model);
    int wanted=llama_tokenize(vocab,argv[2],std::strlen(argv[2]),nullptr,0,false,true);
    if(wanted>=0 || wanted<-32768)return fail("source tokenize sizing");
    std::vector<llama_token> ids(static_cast<size_t>(-wanted));
    int actual=llama_tokenize(vocab,argv[2],std::strlen(argv[2]),ids.data(),ids.size(),false,true);
    if(actual<=0 || actual!=static_cast<int>(ids.size()))return fail("source tokenize");
    last_position=actual-1;
    auto settings=llama_context_default_params();
    settings.n_ctx=std::max(512,actual+64);
    settings.n_batch=std::max(128,prefix_batch);
    settings.n_ubatch=std::max(128,prefix_batch);
    settings.n_threads=8;
    settings.n_threads_batch=8;
    settings.type_k=GGML_TYPE_F32;
    settings.type_v=GGML_TYPE_F32;
    settings.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    settings.cb_eval=observe;
    settings.cb_eval_user_data=nullptr;
    auto *ctx=llama_init_from_model(model,settings);
    if(!ctx)return fail("context initialization");
    capture=std::fopen(argv[3],"wb");
    if(!capture)return fail("output file");
    std::fprintf(stderr,"REFERENCE_TOKENS:");
    for(auto t:ids)std::fprintf(stderr," %d",t);
    std::fputc('\n',stderr);
    int i=0;
    while(i<actual-1){
        const int n=std::min(prefix_batch,actual-1-i);
        current_position=i+n-1;  // never equals last_position during prefix
        auto batch=llama_batch_get_one(ids.data()+i,n);
        if(llama_decode(ctx,batch)!=0)return fail("source batched prefix decode");
        i+=n;
    }
    // Exactly one final source-trained token: full 146 stage vector shape.
    current_position=last_position;
    llama_token last=ids[actual-1];
    auto last_batch=llama_batch_get_one(&last,1);
    if(llama_decode(ctx,last_batch)!=0)return fail("source final-token decode");
    std::fprintf(stderr,"REFERENCE_PREFIX_BATCH=%d SOURCE_INPUT_TOKENS=%d\n",
        prefix_batch,actual);
    std::fflush(capture);
    std::fclose(capture);
    capture=nullptr;
    std::fprintf(stderr,"REFERENCE_LAYER_CAPTURED=%u LAST_POSITION=%d\n",captured,last_position);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
