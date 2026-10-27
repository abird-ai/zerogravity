// Standalone, DEVELOPER-ONLY independent GGUF reference for raw-token-aligned
// top-20 logits. Compiled under ignored target/, never linked into runtime.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <cmath>
static void die(const char*message){std::fprintf(stderr,"ERROR %s\n",message);std::exit(2);}
static std::vector<llama_token> parse_ids(const char *text){
    std::vector<llama_token> out;
    std::stringstream ss(text);
    std::string part;
    while(std::getline(ss,part,',')){
        if (part.empty())die("empty token ID");
        long v=std::strtol(part.c_str(),nullptr,10);
        if(v<0||v>300000)die("invalid token ID");
        out.push_back(static_cast<llama_token>(v));
    }
    if(out.empty() || out.size()>256)die("teacher corpus missing or unbounded");
    return out;
}
static void emit_logits(FILE *file,int step,const float* logits,int n){
    double max=-1e300;
    for(int i=0;i<n;i++)if(logits[i]>max)max=logits[i];
    double denominator=0;
    for(int i=0;i<n;i++)denominator+=std::exp((double)logits[i]-max);
    const double logz=max+std::log(denominator);
    std::vector<int> ord(n);std::iota(ord.begin(),ord.end(),0);
    std::partial_sort(ord.begin(),ord.begin()+20,ord.end(),
       [=](int a,int b){return logits[a]!=logits[b]?logits[a]>logits[b]:a<b;});
    std::fprintf(file,"{\"step\":%d,\"logz\":%.17g,\"top\":[",step,logz);
    for(int i=0;i<20;i++){
        if(i)std::fputc(',',file);
        int idx=ord[i];
        std::fprintf(file,"{\"id\":%d,\"logit\":%.12g,\"logprob\":%.16g}",idx,logits[idx],((double)logits[idx]-logz));
    }
    std::fputs("]}\n",file);
    std::fflush(file);
}
int main(int argc,char**argv){
    if(argc!=5)die("usage: sequence_logits model.gguf prompt teacherIDsCommaSeparated output.jsonl");
    auto teacher=parse_ids(argv[3]);
    FILE *file=std::fopen(argv[4],"wb");if(!file)die("output file cannot be written");
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    auto *model=llama_model_load_from_file(argv[1],mp);if(!model)die("load GGUF");
    auto *vocab=llama_model_get_vocab(model);
    int n=llama_tokenize(vocab,argv[2],std::strlen(argv[2]),nullptr,0,false,true);
    if(n>=0)die("expected tokenizer sizing");
    std::vector<llama_token> prompt(-n);
    int got=llama_tokenize(vocab,argv[2],std::strlen(argv[2]),prompt.data(),prompt.size(),false,true);
    if(got<=0)die("failed raw tokenize"); prompt.resize(got);
    std::fprintf(stderr,"REFERENCE_PROMPT_IDS:");
    for(auto t:prompt)std::fprintf(stderr," %d",t);
    std::fprintf(stderr,"\n");
    auto c=llama_context_default_params();
    c.n_ctx=512;c.n_batch=128;c.n_ubatch=128;
    c.n_threads=8;c.n_threads_batch=8;
    c.type_k=GGML_TYPE_F32;c.type_v=GGML_TYPE_F32;
    c.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto *ctx=llama_init_from_model(model,c);if(!ctx)die("context init");
    for(auto &t:prompt){
        llama_batch batch=llama_batch_get_one(&t,1);
        if(llama_decode(ctx,batch))die("prompt decode");
    }
    const int n_vocab=llama_vocab_n_tokens(vocab);
    for(int step=0;step<(int)teacher.size();step++){
        const float*logits=llama_get_logits(ctx);
        if(!logits)die("logits unavailable");
        emit_logits(file,step,logits,n_vocab);
        llama_token token=teacher[step];
        llama_batch batch=llama_batch_get_one(&token,1);
        if(llama_decode(ctx,batch))die("teacher decode");
    }
    std::fprintf(stderr,"INDEPENDENT_REFERENCE_STEPS=%zu RAW_PROMPT=%zu\n",teacher.size(),prompt.size());
    std::fclose(file);llama_free(ctx);llama_model_free(model);llama_backend_free();
    return 0;
}
