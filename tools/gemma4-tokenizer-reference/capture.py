"""Original offline AutoTokenizer only; never constructs a model."""
import inspect,os
from pathlib import Path
from support import metadata,require,write,sha

def main():
    cpus=[int(x)for x in os.environ['ZG_ALLOWED_CPUS'].split(',')]
    require(len(cpus)==len(set(cpus))==4,'four bounded original CPUs')
    os.sched_setaffinity(0,cpus)
    import transformers,tokenizers
    from transformers import AutoTokenizer
    require(transformers.__version__=='5.17.0'and os.environ['HF_HUB_OFFLINE']=='1','approved original tokenizer version/offline')
    cases=metadata('/probe/cases.json');pins=metadata('/probe/source-pin.json');repeats=[];source=None
    for repeat in range(2):
        tokenizer=AutoTokenizer.from_pretrained('/source',local_files_only=True,trust_remote_code=False)
        require(type(tokenizer).__name__=='GemmaTokenizer','actual original Gemma tokenizer class')
        path=Path(inspect.getsourcefile(type(tokenizer)));raw=path.read_bytes()
        require(0<len(raw)<=2*1024*1024,'bounded actual original class source')
        observed={'bytes':len(raw),'sha256':sha(path),'module':type(tokenizer).__module__,'class':type(tokenizer).__name__}
        if source is None:
            source=observed;(Path('/output')/'tokenization_original.py').write_bytes(raw)
        else:require(source==observed,'same original tokenizer class across fresh constructors')
        rows=[]
        for case in cases['cases']:
            text=case['text'];require(type(text)is str and len(text.encode())<=4096,'bounded genuine Unicode text')
            ids=tokenizer.encode(text,add_special_tokens=True)
            no_special=tokenizer.encode(text,add_special_tokens=False)
            require(len(ids)<=64 and all(type(i)is int and 0<=i<262144 for i in ids),'actual finite source IDs')
            rows.append(dict(name=case['name'],text=text,ids=ids,without_special=no_special,
                decode=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                decode_skip=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)))
        decodes=[]
        for case in cases['decode_tokens']:
            pieces=case['tokens']
            require(type(pieces)is list and 1<=len(pieces)<=64 and all(type(p)is str and len(p.encode())<=93 for p in pieces),
                'declared bounded source token names, not expected IDs')
            ids=tokenizer.convert_tokens_to_ids(pieces)
            require(tokenizer.convert_ids_to_tokens(ids)==pieces and all(type(i)is int and 0<=i<262144 for i in ids),
                'each named byte/special token exists in authentic original vocabulary')
            decodes.append(dict(name=case['name'],tokens=pieces,ids=ids,
                decode=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                decode_skip=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)))
        repeats.append((rows,decodes));del tokenizer
    require(repeats[0]==repeats[1],'independent fresh original tokenizer repetitions exact')
    write('/output/capture.json',dict(schema='zg.gemma4-e2b.original-tokenizer.v2',source=pins,
        transformers_version=transformers.__version__,tokenizers_version=tokenizers.__version__,source_class=source,
        request=cases,repetitions=2,repetition_equal=True,rows=repeats[0][0],decodes=repeats[0][1],
        model_constructed=False,weights_verified=False,rust_tokenizer_qualified=False,
        trained_rust_qualified=False,native_serving_or_performance_qualified=False))
if __name__=='__main__':main()
