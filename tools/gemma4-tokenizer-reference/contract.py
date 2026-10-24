"""Expected-only tokenizer schema; no original/framework/model import."""
from support import require

def validate(capture,pins,cases):
    require(capture['schema']=='zg.gemma4-e2b.original-tokenizer.v2'and capture['source']==pins,
        'same authentic tokenizer source closure')
    require(capture['transformers_version']=='5.17.0'and capture['source_class']['class']=='GemmaTokenizer',
        'actual pinned original tokenizer profile')
    require(capture['request']==cases and capture['repetitions']==2 and capture['repetition_equal']is True,
        'entire declared independently repeated tokenizer workload')
    require(all(capture[k]is False for k in ['model_constructed','weights_verified','rust_tokenizer_qualified',
        'trained_rust_qualified','native_serving_or_performance_qualified']),'no model/weight/math/native authority')
    require(cases['schema']=='zg.gemma4-e2b.tokenizer-cases.v2'and 1<=len(cases['cases'])<=64
        and len(capture['rows'])==len(cases['cases']),'bounded complete tokenizer cases')
    seen=set()
    for actual,case in zip(capture['rows'],cases['cases']):
        require(actual['name']==case['name']and actual['text']==case['text']and case['name']not in seen,'exact input/order/no duplicate')
        seen.add(case['name'])
        require(type(case['text'])is str and len(case['text'].encode())<=4096,'bounded UTF8 source text')
        require(type(actual['ids'])is list and len(actual['ids'])<=64 and
            all(type(n)is int and 0<=n<262144 for n in actual['ids']),'complete finite original source IDs')
        require(actual['without_special']==actual['ids'],'actual identity single postprocessor; no invented BOS')
        require(all(type(actual[k])is str and len(actual[k].encode())<=16384 for k in ['decode','decode_skip']),
            'both complete bounded original decoder modes')
    require(1<=len(cases['decode_tokens'])<=64 and len(capture['decodes'])==len(cases['decode_tokens']),
        'complete bounded independent decoder cohort')
    seen=set()
    for actual,case in zip(capture['decodes'],cases['decode_tokens']):
        require(actual['name']==case['name']and actual['tokens']==case['tokens']and case['name']not in seen,
            'same declared token names/order/no duplicate')
        seen.add(case['name'])
        require(type(case['tokens'])is list and 1<=len(case['tokens'])<=64 and
            all(type(p)is str and len(p.encode())<=93 for p in case['tokens']),'bounded source token names')
        require(type(actual['ids'])is list and len(actual['ids'])==len(case['tokens'])and
            all(type(n)is int and 0<=n<262144 for n in actual['ids']),'actual independently resolved decoder IDs')
        require(all(type(actual[k])is str and len(actual[k].encode())<=16384 for k in ['decode','decode_skip']),
            'complete original byte-fallback decoder outputs in both special modes')
