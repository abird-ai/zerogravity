"""Independent finite E2B producer schema. No Torch/model imports."""
import hashlib
import struct
from support import require


def expected_producers(call_count=1):
    require(type(call_count)is int and 1<=call_count<=3,'bounded genuine producer call count')
    expected={}
    positions=(0,3)+tuple(range(4,3+call_count))
    def add(layer,label,elements,positions=positions):
        for position in positions:expected[(position,layer,label)]=elements
    for label,elements in [('embedding',1536),('ple.embedding',8960),('ple.projection',8960),('ple.norm',8960)]:
        add(None,label,elements)
    for layer in range(35):
        width=512 if layer%5==4 else 256
        ffn=6144 if layer<15 else 12288
        labels={'block.input':1536,'input.norm':1536,'query.projection':8*width,'query.norm':8*width,
                'query.rotary':8*width,'attention.pv':8*width,'attention.projection':1536,
                'attention.norm':1536,'attention.residual':1536,'ffn.norm':1536,'ffn.gate':ffn,
                'ffn.up':ffn,'ffn.gelu':ffn,'ffn.activation':ffn,'ffn.down':1536,'ffn.post_norm':1536,
                'ffn.residual':1536,'ple.input':256,'ple.gate':256,'ple.gelu':256,'ple.activation':256,
                'ple.projection':1536,'ple.post_norm':1536,'block.output':1536}
        if layer<15:
            labels.update({'key.projection':width,'key.norm':width,'key.rotary':width,'value.norm':width})
            labels['value.projection']=width  # Authentic E2B attention_k_eq_v=false, including full attention.
        for label,elements in labels.items():add(layer,label,elements)
        for position in positions:
            add(layer,'attention.qk',8*(position+1),(position,))
            add(layer,'attention.softmax',8*(position+1),(position,))
    for label,elements in [('final_norm',1536),('head',262144),('logits',262144)]:
        add(None,label,elements,tuple(range(3,3+call_count)))
    return expected


def validate_producers(capture,pins,read):
    legacy=capture['schema']=='zg.gemma4-e2b.producers.v1'
    require(legacy or capture['schema']=='zg.gemma4-e2b.producers.v2','known producer schema')
    call_count=1 if legacy else capture['call_count']
    expected=expected_producers(call_count)
    positions=[0,3]+list(range(4,3+call_count))
    require(capture['source']==pins and type(capture['prompt_rows'])is int and
            capture['prompt_rows']==4 and type(capture['positions'])is list and
            all(type(position)is int for position in capture['positions']) and capture['positions']==positions and
            capture['original_results_unchanged'] is True and capture['trained_rust_qualified'] is False,
            'exact original whole35 producer scope')
    if not legacy:
        calls=capture['calls']
        require(type(calls)is list and all(type(row)is dict and
                    all(type(row.get(key))is int for key in ('call','base','input_rows')) and
                    type(row.get('positions'))is list and all(type(p)is int for p in row['positions']) for row in calls) and
                calls==[{'call':call,'base':0 if call==0 else call+3,
                    'input_rows':4 if call==0 else 1,'positions':[0,3] if call==0 else [call+3]}
                    for call in range(call_count)],'actual initial/cached producer contexts')
    records=capture['records'];at=0;seen=set();whole=hashlib.sha256()
    require(type(records)is list and len(records)==len(expected)<=4096,'complete selected source boundary count')
    for row in records:
        require(type(row['position'])is int and (row['layer']is None or type(row['layer'])is int) and
                type(row['label'])is str and all(type(row[name])is int for name in ('elements','bytes','offset')),
                'exact integer producer metadata; no bool coercion')
        key=(row['position'],row['layer'],row['label'])
        require(key in expected and key not in seen and row['elements']==expected[key] and
                row['bytes']==expected[key]*2 and row['offset']==at,'exact finite source producer role/extent')
        raw=read(at,row['bytes'])
        require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'complete producer words/SHA')
        require(all((word&0x7f80)!=0x7f80 for word, in struct.iter_unpack('<H',raw)), 'finite original BF16 producer')
        seen.add(key);at+=len(raw);whole.update(raw)
    require(seen==set(expected) and 0<at<=32*1024*1024 and
            capture['raw']=={'name':'producers.bin','bytes':at,'sha256':whole.hexdigest()},'complete producer raw/EOF')
