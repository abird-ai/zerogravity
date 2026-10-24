"""Independent bounded archive/feedback contract; no framework/model import."""
import hashlib
import math
import struct
from support import require

VOCAB=262144

def validate(capture,pins,read_record):
    require(capture['schema']=='zg.gemma4-e2b.original-text.v1' and capture['source']==pins and
            capture['profile']=='HfTorchCpuBf16Avx2Eager5.17.0' and
            capture['rust_profile']=='AnalyticF32TextV1_UNQUALIFIED' and
            capture['trained_rust_qualified'] is False and capture['repetitions']==2 and
            capture['repetition_equal'] is True, 'exact repeated original/reference scope')
    require(capture['request']=={'text':'One plus one is','add_special_tokens':True,'max_output':3},'one whole text workload')
    ids=capture['prompt_ids']; require(type(ids)is list and 0<len(ids)<=64 and
        all(type(x)is int and 0<=x<VOCAB for x in ids),'genuine bounded tokenizer prompt')
    eos=capture['eos_ids']; require(type(eos)is list and eos and all(type(x)is int and 0<=x<VOCAB for x in eos),'source EOS IDs')
    calls=capture['calls']; records=capture['records']
    require(type(calls)is list and 0<len(calls)<=3 and len(records)==31*len(calls),'complete call/record count')
    cursor=frontier=0; previous={}
    for call,row in enumerate(calls):
        actual_input=ids if call==0 else [calls[call-1]['selected_id']]
        frontier+=len(actual_input)
        selected=row['selected_id']; require(type(selected)is int and 0<=selected<VOCAB,'actual selected token')
        require(row=={'call':call,'input_ids':actual_input,'frontier':frontier,'selected_id':selected,'eos':selected in eos},'actual natural greedy chain')
        require(not row['eos'] or call==len(calls)-1,'never force continuation past source EOS')
        require(call+1<len(calls) or row['eos'] or len(calls)==3,'no truncated healthy trajectory')
        state={}
        for index,record in enumerate(records[31*call:31*(call+1)]):
            if index==0:label='logits';shape=[VOCAB]
            else:
                owner=(index-1)//2; component='key' if index%2 else 'value';width=512 if owner%5==4 else 256
                label=f'kv.{owner}.{component}';shape=[frontier,width]
            count=math.prod(shape)*4
            require(record['call']==call and record['label']==label and record['shape']==shape and
                    record['offset']==cursor and record['bytes']==count and
                    record['dtype']=='original-bf16-widened-f32-le','source owner shape/order/whole extent')
            raw=read_record(cursor,count)
            require(len(raw)==count and hashlib.sha256(raw).hexdigest()==record['sha256'],'complete original record SHA')
            values=[]
            for word, in struct.iter_unpack('<I',raw):
                value=struct.unpack('<f',struct.pack('<I',word))[0]
                require(word&0xffff==0 and math.isfinite(value),'actual finite widened BF16 source bits')
                if index==0:values.append(value)
            if index==0:
                actual=max(range(VOCAB),key=values.__getitem__)
                require(actual==selected,'independent full-vocabulary greedy with first numeric tie')
            else:
                if call:require(raw[:len(previous[label])]==previous[label],'actual immutable consumed-prior continuity')
                state[label]=raw
            cursor+=count
        previous=state
    require(capture['raw']['name']=='raw.bin' and capture['raw']['bytes']==cursor and
            0<cursor<=16*1024*1024,'complete archive/EOF; no unspecified fields/records')
