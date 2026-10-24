"""Whole original matrix validation, expected-only; imports no model/framework."""
import hashlib,struct
from support import require

def bf16(raw):
    require(len(raw)%2==0 and all(word[0]&0x7f80!=0x7f80 for word in struct.iter_unpack('<H',raw)),'authentic finite original BF16 words, no cast')

def validate(capture,pins,request,cell_read,raw_read):
    require(capture['schema']=='zg.gemma4-e2b.ordinary-matrix.v1'and capture['source']==pins and capture['request']==request,'same whole source/workload')
    require(capture['profile']=='HfTorchCpuBf16Avx2Eager5.17.0'and capture['transformers']=='5.17.0'and capture['torch_git']=='cf30153c4c131c8164ee7798e5022d810682e2cb','pinned original arithmetic')
    require(capture['eos_ids']==[1,106,50]and capture['plain_vs_observed_equal']is True and capture['trained_rust_qualified']is False and capture['native_serving_chat_or_performance_qualified']is False,'whole plain/observed scope, no promotion')
    require(request['schema']=='zg.gemma4-e2b.ordinary-request.v1'and request['max_output']==8 and request['add_special_tokens']is True and len(request['cases'])==len(capture['cells'])==7,'entire bounded matrix')
    maximum_frontier=0
    for case,entry in zip(request['cases'],capture['cells']):
        require(entry['name']==case['name']and entry['metadata']['name']==case['name']+'.json','exact source cell order')
        cell=cell_read(entry['metadata'])
        require(cell['schema']=='zg.gemma4-e2b.ordinary-cell.v1'and cell['request']==case and cell['repetitions']==2 and cell['repetition_equal']is True and cell['boundary_repetitions']==1,'fresh plain-vs-observed source cell')
        ids=cell['prompt_ids'];require(type(ids)is list and len(ids)==case['rows']and 1<=len(ids)<=57 and all(type(n)is int and 0<=n<262144 for n in ids),'genuine source-tokenized rows')
        calls=cell['calls'];require(1<=len(calls)<=8,'natural finite stop/output count')
        frontier=0;previous=None
        for index,call in enumerate(calls):
            current=ids if index==0 else [previous]
            require(call['call']==index and call['input_ids']==current and call['base']==frontier,'actual greedy feedback/consumed prior')
            frontier+=len(current);require(call['frontier']==frontier<=64,'actual finite source frontier')
            previous=call['selected_id'];require(type(previous)is int and 0<=previous<262144 and call['eos']==(previous in capture['eos_ids']),'genuine greedy/stop')
            require(not call['eos']or index+1==len(calls),'no forced continuation past EOS')
        require(calls[-1]['eos']or len(calls)==8,'only genuine EOS/length completion')
        maximum_frontier=max(maximum_frontier,frontier)
        for kind,limit in [('raw',16*1024*1024),('boundaries',32*1024*1024)]:
            archive=cell[kind];require(archive['name']==case['name']+('.bin'if kind=='raw'else '-boundaries.bin'),'contained exact archive role')
            require(type(archive['bytes'])is int and 0<archive['bytes']<=limit and len(archive['records'])<=4096,'unchanged archive capacity')
            expected=31*len(calls)if kind=='raw'else 70*len(calls)
            require(len(archive['records'])==expected,'complete fullV/all15KV/all35 boundaries')
            all_bytes=raw_read(archive['name'],0,archive['bytes']);require(len(all_bytes)==archive['bytes']and hashlib.sha256(all_bytes).hexdigest()==archive['sha256'],'whole archive SHA/EOF')
            at=0;seen=set()
            for row in archive['records']:
                require(type(row['call'])is int and 0<=row['call']<len(calls),'actual source call ordinal')
                call=calls[row['call']];shape=row['shape']
                if kind=='raw':
                    label=row['label'];require(row['frontier']==call['frontier'],'actual completed frontier')
                    if label=='logits':require(shape==[262144],'complete262144V')
                    else:
                        parts=label.split('.');require(len(parts)==3 and parts[0]=='kv'and parts[2]in ['key','value']and parts[1].isdigit(),'exact actual KV role')
                        owner=int(parts[1]);require(parts[1]==str(owner)and 0<=owner<15 and shape==[call['frontier'],512 if owner%5==4 else 256],'exact owner row width')
                    key=(row['call'],label)
                else:
                    require(row['base']==call['base']and 0<=row['layer']<35 and row['label']in ['block.input','block.output']and shape==[len(call['input_ids']),1536],'whole source layer/input-output plane')
                    key=(row['call'],row['layer'],row['label'])
                count=1
                for n in shape:require(type(n)is int and n>0,'finite shape');count*=n
                require(type(row['offset'])is int and type(row['bytes'])is int and key not in seen and row['offset']==at and row['bytes']==count*2 and row['dtype']=='bf16-le','unique exhaustive BF16 source record')
                raw=all_bytes[at:at+row['bytes']];require(hashlib.sha256(raw).hexdigest()==row['sha256'],'whole record SHA');bf16(raw)
                seen.add(key);at+=len(raw)
            require(at==archive['bytes'],'complete archive without hidden tail')
        # Every cached prior must equal its actual previous producing archive prefix.
        records=cell['raw']['records']
        for call in range(1,len(calls)):
            for owner in range(15):
                for component in ['key','value']:
                    label=f'kv.{owner}.{component}';a=next(r for r in records if r['call']==call-1 and r['label']==label);b=next(r for r in records if r['call']==call and r['label']==label)
                    require(raw_read(cell['raw']['name'],a['offset'],a['bytes'])==raw_read(cell['raw']['name'],b['offset'],a['bytes']),'entire consumed prior continuity')
    require(maximum_frontier==64,'genuine natural matrix must reach source64, never force tokens')
