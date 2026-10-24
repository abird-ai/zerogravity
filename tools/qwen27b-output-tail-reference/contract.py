"""Finite original final-layer tail protocol; no Torch/model import."""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

HERE=Path(__file__).resolve().parent
IMAGE='sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
IMAGE_DIGEST='sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
MODEL_SHA='762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
TORCH_GIT='cf30153c4c131c8164ee7798e5022d810682e2cb'
SOURCE_REGISTRY_CANONICAL_SHA='2bb924d3c486e14b66cc43f9314000970b627f37a53dc8fc08e8f08079e68a2f'
STAGES={'input':5120,'normalized_ffn':5120,'gate':17408,'up':17408,'gated':17408,'branch':5120,'residual':5120,'normalized':5120,'logits':248320}
FILES=('contract.py','capture.py','run_reference.py','validate_reference.py','owned_container_v4.py','tensor_bits.py','source-pin.json','inputs.json')

def require(ok,why):
    if not ok:raise ValueError(why)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def fingerprint(m):return [m.st_dev,m.st_ino,m.st_size,m.st_mtime_ns,m.st_ctime_ns]
def unique(pairs):
    out={}
    for k,v in pairs:require(k not in out,'duplicate JSON key');out[k]=v
    return out
def read(path,cap=4*1024**2,pin=None):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,'bounded regular held file')
        with os.fdopen(os.dup(fd),'rb')as f:raw=f.read(cap+1)
        require(len(raw)==before.st_size<=cap and fingerprint(before)==fingerprint(os.fstat(fd)),'complete held file/EOF/identity')
    finally:os.close(fd)
    if pin is not None:require(len(raw)==pin['bytes'] and digest(raw)==pin['sha256'],'full content pin')
    return raw
def metadata(path,cap=4*1024**2,pin=None):return json.loads(read(path,cap,pin),object_pairs_hook=unique)
def write(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<=4*1024**2,'bounded output JSON')
    with Path(path).open('xb')as f:os.fchmod(f.fileno(),0o600);f.write(raw)
def sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fd_sha(fd,offset,count):
    h=hashlib.sha256()
    for i in range(0,count,1024**2):
        b=os.pread(fd,min(1024**2,count-i),offset+i);require(len(b)==min(1024**2,count-i),'complete positioned FD read');h.update(b)
    return h.hexdigest()
def package():return {n:{'bytes':(HERE/n).stat().st_size,'sha256':sha(HERE/n)}for n in FILES}
def probe(seed):
    require(type(seed)is int and 0<seed<65536,'finite external probe seed')
    return b''.join(((0xbf00 if (j+seed)%2 else 0x3f00)|((j*13+seed)%128)).to_bytes(2,'little')for j in range(5120))
def finite(raw):return len(raw)%2==0 and all(w&0x7f80!=0x7f80 for w, in struct.iter_unpack('<H',raw))
def specs(plan,pins):
    require(digest(json.dumps(pins,sort_keys=True).encode())==SOURCE_REGISTRY_CANONICAL_SHA,'compiled approved whole original source registry')
    require(plan['config_sha256']=='191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab'and plan['index_sha256']=='77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df','pinned original config/index namespaces')
    require(plan['schema']=='zg.original27b-final-tail-inputs.v1' and plan['snapshot']==pins['snapshot']=='98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e','exact original27B snapshot')
    require(plan['repository']==pins['repository']=='Qwen/Qwen3.8-27B' and plan['revision']==pins['revision']=='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','exact original revision')
    require(plan['torch_call_shape']==[1,5120] and plan['repetitions']==2 and plan['archive_cap']==8*1024**2 and plan['record_cap']==2*1024**2,'closed row/repetition/transport bounds')
    expected=[('ffn_norm','model.language_model.layers.63.post_attention_layernorm.weight',[5120]),('gate','model.language_model.layers.63.mlp.gate_proj.weight',[17408,5120]),('up','model.language_model.layers.63.mlp.up_proj.weight',[17408,5120]),('down','model.language_model.layers.63.mlp.down_proj.weight',[5120,17408]),('final_norm','model.language_model.norm.weight',[5120]),('head','lm_head.weight',[248320,5120])]
    require([(r['role'],r['name'],r['shape'])for r in plan['roles']]==expected,'exact final63 source role order/extent')
    require(plan['selected_shards']==['model-00016-of-00018.safetensors','model-00017-of-00018.safetensors','model-00018-of-00018.safetensors'],'exact selected source shards')
    require(plan['probes']==[{'cohort':0,'row':0,'seed':11},{'cohort':1,'row':0,'seed':353},{'cohort':1,'row':1,'seed':977},{'cohort':1,'row':2,'seed':42}],'declared1+3 independent row cohort')
    require(plan['scope']=={'external_boundary_only':True,'input_trained_full_model_producer_qualified':False,'whole_model_qualified':False,'state_root_qualified':False,'gpu_qualified':False,'serving_qualified':False,'performance_qualified':False},'no widened numerical or publication permit')
    files={r['name']:r for r in pins['files']};require(len(files)==31 and sum(r['bytes']for n,r in files.items()if n.endswith('.safetensors'))==55563006776,'all original31 files/18 shards')
    return files

def source_header(fd,file_pin):
    before=os.fstat(fd);require(stat.S_ISREG(before.st_mode)and before.st_size==file_pin['bytes'],'selected held source extent')
    length=os.pread(fd,8,0);require(len(length)==8,'source header length');n,=struct.unpack('<Q',length)
    require(1<n<=8*1024**2 and 8+n<=before.st_size,'bounded source header before body read')
    raw=os.pread(fd,n,8);require(len(raw)==n and fingerprint(before)==fingerprint(os.fstat(fd)),'complete immutable source header')
    return 8+n,json.loads(raw,object_pairs_hook=unique)

def role_view(header_start,header,role,file_bytes):
    row=header[role['name']];shape=role['shape'];size=2
    for dim in shape:require(type(dim)is int and dim>0,'finite role dimension');size*=dim
    offsets=row['data_offsets'];require(row['dtype']==role['dtype']=='BF16' and row['shape']==shape and len(offsets)==2 and all(type(v)is int for v in offsets)and 0<=offsets[0]<offsets[1] and offsets[1]-offsets[0]==size and header_start+offsets[1]<=file_bytes,'full original role/header interval')
    absolute=header_start+offsets[0];page=os.sysconf('SC_PAGE_SIZE');require(page==4096,'reviewed mapping page')
    aligned=absolute-absolute%page
    return {'absolute':absolute,'bytes':size,'aligned':aligned,'view_offset':absolute-aligned,'mapping_bytes':size+absolute-aligned}
