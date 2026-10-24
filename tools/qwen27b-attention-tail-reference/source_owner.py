"""Original held source owner reused without a model loader."""
from pathlib import Path
import mmap,os,stat
from contract import require,read,metadata,specs,fd_sha,fingerprint,source_header,role_view

class SourceOwner:
    def __init__(self,plan,pins):
        self.plan=plan;self.files=specs(plan,pins);self.fds={};self.before={};self.headers={};self.maps={};self.views={};self.values={};self.versions={};self.readsets={}
        for name in ['config.json','model.safetensors.index.json']:read(Path('/source')/name,pin=self.files[name])
        self.config=metadata('/source/config.json')['text_config'];index=metadata('/source/model.safetensors.index.json')['weight_map']
        t=self.config
        require(t['hidden_size']==5120 and t['intermediate_size']==17408 and t['vocab_size']==248320 and t['num_hidden_layers']==64 and t['rms_norm_eps']==1e-6 and t['hidden_act']=='silu' and t['tie_word_embeddings']is False,'exact original final63 CPU geometry/activation')
        for name in plan['selected_shards']:
            pin=self.files[name];fd=os.open(Path('/source')/name,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);self.fds[name]=fd;m=os.fstat(fd)
            require(stat.S_ISREG(m.st_mode)and m.st_size==pin['bytes'],'selected source extent before any full scan');self.before[name]=fingerprint(m)
            require(fd_sha(fd,0,pin['bytes'])==pin['sha256'] and fingerprint(m)==fingerprint(os.fstat(fd)),'actual full selected source SHA before import')
            self.headers[name]=source_header(fd,pin)
        for role in plan['roles']:
            name=role['name'];shard=role['shard'];require(index[name]==shard,'same actual index→shard role')
            start,header=self.headers[shard];self.views[name]=role_view(start,header,role,self.files[shard]['bytes'])
    def map(self,torch):
        require(sum(v['mapping_bytes']for v in self.views.values())<4*1024**3,'bounded source-only mappings')
        for role in self.plan['roles']:
            name=role['name'];v=self.views[name];m=mmap.mmap(self.fds[role['shard']],v['mapping_bytes'],access=mmap.ACCESS_COPY,offset=v['aligned']);self.maps[name]=m
            x=torch.frombuffer(m,dtype=torch.bfloat16,count=v['bytes']//2,offset=v['view_offset']).reshape(role['shape']);require(x.is_contiguous()and x.device.type=='cpu','authentic full original role view')
            self.values[role['role']]=x;self.versions[role['role']]=x._version;self.readsets[name]=fd_sha(self.fds[role['shard']],v['absolute'],v['bytes'])
    def finish(self):
        for role in self.plan['roles']:
            name=role['name'];v=self.views[name];require(self.values[role['role']]._version==self.versions[role['role']],'source tensor version unchanged')
            h=__import__('hashlib').sha256();view=memoryview(self.maps[name])
            try:
                for i in range(0,v['bytes'],1024**2):h.update(view[v['view_offset']+i:v['view_offset']+min(v['bytes'],i+1024**2)])
            finally:view.release()
            require(h.hexdigest()==self.readsets[name],'mapped source bytes unchanged')
        for name,fd in self.fds.items():require(fingerprint(os.fstat(fd))==self.before[name]and fd_sha(fd,0,self.files[name]['bytes'])==self.files[name]['sha256'],'same full source FD/bytes after math')
