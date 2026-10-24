"""Borrow actual layer9 ATen values. Never replace a tensor/result or math op."""
from contract import require


def make_observer(torch, owner, archive):
    from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

    def meta(t):
        return {'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'device':t.device.type}

    class Inner(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls=[]
            self.reset()
        def reset(self):
            self.seen=set();self.bmm=0;self.raw=0;self.norm=0;self.beta=0;self.ops=[]
        def save(self,label,value):
            require(label not in self.seen,'one actual inner producer per initial prefix')
            with _disable_current_modes():
                archive.add('inner.'+label,9,value)
            self.seen.add(label)
        def complete(self):
            require(self.bmm==6 and len(self.seen)==10,f'original inner observations bmm={self.bmm} seen={sorted(self.seen)}')
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            selected=(owner.active is not None and owner.active['group']=='layer.9' and str(func)=='aten.matmul.default')
            if selected:print(f'ZG_INITIAL_CORE_MATMUL_BEGIN repeat={archive.tag["repeat"]} ordinal={self.bmm}',flush=True)
            result=func(*args,**(kwargs or {}))
            if selected:print(f'ZG_INITIAL_CORE_MATMUL_END repeat={archive.tag["repeat"]} ordinal={self.bmm}',flush=True)
            if owner.active is None or owner.active['group']!='layer.9' or not isinstance(result,torch.Tensor) or result.device.type!='cpu':
                return result
            shape=list(result.shape);name=str(func)
            require(len(self.ops)<512,'bounded original inner operation metadata')
            self.ops.append({'op':name,'shape':shape,'args':[{'shape':list(x.shape),'dtype':str(x.dtype)} for x in args if isinstance(x,torch.Tensor)]})
            if name=='aten.conv1d.default' and shape==[1,10240,7]:
                self.save('convolution',result)
            elif name=='aten.silu.default' and shape==[1,10240,4]:
                self.save('activated',result)
            elif name=='aten.to.dtype' and shape==[1,48,1,128] and result.dtype==torch.float32 and args[0].dtype==torch.bfloat16:
                if self.raw<2:self.save(('query','key')[self.raw],result)
                self.raw+=1
            elif name=='aten.mul.Tensor' and shape==[1,48,1,128] and result.dtype==torch.float32:
                other=args[1]
                if isinstance(other,torch.Tensor) and list(other.shape)==[1,48,1,1]:
                    require(self.norm<2,'original query/key L2 multiply count')
                    self.save(('normalized_query','normalized_key')[self.norm],result);self.norm+=1
                elif isinstance(other,(int,float)):
                    require(other==128**-0.5,'genuine original query scale')
                    self.save('scaled_query',result)
            elif name=='aten.mul.Tensor' and shape==[1,48,64,128] and self.bmm==0:
                if isinstance(args[1],torch.Tensor) and list(args[1].shape)==[1,48,64,1]:
                    if self.beta==0:self.save('beta_value',result)
                    self.beta+=1
            elif name=='aten.matmul.default':
                require(len(args)==2 and len(shape) in (4,5) and shape[:2]==[1,48] and result.dtype==torch.float32,'actual original grouped F32 BMM')
                self.calls.append({'repeat':archive.tag['repeat'],'source_matmul':self.bmm,'operation':name,
                                   'left':meta(args[0]),'right':meta(args[1]),'output':meta(result)})
                if self.bmm==1:self.save('query_key',result)
                self.bmm+=1
            elif name=='aten.add.Tensor' and shape==[1,48,64,128] and self.bmm==5:
                self.save('core_f32',result)
            return result
    return Inner()
