"""CPU-only hostile archive fixtures; no model/source payload imports."""
import copy
import hashlib
import struct
import unittest
from contract import validate, VOCAB


def fixture():
    data=bytearray(); records=[]; calls=[]
    for call in range(3):
        frontier=2+call;selected=2
        calls.append({'call':call,'input_ids':[1,2] if call==0 else [2],
                      'frontier':frontier,'selected_id':selected,'eos':False})
        for i in range(31):
            if i==0:
                label='logits';shape=[VOCAB]
                raw=bytearray(4*VOCAB);struct.pack_into('<f',raw,8,1.0)
            else:
                owner=(i-1)//2;width=512 if owner%5==4 else 256
                label=f'kv.{owner}.{"key" if i%2 else "value"}';shape=[frontier,width]
                raw=struct.pack('<f',float(owner))*frontier*width
            records.append({'call':call,'label':label,'shape':shape,'offset':len(data),'bytes':len(raw),
                'dtype':'original-bf16-widened-f32-le','sha256':hashlib.sha256(raw).hexdigest()})
            data.extend(raw)
    pins={'explicit':'fixture source only'}
    capture={'schema':'zg.gemma4-e2b.original-text.v1','source':pins,
             'profile':'HfTorchCpuBf16Avx2Eager5.17.0','rust_profile':'AnalyticF32TextV1_UNQUALIFIED',
             'trained_rust_qualified':False,'repetitions':2,'repetition_equal':True,
             'request':{'text':'One plus one is','add_special_tokens':True,'max_output':3},
             'prompt_ids':[1,2],'eos_ids':[9],'calls':calls,'records':records,
             'raw':{'name':'raw.bin','bytes':len(data)}}
    return capture,pins,data

class Contract(unittest.TestCase):
    def test_full_v_all_owners_actual_feedback_and_prior(self):
        c,p,d=fixture();validate(c,p,lambda at,n:d[at:at+n])

    def test_wrong_scope_shape_owner_feedback_and_truncation(self):
        c,p,d=fixture()
        def check(change):
            bad=copy.deepcopy(c);change(bad)
            with self.assertRaises(ValueError):validate(bad,p,lambda at,n:d[at:at+n])
        for change in [lambda x:x.update(repetition_equal=False),
                       lambda x:x.update(trained_rust_qualified=True),
                       lambda x:x['calls'][1].update(input_ids=[42]),
                       lambda x:x['calls'][0].update(selected_id=1),
                       lambda x:x['records'][2].update(label='kv.1.value'),
                       lambda x:x['records'][0].update(bytes=4),
                       lambda x:x['records'][-1].update(offset=0),
                       lambda x:x['raw'].update(bytes=len(d)+4),
                       lambda x:x.update(calls=x['calls'][:1],records=x['records'][:31])]:check(change)

    def test_full_record_hash_and_continuity_are_independent(self):
        c,p,d=fixture()
        r=c['records'][32]; at=r['offset']
        # Corrupt cached owner prior and honestly reseal that record: continuity still refuses.
        struct.pack_into('<f',d,at,2.0)
        r['sha256']=hashlib.sha256(d[at:at+r['bytes']]).hexdigest()
        with self.assertRaisesRegex(ValueError,'continuity'):validate(c,p,lambda at,n:d[at:at+n])

    def test_widened_bf16_words_and_greedy_numeric_signedzero_ties(self):
        c,p,d=fixture();r=c['records'][0]
        struct.pack_into('<I',d,0,0x80000000)
        struct.pack_into('<f',d,8,0.0)
        r['sha256']=hashlib.sha256(d[:r['bytes']]).hexdigest()
        # Numeric -0/+0 tie chooses first0, not bitwise total-order2.
        with self.assertRaisesRegex(ValueError,'greedy'):validate(c,p,lambda at,n:d[at:at+n])
        struct.pack_into('<I',d,0,0x3f800001)
        r['sha256']=hashlib.sha256(d[:r['bytes']]).hexdigest()
        with self.assertRaisesRegex(ValueError,'widened BF16'):validate(c,p,lambda at,n:d[at:at+n])

if __name__=='__main__':unittest.main()
