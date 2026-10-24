"""Hostile finite schema fixtures; synthetic zeros are not learned evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from producer_contract import expected_producers,validate_producers


def fixture(call_count=1):
    at=0;records=[]
    for (position,layer,label),elements in expected_producers(call_count).items():
        n=elements*2
        records.append({'position':position,'layer':layer,'label':label,'elements':elements,
                        'bytes':n,'offset':at,'sha256':hashlib.sha256(bytes(n)).hexdigest()})
        at+=n
    raw=bytes(at);pins={'fixture':'no learned payload'}
    capture={'schema':'zg.gemma4-e2b.producers.v1','source':pins,'prompt_rows':4,'positions':[0,3],
             'records':records,'raw':{'name':'producers.bin','bytes':at,'sha256':hashlib.sha256(raw).hexdigest()},
             'original_results_unchanged':True,'trained_rust_qualified':False}
    if call_count!=1:
        capture.update(schema='zg.gemma4-e2b.producers.v2',call_count=call_count,
                       positions=[0,3]+list(range(4,3+call_count)),
                       calls=[{'call':call,'base':0 if call==0 else call+3,
                            'input_rows':4 if call==0 else 1,'positions':[0,3] if call==0 else [call+3]}
                            for call in range(call_count)])
    return capture,pins,raw


class Producers(unittest.TestCase):
    def test_complete_all35_roles_and_source_dtype_geometry(self):
        c,p,r=fixture();validate_producers(c,p,lambda at,n:r[at:at+n])
        self.assertLessEqual(len(r),32*1024*1024)
        self.assertEqual({row['layer'] for row in c['records'] if row['layer'] is not None},set(range(35)))
        self.assertEqual(expected_producers()[(0,4,'value.projection')],512)
        self.assertNotIn((0,15,'key.projection'),expected_producers())
        self.assertEqual(expected_producers()[(3,34,'attention.qk')],32)

    def test_missing_duplicate_wrong_issuer_and_fabricated_shared_kv_refuse(self):
        c,p,r=fixture()
        for mutate in [lambda x:x['records'].pop(),lambda x:x['records'].__setitem__(1,x['records'][0]),
                       lambda x:x.__setitem__('source',{}),lambda x:x['records'][0].__setitem__('layer',15),
                       lambda x:x['records'][0].__setitem__('position',False)]:
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises((AssertionError,RuntimeError,ValueError)):validate_producers(bad,p,lambda at,n:r[at:at+n])

    def test_truncation_nonfinite_words_and_extra_bytes_refuse(self):
        c,p,r=fixture()
        with self.assertRaises((AssertionError,RuntimeError,ValueError)):
            validate_producers(c,p,lambda at,n:r[at:at+n-1])
        bad=copy.deepcopy(c);word=b'\x80\x7f'+r[2:]
        bad['records'][0]['sha256']=hashlib.sha256(word[:bad['records'][0]['bytes']]).hexdigest()
        bad['raw']['sha256']=hashlib.sha256(word).hexdigest()
        with self.assertRaises((AssertionError,RuntimeError,ValueError)):validate_producers(bad,p,lambda at,n:word[at:at+n])
        bad=copy.deepcopy(c);bad['raw']['bytes']+=2
        with self.assertRaises((AssertionError,RuntimeError,ValueError)):validate_producers(bad,p,lambda at,n:r[at:at+n])

    def test_profile_and_observation_scope_cannot_claim_trained_or_cached_closure(self):
        c,p,r=fixture()
        for key,value in [('trained_rust_qualified',True),('positions',[0,4]),('prompt_rows',5),
                          ('original_results_unchanged',False)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises((AssertionError,RuntimeError,ValueError)):validate_producers(bad,p,lambda at,n:r[at:at+n])

    def test_all_actual_three_calls_have_full_roles_and_unchanged_archive_fit(self):
        c,p,r=fixture(3);validate_producers(c,p,lambda at,n:r[at:at+n])
        self.assertEqual(len(c['records']),3965)
        self.assertEqual(len(r),22598144)
        self.assertLess(len(c['records']),4096)
        self.assertEqual(expected_producers(3)[(5,34,'attention.qk')],48)
        self.assertEqual(expected_producers(3)[(4,None,'head')],262144)
        from producers import ProducerTrace
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'producers.json';ProducerTrace.write_metadata(path,c)
            self.assertLessEqual(path.stat().st_size,1024*1024)
            self.assertEqual(json.loads(path.read_text()),c)

    def test_fabricated_cached_context_or_missing_late_boundary_refuses(self):
        c,p,r=fixture(3)
        changes=[lambda x:x['calls'][1].update(base=3),
                 lambda x:x['calls'][1].update(input_rows=4),
                 lambda x:x['calls'][1].update(input_rows=True),
                 lambda x:x.update(call_count=True),lambda x:x.update(call_count=4),
                 lambda x:x['positions'].__setitem__(0,False),
                 lambda x:x['records'][-1].update(position=3),lambda x:x['records'].pop()]
        for mutate in changes:
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises((AssertionError,RuntimeError,ValueError)):
                validate_producers(bad,p,lambda at,n:r[at:at+n])

    def test_cached_token_axis_maps_to_actual_global_position(self):
        from producers import ProducerTrace
        trace=ProducerTrace.__new__(ProducerTrace)
        trace.active=False;trace.layer=None;trace.calls=[{},{}]
        trace.begin_call(2,1,5)
        class Tensor:
            ndim=3;shape=(1,1,2)
            def __getitem__(self,key):return key
        records=[]
        trace.record=lambda *args:records.append(args)
        trace.tokens(0,'query.norm',Tensor())
        self.assertEqual(records,[(5,0,'query.norm',(0,0))])
        with self.assertRaises(ValueError):trace.begin_call(2,1,5)
        trace.active=False
        with self.assertRaises(ValueError):trace.begin_call(2,1,4)

if __name__=='__main__':unittest.main()
