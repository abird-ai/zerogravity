"""Tiny prefix fixtures: no framework import, model or learned source scan."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contract import (KINDS, STAGES, extent, full_scope, metadata, prefix_bytes,
                      record_metadata, record_specs, roles, scope, specs, state_specs)
from validate_reference import Held, validate_calls, validate_inner_calls
from run_reference import source_after


def calls():
    result=[]
    for repeat in range(2):
        row=dict(repeat=repeat,call=0,position=0,teacher=11,cache_frontier=1,
                 original_module_order=['embedding']+['layer.'+str(i) for i in range(10)],
                 stopped_after_source_unload=True,full_forward_returned=False)
        for moment in ('before','after'):
            row[moment]=[]
            for i,kind in enumerate(KINDS):
                flag=dict(layer=i,kind=kind)
                for key in (('key','value') if kind=='full_attention' else ('conv','recurrent','previous')):
                    flag[key]=moment=='after' and i<=9
                row[moment].append(flag)
        result.append(row)
    return result


class Protocol(unittest.TestCase):
    def test_source_contract_is_full_original_but_prefix_claim_is_narrow(self):
        here=Path(__file__).resolve().parent
        plan=metadata(here/'inputs.json');pins=metadata(here/'source-pin.json')
        self.assertEqual(len(specs(plan,pins)),31)
        self.assertEqual(plan['scope'],full_scope())
        used=[r for r in roles() if r['group']=='embedding' or r['layer'] is not None and r['layer']<=9]
        self.assertEqual(len(used),135)
        self.assertEqual(sum(2 if k=='linear_attention' else 1 for k in KINDS[:10]),18)
        self.assertFalse(scope()['original_full_text_forward'])
        bad=copy.deepcopy(plan);bad['revision']='other'
        with self.assertRaises(ValueError):specs(bad,pins)

    def test_exact_original_producer_order_extent_and_no_final_claim(self):
        s=list(record_specs());self.assertEqual(len(s),116)
        self.assertEqual(sum(extent(r[5],r[6]) for r in s),prefix_bytes())
        self.assertEqual([r[4] for r in s if r[0]==0 and r[4].startswith('layer9.')],[r[0] for r in STAGES])
        self.assertEqual(len(state_specs(0,'after')),20)
        self.assertFalse(any(r[4] in ('normalized','logits') or r[1]!=0 for r in s))
        with self.assertRaises(ValueError):state_specs(1,'after')

    def test_real_initialized_cut_refuses_unexecuted_or_missing_layer(self):
        good=calls();validate_calls(good)
        for layer,value in [(10,True),(9,False)]:
            bad=copy.deepcopy(good);bad[0]['after'][layer]['conv']=value
            with self.assertRaises(ValueError):validate_calls(bad)
        for key,value in [('full_forward_returned',True),('stopped_after_source_unload',False),('cache_frontier',2)]:
            bad=copy.deepcopy(good);bad[0][key]=value
            with self.assertRaises(ValueError):validate_calls(bad)

    def test_actual_stride_and_codec_are_checked(self):
        s=next(r for r in record_specs() if r[4]=='layer9.core')
        row=dict(repeat=0,call=0,teacher=11,position=0,layer=9,label=s[4],shape=[48,128],dtype='BF16',device='cpu',offset=0,bytes=12288,original_stride=[128,1])
        self.assertEqual(record_metadata(row,s,0),12288)
        for key,value in [('shape',[1,48,128]),('original_stride',[129,1]),('dtype','F32'),('layer',8),('offset',1)]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(ValueError):record_metadata(bad,s,0)

    def test_held_metadata_refuses_replacement_after_original_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(b'{"original":true}')
            with Held(p,1024) as held:
                self.assertEqual(held.json(),{'original':True})
                q=Path(d)/'b';q.write_bytes(b'{"changed":true}');q.replace(p)
                with self.assertRaises(ValueError):held.pin()

    def test_source_after_refuses_same_size_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(b'old');fd=os.open(p,os.O_RDONLY)
            try:
                from contract import fingerprint,digest
                rows=[{'name':'a','bytes':3,'sha256':digest(b'old'),'fd_identity':fingerprint(os.fstat(fd))}]
                source_after({'a':fd},rows,d)
                p.write_bytes(b'new')
                with self.assertRaises(ValueError):source_after({'a':fd},rows,d)
            finally:os.close(fd)

class InnerDispatch(unittest.TestCase):
    def test_actual_synthetic_padded64_source_shapes_and_strides_are_closed(self):
        calls=metadata(Path(__file__).resolve().parent/'tests/synthetic-padded64-calls.json')
        validate_inner_calls(calls)
        for key,value in [('shape',[1,48,1,1,128]),('stride',[393216,8192,8192,1,128]),('dtype','torch.bfloat16')]:
            bad=copy.deepcopy(calls);bad[0]['left'][key]=value
            with self.assertRaises(ValueError):validate_inner_calls(bad)
        bad=copy.deepcopy(calls);bad[1]['source_matmul']=0
        with self.assertRaises(ValueError):validate_inner_calls(bad)

if __name__=='__main__':unittest.main()
