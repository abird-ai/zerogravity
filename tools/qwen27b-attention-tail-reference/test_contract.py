"""Small source-free fixtures; importing this suite never imports Torch."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contract import HERE,archive_words,digest,probe,stage_counts,specs
from validate_reference import HeldReference,producer_joins,records

def fixture():
    plan=json.loads((HERE/'inputs.json').read_text());rows=[];raw=bytearray()
    for p in plan['probes']:
        for repeat in range(2):
            for label,n in stage_counts(p['row']).items():
                b=probe(p['seed'])if label=='input'else bytes(n*2)
                rows.append({**p,'repeat':repeat,'position':p['row'],'layer':63,'label':label,'shape':[1,n],'stride':[n,1],
                    'dtype':'torch.bfloat16','device':'cpu','offset':len(raw),'bytes':len(b),'sha256':digest(b)})
                raw.extend(b)
    return plan,dict(records=rows,raw=dict(bytes=len(raw))),bytes(raw)

class Protocol(unittest.TestCase):
    def test_complete_28_stage_causal_cohort_full_extent_and_cache_rows(self):
        plan,r,raw=fixture();self.assertEqual(records(plan,r,raw),3_187_696)
        self.assertEqual(archive_words(plan),3_187_696)
        self.assertEqual(stage_counts(2)['keys'],3072);self.assertEqual(stage_counts(2)['probabilities'],72)
        for bad in (-1,64,False):
            with self.assertRaises(ValueError):stage_counts(bad)
    def test_source_roles_and_no_fullmodel_authority_are_exact(self):
        plan=json.loads((HERE/'inputs.json').read_text());pins=json.loads((HERE/'source-pin.json').read_text())
        self.assertEqual(len(specs(plan,pins)),31)
        for change in ('shape','scope'):
            bad=copy.deepcopy(plan)
            if change=='shape':bad['roles'][1]['shape'][0]-=1
            else:bad['scope']['whole_model_qualified']=True
            with self.assertRaises(ValueError):specs(bad,pins)
    def test_truncation_reordered_or_duplicate_stage_never_seals(self):
        plan,r,raw=fixture()
        for bad in (r['records'][:-1],list(reversed(r['records'])),r['records'][:1]+r['records'][:-1]):
            with self.assertRaises(ValueError):records(plan,dict(records=bad,raw=r['raw']),raw)
        with self.assertRaises(ValueError):records(plan,r,raw[:-2])
    def test_prior_history_and_interleaved_query_gate_have_true_producer_joins(self):
        plan,report,raw=fixture();blocks={(r['cohort'],r['row'],r['repeat'],r['label']):raw[r['offset']:r['offset']+r['bytes']]for r in report['records']}
        producer_joins(plan,blocks)
        bad=dict(blocks);b=bytearray(bad[(1,1,0,'values')]);b[0]=1;bad[(1,1,0,'values')]=bytes(b)
        with self.assertRaises(ValueError):producer_joins(plan,bad)
        bad=dict(blocks);bad[(0,0,0,'gate')]=b'\0\x80'+bad[(0,0,0,'gate')][2:]
        with self.assertRaises(ValueError):producer_joins(plan,bad)
    def test_held_reference_path_replacement_or_mutation_refuses(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'meta';p.write_bytes(b'{}')
            with HeldReference(p,16)as h:
                q=Path(tmp)/'new';q.write_bytes(b'{}');q.replace(p)
                with self.assertRaises(ValueError):h.pin()
            with HeldReference(p,16)as h:
                p.write_bytes(b'[]')
                with self.assertRaises(ValueError):h.pin()
    def test_capture_import_cannot_load_model_or_framework(self):
        import capture
        self.assertIsNone(capture.OWNER)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

if __name__=='__main__':unittest.main()
