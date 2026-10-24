"""Tiny whole-contract fixtures; no tokenizer/framework/source payload load."""
import copy,unittest
from pathlib import Path
from support import metadata
from contract import validate
HERE=Path(__file__).resolve().parent
class Contract(unittest.TestCase):
    def fixture(self):
        cases=metadata(HERE/'cases.json');pins=metadata(HERE/'source-pin.json')
        c=dict(schema='zg.gemma4-e2b.original-tokenizer.v2',source=pins,transformers_version='5.17.0',
            source_class={'class':'GemmaTokenizer'},request=cases,repetitions=2,repetition_equal=True,
            model_constructed=False,weights_verified=False,rust_tokenizer_qualified=False,
            trained_rust_qualified=False,native_serving_or_performance_qualified=False,
            rows=[dict(name=r['name'],text=r['text'],ids=[],without_special=[],decode='',decode_skip='')for r in cases['cases']],
            decodes=[dict(name=r['name'],tokens=r['tokens'],ids=[0]*len(r['tokens']),decode='',decode_skip='')for r in cases['decode_tokens']])
        return c,pins,cases
    def test_exact_complete_declared_cases_and_no_weight_claim(self):
        c,p,cases=self.fixture();validate(c,p,cases)
        self.assertEqual(len(c['rows']),46);self.assertEqual(len(c['decodes']),15);self.assertEqual(len(p['files']),7)
        self.assertFalse(any(r['name']=='model.safetensors'for r in p['files']))
    def test_missing_reordered_or_duplicate_case_refuses(self):
        c,p,cases=self.fixture()
        for mutate in [lambda x:x['rows'].pop(),lambda x:x['rows'].reverse(),lambda x:x['rows'].__setitem__(1,x['rows'][0])]:
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises(ValueError):validate(bad,p,cases)
    def test_integer_domain_and_identity_postprocessor_are_strict(self):
        c,p,cases=self.fixture()
        for value in [[True],[262144],list(range(65))]:
            bad=copy.deepcopy(c);bad['rows'][0]['ids']=value;bad['rows'][0]['without_special']=value
            with self.assertRaises(ValueError):validate(bad,p,cases)
        bad=copy.deepcopy(c);bad['rows'][0]['without_special']=[2]
        with self.assertRaises(ValueError):validate(bad,p,cases)
    def test_original_scope_never_becomes_model_math_or_native_authority(self):
        c,p,cases=self.fixture()
        for key in ['model_constructed','weights_verified','rust_tokenizer_qualified','trained_rust_qualified','native_serving_or_performance_qualified']:
            bad=copy.deepcopy(c);bad[key]=True
            with self.assertRaises(ValueError):validate(bad,p,cases)
    def test_incomplete_decoder_cohort_and_expected_id_injection_refuse(self):
        c,p,cases=self.fixture()
        for mutate in [lambda x:x['decodes'].pop(),lambda x:x['decodes'].reverse(),
            lambda x:x['decodes'][0].__setitem__('ids',[True]),lambda x:x['decodes'][0].__setitem__('tokens',['<fake>'])]:
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises(ValueError):validate(bad,p,cases)
    def test_source_mutation_during_final_hash_refuses(self):
        import os,tempfile,hashlib
        import run_reference
        from support import fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source';path.write_bytes(b'trusted')
            fd=os.open(path,os.O_RDONLY);before=fingerprint(os.fstat(fd));original=run_reference.fd_sha
            row=dict(name='source',bytes=7,sha256=hashlib.sha256(b'trusted').hexdigest(),fd_identity=before)
            def mutation_after_read(*args):
                path.write_bytes(b'changed')
                return row['sha256']
            try:
                run_reference.fd_sha=mutation_after_read
                with self.assertRaises(ValueError):run_reference.source_after(Path(tmp),{'source':fd},[row])
            finally:
                run_reference.fd_sha=original;os.close(fd)
if __name__=='__main__':unittest.main()
