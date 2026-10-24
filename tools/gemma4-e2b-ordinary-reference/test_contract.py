"""Bounded protocol fixtures only; no Torch/model/tokenizer load."""
import copy,struct,unittest
from pathlib import Path
from support import metadata
from contract import bf16,validate
HERE=Path(__file__).resolve().parent
class Contract(unittest.TestCase):
    def test_lossless_full_matrix_fits_existing_per_file_caps(self):
        request=metadata(HERE/'request.json')
        self.assertEqual([r['rows']for r in request['cases']],[1,2,3,4,32,57,3])
        for case in request['cases']:
            rows=case['rows'];n=8
            raw=n*262144*2+18432*(n*rows+n*(n-1)//2)
            boundaries=(rows+n-1)*35*2*1536*2
            self.assertLessEqual(raw,16*1024*1024);self.assertLessEqual(boundaries,32*1024*1024)
        self.assertEqual(8*262144*2+18432*(8*57+28),13115392)
        self.assertEqual(64*35*2*1536*2,13762560)
    def test_bf16_storage_is_exact_and_nonfinite_is_never_sanitized(self):
        bf16(struct.pack('<4H',0,0x8000,0x3f80,0xbf80))
        for word in [0x7f80,0x7fc0,0xff80]:
            with self.assertRaises(ValueError):bf16(struct.pack('<H',word))
        with self.assertRaises(ValueError):bf16(b'x')
    def fixture(self):
        request=metadata(HERE/'request.json');pins=metadata(HERE/'source-pin.json')
        c=dict(schema='zg.gemma4-e2b.ordinary-matrix.v1',source=pins,request=request,
            profile='HfTorchCpuBf16Avx2Eager5.17.0',transformers='5.17.0',torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb',eos_ids=[1,106,50],plain_vs_observed_equal=True,trained_rust_qualified=False,native_serving_chat_or_performance_qualified=False,
            cells=[dict(name=r['name'],metadata=dict(name=r['name']+'.json'))for r in request['cases']])
        return c,pins,request
    def test_scope_source_and_coverage_fail_before_any_raw_payload(self):
        c,pins,request=self.fixture()
        def forbidden(*_args):self.fail('wrong source/scope must refuse before payload access')
        for key,value in [('source',{}),('trained_rust_qualified',True),('native_serving_chat_or_performance_qualified',True),('plain_vs_observed_equal',False),('cells',[])]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate(bad,pins,request,forbidden,forbidden)
    def test_cell_name_cannot_escape_or_relabel(self):
        c,pins,request=self.fixture();c['cells'][0]['metadata']['name']='../foreign.json'
        def forbidden(*_args):self.fail('foreign path must refuse before open')
        with self.assertRaises(ValueError):validate(c,pins,request,forbidden,forbidden)
if __name__=='__main__':unittest.main()
