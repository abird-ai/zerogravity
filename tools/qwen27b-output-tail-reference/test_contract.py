"""Tiny source/metadata fixtures only; no Torch, model, CUDA or learned bytes."""
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from contract import HERE,STAGES,require,metadata,read,write,probe,finite,specs,source_header,role_view,digest
from run_reference import output_path
from validate_reference import records,HeldReference

class Contract(unittest.TestCase):
    def test_closed_inputs_and_source_registry(self):
        p=metadata(HERE/'inputs.json');pins=metadata(HERE/'source-pin.json');self.assertEqual(len(specs(p,pins)),31)
        for key,value in [('repetitions',1),('torch_call_shape',[3,5120]),('archive_cap',1024),('selected_shards',p['selected_shards'][:-1])]:
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises(ValueError):specs(q,pins)
        q=copy.deepcopy(p);q['roles'][1]['name']='model.language_model.layers.0.mlp.gate_proj.weight'
        with self.assertRaises(ValueError):specs(q,pins)
    def test_probe_is_finite_raw_bf16_independent_rows(self):
        self.assertEqual(len(probe(11)),10240);self.assertTrue(finite(probe(11)));self.assertNotEqual(probe(11),probe(353))
        for bad in [True,0,-1,65536]:
            with self.assertRaises(ValueError):probe(bad)
        self.assertFalse(finite(b'\x80\x7f'));self.assertFalse(finite(b'x'))
    def test_actual_header_extent_checked_before_mapping(self):
        with tempfile.TemporaryDirectory()as t:
            path=Path(t)/'source';h={'w':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}};raw=json.dumps(h).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+b'\x00\x3f'*2)
            fd=os.open(path,os.O_RDONLY)
            try:
                start,head=source_header(fd,{'bytes':path.stat().st_size});r={'name':'w','shape':[2],'dtype':'BF16'};v=role_view(start,head,r,path.stat().st_size);self.assertEqual(v['absolute'],start);self.assertEqual(v['bytes'],4)
                head['w']['data_offsets']=[0,6]
                with self.assertRaises(ValueError):role_view(start,head,r,path.stat().st_size)
            finally:os.close(fd)
    def test_held_fixture_sha_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/'bits';p.write_bytes(b'actual tiny fixture');pin={'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())}
            self.assertEqual(read(p,pin=pin),b'actual tiny fixture');pin['sha256']='0'*64
            with self.assertRaises(ValueError):read(p,pin=pin)
            (Path(t)/'link').symlink_to(p)
            with self.assertRaises(OSError):read(Path(t)/'link')
    def test_fresh_output_is_disjoint_from_source_and_package(self):
        with tempfile.TemporaryDirectory()as t:
            d=Path(t);source=d/'source';source.mkdir();p={'source_directory':str(source)}
            self.assertEqual(output_path(d/'fresh',p),d/'fresh')
            for bad in [source/'fresh',HERE/'fresh',Path('relative')]:
                with self.assertRaises(ValueError):output_path(bad,p)
    def test_entire_raw_cohort_shape_coverage_and_repeats(self):
        plan=metadata(HERE/'inputs.json');raw=bytearray();rows=[]
        for p in plan['probes']:
            for repeat in range(2):
                for label,count in STAGES.items():
                    b=probe(p['seed'])if label=='input'else b'\x00\x00'*count
                    rows.append({**p,'repeat':repeat,'layer':63,'label':label,'shape':[1,count],'stride':[count,1],'dtype':'torch.bfloat16','device':'cpu','offset':len(raw),'bytes':len(b),'sha256':digest(b)});raw.extend(b)
        good={'records':rows,'raw':{'bytes':len(raw)}};self.assertEqual(records(plan,good,raw),2609152)
        for kind in range(4):
            bad=copy.deepcopy(good)
            if kind==0:bad['records'].pop()
            if kind==1:bad['records'][2]['shape']=[3,17408]
            if kind==2:bad['records'][18]['seed']=11
            if kind==3:bad['records'][0]['offset']=1
            with self.assertRaises(ValueError):records(plan,bad,raw)
    def test_reference_pair_changed_during_source_scan_cannot_be_sealed(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);raw=root/'raw.bin';manifest=root/'manifest.json'
            raw.write_bytes(b'original')
            manifest.write_text(json.dumps({'raw':{'bytes':8,'sha256':digest(b'original')}}))
            with HeldReference(raw,128)as held_raw,HeldReference(manifest,1024)as held_manifest:
                self.assertEqual(held_raw.data,b'original')
                self.assertEqual(held_manifest.metadata()['raw']['sha256'],digest(b'original'))
                # Simulate a mutually consistent replacement while the real
                # validator is busy scanning the original source shards.
                raw.write_bytes(b'changed!')
                manifest.write_text(json.dumps({'raw':{'bytes':8,'sha256':digest(b'changed!')}}))
                for held_file in [held_raw,held_manifest]:
                    with self.assertRaises(ValueError):held_file.pin()
    def test_reference_path_replacement_and_tail_refuse_same_old_fd(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);path=root/'raw.bin';path.write_bytes(b'original')
            with HeldReference(path,128)as held:
                self.assertEqual(held.pin()['sha256'],digest(b'original'))
                replacement=root/'other';replacement.write_bytes(b'original');replacement.replace(path)
                with self.assertRaises(ValueError):held.pin()
            with HeldReference(path,128)as held:
                with path.open('ab')as stream:stream.write(b'tail')
                with self.assertRaises(ValueError):held.pin()
    def test_package_imports_never_import_torch_or_old_support(self):
        import ast
        for name in ['contract.py','run_reference.py','validate_reference.py','tensor_bits.py']:
            tree=ast.parse((HERE/name).read_text());imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
            names=[n.module if isinstance(n,ast.ImportFrom)else a.name for n in imports for a in (n.names if isinstance(n,ast.Import)else [None])]
            self.assertNotIn('support',names);self.assertNotIn('torch',names);self.assertFalse(any(x and x.startswith('transformers')for x in names))
if __name__=='__main__':unittest.main()
