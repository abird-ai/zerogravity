import {test} from "node:test";
import assert from "node:assert/strict";
import {history,tps,latency,layerStrip,providerLabel} from "./performance.js";
test("model performance samples use actual elapsed owner time, idle zero and reset on replacement",()=>{
  const sample=history();
  const model=(id,ms,tokens)=>({registration_id:id,performance:{loaded_ms:ms,output_tokens:tokens}});
  assert.deepEqual(sample(model("1",1000,10)),[]);
  assert.deepEqual(sample(model("1",1200,12)),[]);
  assert.deepEqual(sample(model("1",2000,50)),[40]);
  assert.deepEqual(sample(model("1",3000,50)),[40,0]);
  assert.deepEqual(sample(model("1",10000,50)),[]);
  assert.deepEqual(sample(model("1",11000,70)),[20]);
  assert.deepEqual(sample(model("2",0,0)),[]);
  assert.equal(tps(null),"—");
  assert.equal(tps(0),"0.0");
  assert.equal(latency(1234),"1.23 s");
});

test("layer strip uses actual ranges, stable physical identities and explicit mixed bins",()=>{
  const cpu={kind:"host_cpu"};
  const gpu={kind:"device",device:{backend:"pci",device:"0000:01:00.0"}};
  const map={total_layers:6,spans:[{start:0,end:2,execution:cpu},{start:2,end:5,execution:gpu},{start:5,end:6,execution:cpu}]};
  assert.deepEqual(layerStrip(map).cells.map(cell=>cell.owner),[0,0,1,1,1,0]);
  assert.deepEqual(layerStrip(map).devices.map(device=>device.label),["Host CPU","pci · 0000:01:00.0"]);
  assert.deepEqual(layerStrip(map,2).cells,[{start:0,end:3,owner:null},{start:3,end:6,owner:null}]);
  assert.equal(layerStrip({...map,spans:map.spans.slice(1)}),null);
  assert.equal(providerLabel("Hip"),"ROCm");
  assert.equal(providerLabel(null),"Not reported");
});
