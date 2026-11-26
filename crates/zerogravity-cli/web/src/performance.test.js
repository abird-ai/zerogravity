import {test} from "node:test";
import assert from "node:assert/strict";
import {history,tps,latency} from "./performance.js";
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
