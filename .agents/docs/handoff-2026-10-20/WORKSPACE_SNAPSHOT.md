# Read-only workspace snapshot — 20 October 2026

Dated observation, not authorization or runtime qualification. No remote fetch.

## HEAD and signature

```text
fb68d8fa165de16e21d1bd37afef3790b80da971 G perf: combine GPU state scatter certification
```

## Local tracking divergence

```text
34	0
```

## MAIN pending changes

```text
 M .agents/plans/model-execution/TODO.md
 M .agents/plans/numerics-and-context/TODO.md
 M .agents/plans/serving-and-speculation/TODO.md
 M crates/zerogravity-gpu/src/lib.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/context.rs
?? .agents/docs/handoff-2026-10-20/
?? .agents/plans/NEXT_AGENT_PROMPT_2026-10-20.md
?? .agents/plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md
?? crates/zerogravity-gpu/src/ordered_silu.rs
?? crates/zerogravity-gpu/src/ordered_silu/
?? crates/zerogravity-gpu/src/resident_program/silu.rs
?? crates/zerogravity-gpu/src/resident_program/silu/
```

## Staged paths (empty at snapshot)

```text

```

## Root README blob

```text
a7be69c35cf61d14e8810f9cb5a6a43d1397bf27
```

## Commits since previous handoff

```text
35ebc5b343fd82201eac768edf86ba7a81c41846 2026-10-19T18:08:46+08:00 G feat(gpu): fuse native resident model batches
904a190cf39aca74e06c979dbaddb0db7f4404f1 2026-10-19T18:09:43+08:00 G docs(agents): record native batch acceptance
54fd580815dbcfcf0916e4e737beaf21728818b2 2026-10-19T19:51:39+08:00 G feat(gpu): keep batch greedy feedback on device
570d865696ab398f0c5c37f08c5c951595eadd75 2026-10-19T19:52:16+08:00 G docs(agents): record batch feedback acceptance
2677d8b664c78b657c26e7bf46c835317b003913 2026-10-19T22:33:08+08:00 G feat(serve): run native CUDA B2 waves
7fc9ad475a30a3d2120f307a7da2ff75a80e337a 2026-10-19T22:33:38+08:00 G docs(serve): bound native B2 acceptance
f26232e8b03bf5ab3c13e00ed641eb0d602360d2 2026-10-20T00:28:17+08:00 G fix(qwen): honor tied original output heads
6bfcdbe72f0133a6c79e2e7de169c3de74b3da8a 2026-10-20T00:30:04+08:00 G feat(bench): account for offered serving work
d9fa829ceba092c7f8a3fa9c2746a11a0906eaba 2026-10-20T00:40:08+08:00 G feat(serve): qualify native CUDA rows 4 and 8
9815504a6975ce766766fc8089729be8121df3ac 2026-10-20T01:13:36+08:00 G feat(qwen): bind original HF diagnostic profiles
dbb5f4f4cc42e28c91b12e906f9091ef76510c03 2026-10-20T01:53:12+08:00 G feat(serve): fuse unequal decode frontiers
cadcbd9d1216b00ca4b69cae9bc04a409e347193 2026-10-20T02:07:20+08:00 G feat(qwen): qualify pinned CPU HF operators
1b599df06d8d7c8c4912838836b272e28e04aca1 2026-10-20T02:08:49+08:00 G feat(bench): isolate owned provider lifecycles
d41163432fd6233c7bd88f07b64fb71b1086a902 2026-10-20T02:11:18+08:00 G feat(artifacts): own verified tensor windows
2a32aff615fc4f7f6718f14a15187825730d6897 2026-10-20T02:43:37+08:00 G feat(qwen): compose typed source operators
c5eb1f8e717c79146937e585a145a0052eff747b 2026-10-20T03:22:04+08:00 G feat(cpu): qualify source sigmoid and RMS
d0a60844217fc106fe8c70bb3a99073eef0387c8 2026-10-20T04:19:25+08:00 G perf(gpu): fuse source activation preparation
5350a4d40b535106240d48fec8aedd459d5cd17a 2026-10-20T04:56:45+08:00 G feat(cpu): qualify typed source windows
f78892957f09a84838246aede4fef2be040a0d18 2026-10-20T06:13:20+08:00 G feat(gpu): fuse native teacher-prefix serving
bfe62df68d763b9d5c6b87c891be6a62b3593de9 2026-10-20T06:35:48+08:00 G feat(cpu): execute leased source-window plans
fe2a34213009cc160f4d82068bc7dd1f49d05c6c 2026-10-20T08:12:19+08:00 G feat(cpu): qualify original attention fragments
9a199017f18f1bcb1a8053685be3c3c29236ec90 2026-10-20T08:21:04+08:00 G feat(gpu): qualify batched graph replay
fff1842594f8011b761d08fcc1690311b72543e3 2026-10-20T09:20:27+08:00 G feat: qualify trained BF16 operator slices
fb68d8fa165de16e21d1bd37afef3790b80da971 2026-10-20T13:27:39+08:00 G perf: combine GPU state scatter certification
```

## Complete reachable current HEAD history (text is not missing original ancestry)

```text
be95424837deb7919adf34503db71d27d3ad3adb 2026-10-13T12:00:00+08:00 docs(design): product, architecture, prior-art and research plans
1e10b5cc1d5cf34832477bc83ef39117a765c6a0 2026-10-13T12:00:00+08:00 feat(core): immutable source identity, state roots and physical resource ownership
af9a5d3969c9a87951990223b6d289aba69c63fb 2026-10-14T12:00:00+08:00 feat(artifacts): sharded source artifacts and model identity
b276fd01e421f1b7e416ae773610b5823b9da36a 2026-10-14T12:00:00+08:00 test(artifacts): cache transactions
586d440b144727bdd9acea7455afb27f6c38ae25 2026-10-15T12:00:00+08:00 test(artifacts): qwen text contract
782e6fc48f0924cc055232d56338da62b68b1a79 2026-10-15T12:00:00+08:00 feat(cpu): portable CPU math and weights
1fd12da7840151ba47ab43368b4e4f435aac3d3b 2026-10-16T12:00:00+08:00 feat(model-qwen35): source-verified Qwen3.5 forward and tokenizer
901ba9ae2f2ed531d992a6d657a37dc362518bd0 2026-10-16T12:00:00+08:00 test(model-qwen35): source layout contract
e6ecf02bb7d951e01e66b430a24553c690b4ac5b 2026-10-17T12:00:00+08:00 feat(model-gemma4): Gemma contract, MoE and 26B variant
81642a9af4d88d928327a1012af1594168704956 2026-10-17T12:00:00+08:00 test(model-gemma4): e2b contract
7aa19c268026069ad916f6a69440dab602aaa94a 2026-10-18T12:00:00+08:00 test(model-gemma4): family contract
65b026f902501e85afad5fd31f3ca53968a3e836 2026-10-18T12:00:00+08:00 test(model-gemma4): variant 26b contract
b03c16451d65c33a05b3c495c1205240db4ccac0 2026-10-19T12:00:00+08:00 feat(data): vendored CubeCL license and provenance
f491e34fe55e67eed434ac577c32d44eca51430e 2026-10-19T12:00:00+08:00 feat(data): vendor CubeCL cubecl-ir 0.11.0
627c58ec80ff7b33fa1ce5086a729946c5b1cd44 2026-10-20T12:00:00+08:00 feat(data): vendor CubeCL cubecl-cpp 0.11.0
3664f975e79242d8b1a12bc4eefe886f81a33c2f 2026-10-20T12:00:00+08:00 feat(data): vendor CubeCL cubecl-cuda 0.11.0
f77ae37ec0317f39a2e1ce4466d82c645cd344da 2026-10-21T12:00:00+08:00 feat(data): vendor CubeCL cubecl-hip 0.11.0
7d768a080a84b6132595f7cadbc84b75b69106e4 2026-10-21T12:00:00+08:00 feat(gpu): source-exact resident GPU execution and physical ownership
a9723bceb8ad4aa4df639dbe660efcae6c22711b 2026-10-22T12:00:00+08:00 feat(runtime): facade, semantic program, task drivers and state publication
c3d0d0b06fb1548c8bf3db10f2965e5af52be9bc 2026-10-22T12:00:00+08:00 test(runtime): gemma contract
4add50a023d4fb5dd989989cab75ac4298d46d29 2026-10-23T12:00:00+08:00 test(runtime): layout api
b4a36b829031fba498e45bebb71a6cd8f01e7ef9 2026-10-23T12:00:00+08:00 test(runtime): multimodal mm0
a691e21c040d2dcf46a52f2cbcf0e0eb934c8369 2026-10-24T12:00:00+08:00 test(runtime): qwen real model
25ec3e27e3c208158e9763944026003005456461 2026-10-24T12:00:00+08:00 test(runtime): qwen shared model
047e3b70bbcbd7acbb5176712208cc4bcc02dd29 2026-10-25T12:00:00+08:00 feat(cli): single-binary inspect and run
52e37313bff57c2a7e831f50e9236fa983e13d02 2026-10-25T12:00:00+08:00 test(cli): CLI contract
22e0cc5abb4a067a95324206c2881de9e3327427 2026-10-26T12:00:00+08:00 feat(tooling): workspace GPU qualification crate
4d82c6704dad47875fa42bdec65f9a65de79722e 2026-10-26T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_layers
a2d6a02f1577554ea8dba669f77ba7b903c1bb29 2026-10-27T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_logits
a8ea8926b53d7b4a9c6d00f3d9be04906f8d64bb 2026-10-27T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_qk
6a0079eb167d1f1593fe2a9372039138cd949770 2026-10-28T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_recurrent
be88e30733f9459eda5f150bdbd70d9d0f8c6684 2026-10-28T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_rope
77778a2c258751c64c56c4ffd3dba2ff70f7296c 2026-10-29T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_softmax
6015241a5f5c2bb7ee65715a998988a026ff800c 2026-10-29T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_value
e190aff1fdd392696e8086010a7bbc1f12d3df4f 2026-10-30T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_value_padding
0aea8de66e3eb3b704e7d26687731a3e5b07d3bb 2026-10-30T12:00:00+08:00 test(tooling): build-release-artifacts.sh
057987203d5de45ad268ba02cfc5084f76f48229 2026-10-31T12:00:00+08:00 test(tooling): check.sh
c74450a27b1e3080e9576b0f95922f17c7cdca8b 2026-10-31T12:00:00+08:00 test(tooling): compare-attention-value.py
9f54d2435fe6af15b3eaf28980bc1039ad02b559 2026-11-01T12:00:00+08:00 test(tooling): compare-qwen-layers.py
7458ffc286f7e72c70ac18acfd3f776186aa204a 2026-11-01T12:00:00+08:00 test(tooling): cpu-qualification-gate.py
73cb6e55e1795e2c13533f60621dc29619cf5e2c 2026-11-02T12:00:00+08:00 test(tooling): native-gpu.py
80145a7f91724fb34a6f67e9972850a279982bf3 2026-11-02T12:00:00+08:00 test(tooling): test-install.sh
ca1138b8cf866c81bbfb276358d03a62576df16a 2026-11-03T12:00:00+08:00 test(tooling): test-native-gpu.py
5a0636c3f9c744c2f979ded40943a269ba5a1176 2026-11-03T12:00:00+08:00 test(tooling): test-numeric-resume.py
9a6c504ebef037e09bd187d65e2bec7e2e456eef 2026-11-05T12:00:00+08:00 chore(workspace): restore the locked full workspace manifest
57a9ad7fe78bf7c5e64d3870a6ba176ceba4cc07 2026-11-05T12:00:00+08:00 test(tooling): independent reference probe notes
1af2ac434279cb5f4b635a51884c936f19b54aa3 2026-10-16T21:39:45+08:00 fix(nix): give module check a complete base system
0452c594d99df6d3b4c972ceaae8df6ea56925e3 2026-10-16T23:45:32+08:00 docs(plan): record module eval check fix
2afa32aa74aff7f35ec1b52e4caa7079a5f68980 2026-10-16T23:45:32+08:00 docs(readme): reorder engineering and FAQ sections
a054d396e1ec12fc5e9b8e2ca60e8060954870c5 2026-10-17T08:33:07+08:00 docs(agents): document plan layout and TODO spec
cc3cab27a2713fdb64bb3f1a42c8138c060c5dce 2026-10-17T08:33:07+08:00 docs(contributing): align guide and AI disclosure
7e61cb63efd0c1429aa96044f2f7d28fda1e59aa 2026-10-17T08:41:24+08:00 docs(agents): gate pushes and destructive actions
9239719c25128028fd78a6a60f817ce6b626517f 2026-10-17T09:54:54+08:00 docs: retire historical archive bodies
6deebf22228315863f563723b5e6761ac6b51c5c 2026-10-17T10:11:15+08:00 refactor(cache): namespace model cache under abird/zerogravity
fa6affa425089fdf9a15bfa10b330e03f738fec8 2026-10-17T10:30:09+08:00 feat(artifacts): add pinned model pulls
3c97c52075563f65928c564818ce73502168778f 2026-10-17T13:44:29+08:00 feat(artifacts): resume interrupted pulls
baa9d8818061958a63eabeb61a7239a103a738f0 2026-10-17T14:12:22+08:00 feat(artifacts): load pinned cache components
20f449d20320b6f2912698c7b97cfdfd82472ba2 2026-10-17T14:37:44+08:00 feat(artifacts): recover missing pull sizes
f44264c017856a015584a094566a077638939e55 2026-10-17T15:03:41+08:00 feat(cache): add protected snapshot removal
ea700f67fca7ff7871613044a989e03b8ad6f9d7 2026-10-17T15:23:55+08:00 feat(pull)!: add capped auxiliary streaming
03b69482a5fb10b8c1cc7032e7cbdf39e2e5fead 2026-10-17T15:41:45+08:00 feat(pull): pin multi-asset release components
e44f4ee8020d5661dd8167cb07dfcb085906fcb7 2026-10-17T16:16:36+08:00 feat(cache): import local HF snapshots
fce1353e649df2eb73fe6a39437043fc8551aef5 2026-10-17T16:33:28+08:00 feat(cache): qualify cross-filesystem imports
e0073a5593e5641494c0f48e8d2dcc9e9e61a119 2026-10-17T17:05:07+08:00 feat(pull): add pinned HTTPS objects
e82d76b8f8e91f7d4b49c211e26baa081ff523e5 2026-10-17T17:14:19+08:00 feat(pull): add pinned OCI GGUF layers
f74a525366e6280a9806ecaabd7e1f4df46a6a97 2026-10-17T17:26:28+08:00 feat(pull): add version-pinned GCS objects
833d68f716783c255b24691552a081a1d2b88e76 2026-10-17T17:51:00+08:00 feat(pull): add versioned S3 and SigV4
e62222276e0dac62ed554a3fec2f2ae86edad497 2026-10-17T18:13:08+08:00 feat(pull): add pinned Azure blobs
87f6454fb7c4832262d36104a806baad9a9734a7 2026-10-17T18:24:38+08:00 feat(pull): add pinned ModelScope files
223ca794c9d45a0e246ad1cd9ba8613f2f203cd4 2026-10-17T18:33:14+08:00 feat(pull): add Xet hash and chunk primitives
a774ded154ed066c8f4b134ce7f5cba32ccdd09c 2026-10-17T18:45:39+08:00 feat(pull): add cloud content contracts
837e77f5080c0c2c07fee90b2ef9fb791068a7d3 2026-10-17T18:55:23+08:00 fix(pull): enforce offline acquisition rules
52e42fcdf0e7bead1ebefe55d8103501c7bd4f64 2026-10-17T19:15:13+08:00 feat(pull): add native Xet reconstruction
46cf5a5830e7e8d8096234341ef51fafdf63cc16 2026-10-17T19:24:22+08:00 feat(pull): add conditional cloud reads
4db5efbcce973d21bf1c8ca750471e6b2ab956ab 2026-10-17T19:34:10+08:00 feat(pull): recover pinned ModelScope sizes
f22f75972e63ea70771c59fbce065e371d5f29d4 2026-10-17T19:43:00+08:00 feat(pull): list pinned file candidates
f197c9cf4a44e06a3c1cf3c532855545bed945fa 2026-10-17T19:52:21+08:00 feat(pull): select explicit GGUF quant families
1a22597d9fda50ad71c43ae08f8660fb5b014218 2026-10-17T17:44:06Z docs: preserve complete October 17 takeover and paused sources
04363c3e83368517475b6dccd4894ff437639927 2026-10-17T17:44:24Z feat(pull): qualify bounded full-path filters and exclusion closure
8195c18e43bd2aba51d727c03ebe03df6d1fd166 2026-10-17T18:12:21Z feat(pull): reconstruct verified required native Xet indexes
f56b57400eb913240d6e2b2eeea5d60d9ab6f35e 2026-10-17T19:30:00Z feat(pull): authenticate versioned AWS FULL_OBJECT SHA256
d5aa5cb0ba10d2603ae34488e54755c5495f63b3 2026-10-17T19:48:40Z feat(pull): finish v1 original-model companion closure
fee0bb2e2d690956069a0de151770c21168bea63 2026-10-17T19:57:15Z feat(pull): close official Qwen and Gemma source metadata
4bf624e300254f845e9e047fcb344026c9252d5d 2026-10-17T21:31:11Z feat(pipeline): scaffold shared multimodel routing and opt-in chat context
a9984a47052baaf1aeda558154256d361dcc64af 2026-10-18T09:26:30+08:00 chore(gitignore): share one canonical ignore block
ac44e836b46f6c5aff44afa298bdb342ff65433e 2026-10-18T02:23:14Z feat(pipeline): compose source-scoped context filters and typed decisions
f28a62ceb47229279b70283f4e88747d1719e2ed 2026-10-18T03:23:06Z feat(context): snapshot authorized committed history for pipelines
f0746d1d7e22b94b55c93cb734a8c4ad3b2ab67a 2026-10-18T04:38:18Z feat(pipeline): finish YAML-first typed multimodal scaffolding
93a562288f0e00240da92c37c9dd7f4b846d62d3 2026-10-18T05:21:30Z feat(source): bind verified immutable safetensors shards
b796a2e17a4f20086c4db59743aea9558ae1a892 2026-10-18T06:27:12Z feat(qwen): bind original source tensors to shared CPU value operators
b11e58cb9ef7dd0d607f301d90c1870897cebd34 2026-10-18T07:23:47Z feat(qwen): qualify original hybrid layer CPU proposals
1b761119c2de2690723b9531b1d43660a66c1001 2026-10-18T15:29:41+08:00 docs(agents): unify shared conventions and plan model
631e59fd614fa6576985e8ff8bf5503d88f99359 2026-10-18T15:30:21+08:00 docs(plans): migrate historical plans to the unified model
6329cffbb3e1b93811235867fa697f5f76f3d27a 2026-10-18T08:18:40Z feat(qwen): admit source-bound original layer states atomically
e43ba4c63106bab9804ff0f581457ce4f370741e 2026-10-18T09:16:57Z feat(qwen): compose original full-model FP32 diagnostic forward
28e47f5a4728679e33659298db043da728f225c1 2026-10-18T17:17:44+08:00 docs(agents): add terse-writing guidance
7e328411a8de3c0cc2b9cb9690f2e4a0aa9a54c9 2026-10-18T17:24:36+08:00 docs(agents): strengthen ownership and commit-scope rules
aaf9ba6ec94a9d1bbd4e09b747d9338637fc9e87 2026-10-18T10:09:54Z feat(qwen): qualify source-bound BF16 dtype diagnostics
1c0bb41f2e7b2f635b255d5285040ed15b9c25b6 2026-10-18T11:26:23Z feat(qwen): bind pinned source bundles to numerical reference traces
e90ff8868c46f321ff8bd0bf4c2154a71606aab2 2026-10-18T19:38:06+08:00 docs(agents): add terse-output guidance
35b90932f9bdf93127059df426e7d7eeaee54cf5 2026-10-19T00:19:08Z feat(serve): add bounded OpenAI CPU router and verified SSD extents
9790fa2ef6c4993be0b93014ad1e63d5db5cabbb 2026-10-19T02:24:07Z feat(serve): admit resident GPU requests with fair wave scheduling
114bba566bb2d150afe5f8a0fb5701880cae713f 2026-10-19T03:59:32Z feat(gpu): qualify native multi-row F32 projection on CUDA
2869f8784d986dd80973da03afafde89da34c947 2026-10-19T05:11:09Z feat(gpu): qualify batched attention GDN KV and elementwise operators
08be9c64ea60b5dcf4391c54cc183d63666e4a70 2026-10-19T07:43:08Z feat(gpu): qualify resident B2-B8 state cohort publication
3ec431a2df62bbcae5091a048f285801f151da7e 2026-10-19T08:04:35Z feat(gpu): qualify native batched GGUF quantized projections
c85ad250334f590a87c99710e95c71a3deb28323 2026-10-19T08:41:20Z docs(agents): preserve complete Zerogravity history and P12 handoff
35ebc5b343fd82201eac768edf86ba7a81c41846 2026-10-19T18:08:46+08:00 feat(gpu): fuse native resident model batches
904a190cf39aca74e06c979dbaddb0db7f4404f1 2026-10-19T18:09:43+08:00 docs(agents): record native batch acceptance
54fd580815dbcfcf0916e4e737beaf21728818b2 2026-10-19T19:51:39+08:00 feat(gpu): keep batch greedy feedback on device
570d865696ab398f0c5c37f08c5c951595eadd75 2026-10-19T19:52:16+08:00 docs(agents): record batch feedback acceptance
2677d8b664c78b657c26e7bf46c835317b003913 2026-10-19T22:33:08+08:00 feat(serve): run native CUDA B2 waves
7fc9ad475a30a3d2120f307a7da2ff75a80e337a 2026-10-19T22:33:38+08:00 docs(serve): bound native B2 acceptance
f26232e8b03bf5ab3c13e00ed641eb0d602360d2 2026-10-20T00:28:17+08:00 fix(qwen): honor tied original output heads
6bfcdbe72f0133a6c79e2e7de169c3de74b3da8a 2026-10-20T00:30:04+08:00 feat(bench): account for offered serving work
d9fa829ceba092c7f8a3fa9c2746a11a0906eaba 2026-10-20T00:40:08+08:00 feat(serve): qualify native CUDA rows 4 and 8
9815504a6975ce766766fc8089729be8121df3ac 2026-10-20T01:13:36+08:00 feat(qwen): bind original HF diagnostic profiles
dbb5f4f4cc42e28c91b12e906f9091ef76510c03 2026-10-20T01:53:12+08:00 feat(serve): fuse unequal decode frontiers
cadcbd9d1216b00ca4b69cae9bc04a409e347193 2026-10-20T02:07:20+08:00 feat(qwen): qualify pinned CPU HF operators
1b599df06d8d7c8c4912838836b272e28e04aca1 2026-10-20T02:08:49+08:00 feat(bench): isolate owned provider lifecycles
d41163432fd6233c7bd88f07b64fb71b1086a902 2026-10-20T02:11:18+08:00 feat(artifacts): own verified tensor windows
2a32aff615fc4f7f6718f14a15187825730d6897 2026-10-20T02:43:37+08:00 feat(qwen): compose typed source operators
c5eb1f8e717c79146937e585a145a0052eff747b 2026-10-20T03:22:04+08:00 feat(cpu): qualify source sigmoid and RMS
d0a60844217fc106fe8c70bb3a99073eef0387c8 2026-10-20T04:19:25+08:00 perf(gpu): fuse source activation preparation
5350a4d40b535106240d48fec8aedd459d5cd17a 2026-10-20T04:56:45+08:00 feat(cpu): qualify typed source windows
f78892957f09a84838246aede4fef2be040a0d18 2026-10-20T06:13:20+08:00 feat(gpu): fuse native teacher-prefix serving
bfe62df68d763b9d5c6b87c891be6a62b3593de9 2026-10-20T06:35:48+08:00 feat(cpu): execute leased source-window plans
fe2a34213009cc160f4d82068bc7dd1f49d05c6c 2026-10-20T08:12:19+08:00 feat(cpu): qualify original attention fragments
9a199017f18f1bcb1a8053685be3c3c29236ec90 2026-10-20T08:21:04+08:00 feat(gpu): qualify batched graph replay
fff1842594f8011b761d08fcc1690311b72543e3 2026-10-20T09:20:27+08:00 feat: qualify trained BF16 operator slices
fb68d8fa165de16e21d1bd37afef3790b80da971 2026-10-20T13:27:39+08:00 perf: combine GPU state scatter certification
```

## Branches

```text
+ agent/p12-activation-prep-20261020      d411634 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-activation-prep-20261020) feat(artifacts): own verified tensor windows
+ agent/p12-activation-prep-cold-20261020 d411634 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-activation-prep-cold-20261020) feat(artifacts): own verified tensor windows
+ agent/p12-bf16-dot-20261020             7fc9ad4 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-bf16-dot-20261020) docs(serve): bound native B2 acceptance
+ agent/p12-hf-axes-20261019              7fc9ad4 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-axes-20261019) docs(serve): bound native B2 acceptance
+ agent/p12-hf-binding-20261020           6bfcdbe (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-binding-20261020) feat(bench): account for offered serving work
+ agent/p12-hf-cached-gdn-20261020        d9fa829 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-cached-gdn-20261020) feat(serve): qualify native CUDA rows 4 and 8
+ agent/p12-hf-conv-20261020              d9fa829 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-conv-20261020) feat(serve): qualify native CUDA rows 4 and 8
+ agent/p12-hf-cuda-boundary-20261020     5350a4d (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-cuda-boundary-20261020) feat(cpu): qualify typed source windows
+ agent/p12-hf-norm-20261020              9815504 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-norm-20261020) feat(qwen): bind original HF diagnostic profiles
+ agent/p12-hf-ordinary-contract-20261020 2a32aff (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-contract-20261020) feat(qwen): compose typed source operators
+ agent/p12-hf-ordinary-rms-20261020      d411634 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020) feat(artifacts): own verified tensor windows
+ agent/p12-hf-raw-parameters-20261020    dbb5f4f (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-raw-parameters-20261020) feat(serve): fuse unequal decode frontiers
+ agent/p12-hf-sigmoid-20261020           d411634 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-sigmoid-20261020) feat(artifacts): own verified tensor windows
+ agent/p12-hf-silu-20261020              9815504 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-silu-20261020) feat(qwen): bind original HF diagnostic profiles
+ agent/p12-hf-softplus-20261020          dbb5f4f (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-softplus-20261020) feat(serve): fuse unequal decode frontiers
+ agent/p12-hf-typed-20261020             dbb5f4f (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-typed-20261020) feat(serve): fuse unequal decode frontiers
+ agent/p12-native-prefix-20261020        c5eb1f8 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-native-prefix-20261020) feat(cpu): qualify source sigmoid and RMS
+ agent/p12-parameter-window-20261020     c5eb1f8 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-parameter-window-20261020) feat(cpu): qualify source sigmoid and RMS
+ agent/p12-ragged-20261019               7fc9ad4 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-ragged-20261019) docs(serve): bound native B2 acceptance
+ agent/p12-serialized-bench-20261020     d9fa829 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-serialized-bench-20261020) feat(serve): qualify native CUDA rows 4 and 8
+ agent/p12-tensor-window-20261020        d9fa829 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-tensor-window-20261020) feat(serve): qualify native CUDA rows 4 and 8
+ agent/p12-token-metrics-20261020        7fc9ad4 (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-token-metrics-20261020) docs(serve): bound native B2 acceptance
+ agent/p12-torch-gdn-reduce-20261020     6bfcdbe (/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-torch-gdn-reduce-20261020) feat(bench): account for offered serving work
* main                                    fb68d8f [origin/main: ahead 34] perf: combine GPU state scatter certification
```

## Worktree registry

```text
worktree /home/pvl/spaces/abird/src/zerogravity
HEAD fb68d8fa165de16e21d1bd37afef3790b80da971
branch refs/heads/main

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-activation-prep-20261020
HEAD d41163432fd6233c7bd88f07b64fb71b1086a902
branch refs/heads/agent/p12-activation-prep-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-activation-prep-cold-20261020
HEAD d41163432fd6233c7bd88f07b64fb71b1086a902
branch refs/heads/agent/p12-activation-prep-cold-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-bf16-dot-20261020
HEAD 7fc9ad475a30a3d2120f307a7da2ff75a80e337a
branch refs/heads/agent/p12-bf16-dot-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-capture-host-tests-20261020
HEAD bfe62df68d763b9d5c6b87c891be6a62b3593de9
detached

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-cuda-graph-20261020
HEAD 5350a4d40b535106240d48fec8aedd459d5cd17a
detached

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-axes-20261019
HEAD 7fc9ad475a30a3d2120f307a7da2ff75a80e337a
branch refs/heads/agent/p12-hf-axes-20261019

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-binding-20261020
HEAD 6bfcdbe72f0133a6c79e2e7de169c3de74b3da8a
branch refs/heads/agent/p12-hf-binding-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-cached-gdn-20261020
HEAD d9fa829ceba092c7f8a3fa9c2746a11a0906eaba
branch refs/heads/agent/p12-hf-cached-gdn-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-conv-20261020
HEAD d9fa829ceba092c7f8a3fa9c2746a11a0906eaba
branch refs/heads/agent/p12-hf-conv-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-cuda-boundary-20261020
HEAD 5350a4d40b535106240d48fec8aedd459d5cd17a
branch refs/heads/agent/p12-hf-cuda-boundary-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-norm-20261020
HEAD 9815504a6975ce766766fc8089729be8121df3ac
branch refs/heads/agent/p12-hf-norm-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-contract-20261020
HEAD 2a32aff615fc4f7f6718f14a15187825730d6897
branch refs/heads/agent/p12-hf-ordinary-contract-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020
HEAD d41163432fd6233c7bd88f07b64fb71b1086a902
branch refs/heads/agent/p12-hf-ordinary-rms-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-raw-parameters-20261020
HEAD dbb5f4f4cc42e28c91b12e906f9091ef76510c03
branch refs/heads/agent/p12-hf-raw-parameters-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-sigmoid-20261020
HEAD d41163432fd6233c7bd88f07b64fb71b1086a902
branch refs/heads/agent/p12-hf-sigmoid-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-silu-20261020
HEAD 9815504a6975ce766766fc8089729be8121df3ac
branch refs/heads/agent/p12-hf-silu-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-softplus-20261020
HEAD dbb5f4f4cc42e28c91b12e906f9091ef76510c03
branch refs/heads/agent/p12-hf-softplus-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-typed-20261020
HEAD dbb5f4f4cc42e28c91b12e906f9091ef76510c03
branch refs/heads/agent/p12-hf-typed-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-native-prefix-20261020
HEAD c5eb1f8e717c79146937e585a145a0052eff747b
branch refs/heads/agent/p12-native-prefix-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-parameter-window-20261020
HEAD c5eb1f8e717c79146937e585a145a0052eff747b
branch refs/heads/agent/p12-parameter-window-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-qwen-window-resolver-20261020
HEAD d0a60844217fc106fe8c70bb3a99073eef0387c8
detached

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-ragged-20261019
HEAD 7fc9ad475a30a3d2120f307a7da2ff75a80e337a
branch refs/heads/agent/p12-ragged-20261019

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-serialized-bench-20261020
HEAD d9fa829ceba092c7f8a3fa9c2746a11a0906eaba
branch refs/heads/agent/p12-serialized-bench-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-tensor-window-20261020
HEAD d9fa829ceba092c7f8a3fa9c2746a11a0906eaba
branch refs/heads/agent/p12-tensor-window-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-token-metrics-20261020
HEAD 7fc9ad475a30a3d2120f307a7da2ff75a80e337a
branch refs/heads/agent/p12-token-metrics-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/p12-torch-gdn-reduce-20261020
HEAD 6bfcdbe72f0133a6c79e2e7de169c3de74b3da8a
branch refs/heads/agent/p12-torch-gdn-reduce-20261020

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020
HEAD fff1842594f8011b761d08fcc1690311b72543e3
detached

worktree /home/pvl/spaces/abird/src/zerogravity-worktrees/root-default-bulk-20261020
HEAD fb68d8fa165de16e21d1bd37afef3790b80da971
detached
```

## GPU point telemetry

```text
NVIDIA GeForce RTX 3060 Laptop GPU, GPU-89d28939-1000-40b8-e243-b18f72b3c381, 6144 MiB, 41 MiB, 0 %, 51
```

## GPU compute processes

```text

```

## Worktree pending paths

All observed trees contain pending changes. Ownership must be checked before edits; this inventory does not authorize cleanup or merge.

### `/home/pvl/spaces/abird/src/zerogravity`

```text
 M .agents/plans/model-execution/TODO.md
 M .agents/plans/numerics-and-context/TODO.md
 M .agents/plans/serving-and-speculation/TODO.md
 M crates/zerogravity-gpu/src/lib.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/context.rs
?? .agents/docs/handoff-2026-10-20/
?? .agents/plans/NEXT_AGENT_PROMPT_2026-10-20.md
?? .agents/plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md
?? crates/zerogravity-gpu/src/ordered_silu.rs
?? crates/zerogravity-gpu/src/ordered_silu/
?? crates/zerogravity-gpu/src/resident_program/silu.rs
?? crates/zerogravity-gpu/src/resident_program/silu/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-activation-prep-20261020`

```text
?? crates/zerogravity-gpu/src/activation_quant_fused.rs
?? crates/zerogravity-gpu/src/activation_quant_fused/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-activation-prep-cold-20261020`

```text
 M crates/zerogravity-core/src/accelerator/program.rs
 M crates/zerogravity-cpu/src/original_hf/transcendentals.rs
 M crates/zerogravity-gpu/src/device_activation.rs
 M crates/zerogravity-gpu/src/lib.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-model-qwen35/src/forward.rs
 M crates/zerogravity-model-qwen35/src/host_math.rs
 M crates/zerogravity-model-qwen35/src/lib.rs
 M crates/zerogravity-model-qwen35/src/original_binding.rs
 M crates/zerogravity-model-qwen35/src/original_binding_tests.rs
?? crates/zerogravity-core/src/accelerator/program/
?? crates/zerogravity-cpu/src/original_hf/transcendentals/
?? crates/zerogravity-gpu/src/activation_quant_fused.rs
?? crates/zerogravity-gpu/src/activation_quant_fused/
?? crates/zerogravity-gpu/src/resident_program/typed_refusal_tests.rs
?? crates/zerogravity-model-qwen35/src/operator_slice.rs
?? crates/zerogravity-model-qwen35/src/operator_slice_tests.rs
?? crates/zerogravity-model-qwen35/src/original_binding/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-bf16-dot-20261020`

```text
 M crates/zerogravity-cli/src/serve/gpu_batch.rs
 M crates/zerogravity-cli/src/serve/gpu_engine.rs
 M crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
 M crates/zerogravity-cli/src/serve/gpu_engine/qualification.rs
 M crates/zerogravity-cli/src/serve/mod.rs
 M crates/zerogravity-cli/tests/serve_gpu.rs
 M crates/zerogravity-cpu/src/original_hf.rs
 M crates/zerogravity-cpu/src/weights.rs
 M crates/zerogravity-gpu/src/resident_inference.rs
 M crates/zerogravity-gpu/src/resident_inference/fused.rs
 M crates/zerogravity/src/original_qwen_cpu.rs
 M crates/zerogravity/src/original_qwen_math.rs
 M crates/zerogravity/src/original_qwen_precision.rs
 M crates/zerogravity/src/original_qwen_state.rs
 M crates/zerogravity/src/original_qwen_trace.rs
 M crates/zerogravity/tests/qwen_original_cpu.rs
?? crates/zerogravity-gpu/src/resident_inference/admission_qualification.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-capture-host-tests-20261020`

```text
 M Cargo.toml
 M crates/zerogravity-gpu/src/physical_queue.rs
?? crates/zerogravity-gpu/src/physical_queue/
?? crates/zerogravity-gpu/vendor/cubecl-common/
?? crates/zerogravity-gpu/vendor/cubecl-runtime/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-cuda-graph-20261020`

```text
 M crates/zerogravity-gpu/src/physical_quarantine.rs
 M crates/zerogravity-gpu/src/resident_inference.rs
 M crates/zerogravity-gpu/src/resident_inference/fused.rs
 M crates/zerogravity-gpu/src/resident_leases.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/capture.rs
 M crates/zerogravity-gpu/vendor/cubecl-cuda/src/lib.rs
?? crates/zerogravity-gpu/src/resident_inference/graph.rs
?? crates/zerogravity-gpu/src/resident_inference/graph/
?? crates/zerogravity-gpu/src/resident_program/batch/
?? crates/zerogravity-gpu/src/rope_angles.rs
?? crates/zerogravity-gpu/src/rope_angles/
?? crates/zerogravity-gpu/vendor/cubecl-cuda/src/owned_command.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-axes-20261019`

```text
 M crates/zerogravity-cli/src/serve/gpu_batch.rs
 M crates/zerogravity-cli/src/serve/gpu_engine.rs
 M crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
 M crates/zerogravity-cli/src/serve/gpu_engine/qualification.rs
 M crates/zerogravity-cli/src/serve/mod.rs
 M crates/zerogravity-cli/tests/serve_gpu.rs
 M crates/zerogravity-core/src/accelerator/program.rs
 M crates/zerogravity-gpu/src/resident_inference.rs
 M crates/zerogravity-gpu/src/resident_inference/cold_tests.rs
 M crates/zerogravity-gpu/src/resident_inference/fused.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-model-qwen35/src/forward.rs
 M crates/zerogravity-model-qwen35/src/host_math.rs
?? crates/zerogravity-gpu/src/resident_inference/admission_qualification.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-binding-20261020`

```text
 M crates/zerogravity-core/src/accelerator/program.rs
 M crates/zerogravity-gpu/src/resident_inference/cold_tests.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-model-qwen35/Cargo.toml
 M crates/zerogravity-model-qwen35/src/checkpoint.rs
 M crates/zerogravity-model-qwen35/src/forward.rs
 M crates/zerogravity-model-qwen35/src/host_math.rs
 M crates/zerogravity-model-qwen35/src/lib.rs
?? crates/zerogravity-model-qwen35/src/original_binding.rs
?? crates/zerogravity-model-qwen35/src/original_binding_tests.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-cached-gdn-20261020`

```text
 M crates/zerogravity-cpu/src/original_hf.rs
?? crates/zerogravity-cpu/src/original_hf/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-conv-20261020`

```text
 M crates/zerogravity-cpu/src/original_hf.rs
 M crates/zerogravity-cpu/src/weights.rs
?? crates/zerogravity-cpu/src/original_hf/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-cuda-boundary-20261020`

```text
 M crates/zerogravity-gpu/src/lib.rs
 M crates/zerogravity-gpu/src/resident_leases.rs
 M crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/server.rs
 M crates/zerogravity-gpu/vendor/cubecl-cuda/src/lib.rs
?? crates/zerogravity-gpu/src/hf_cuda.rs
?? crates/zerogravity-gpu/src/hf_cuda/
?? crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/server/
?? crates/zerogravity-gpu/vendor/cubecl-cuda/src/owned_command.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-norm-20261020`

```text
 M crates/zerogravity-cpu/src/original_hf.rs
?? crates/zerogravity-cpu/src/original_hf/norm.rs
?? crates/zerogravity-cpu/src/original_hf/norm/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-contract-20261020`

```text
 M crates/zerogravity-core/src/accelerator/program.rs
 M crates/zerogravity-core/src/accelerator/program/typed.rs
 M crates/zerogravity-core/src/accelerator/program/typed_tests.rs
 M crates/zerogravity-cpu/src/original_hf/norm.rs
 M crates/zerogravity-cpu/src/original_hf/transcendentals.rs
 M crates/zerogravity-model-qwen35/src/forward.rs
 M crates/zerogravity-model-qwen35/src/host_math.rs
 M crates/zerogravity-model-qwen35/src/operator_slice.rs
 M crates/zerogravity-model-qwen35/src/operator_slice_tests.rs
?? crates/zerogravity-cpu/src/original_hf/norm/ordinary.rs
?? crates/zerogravity-cpu/src/original_hf/norm/ordinary/
?? crates/zerogravity-cpu/src/original_hf/transcendentals/sigmoid.rs
?? crates/zerogravity-cpu/src/original_hf/transcendentals/sigmoid_beta_golden.txt
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020`

```text
 M crates/zerogravity-cpu/Cargo.toml
 M crates/zerogravity-cpu/src/original_hf.rs
 M crates/zerogravity-cpu/src/original_hf/norm.rs
 M crates/zerogravity-cpu/src/original_hf/transcendentals.rs
 M crates/zerogravity-cpu/src/weights.rs
?? crates/zerogravity-cpu/src/original_hf/norm/ordinary.rs
?? crates/zerogravity-cpu/src/original_hf/norm/ordinary/
?? crates/zerogravity-cpu/src/original_hf/transcendentals/
?? crates/zerogravity-cpu/src/original_hf/window_projection.rs
?? crates/zerogravity-cpu/src/original_hf/window_projection/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-raw-parameters-20261020`

```text
 M crates/zerogravity-model-qwen35/src/original_binding.rs
 M crates/zerogravity-model-qwen35/src/original_binding_tests.rs
?? crates/zerogravity-model-qwen35/src/original_binding/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-sigmoid-20261020`

```text
 M crates/zerogravity-cpu/src/original_hf/transcendentals.rs
?? crates/zerogravity-cpu/src/original_hf/transcendentals/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-silu-20261020`

```text
 M crates/zerogravity-cpu/src/original_hf.rs
?? crates/zerogravity-cpu/src/original_hf/transcendentals.rs
?? crates/zerogravity-cpu/src/original_hf/transcendentals_silu_golden.txt
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-softplus-20261020`

```text
 M crates/zerogravity-cpu/src/original_hf.rs
?? crates/zerogravity-cpu/src/original_hf/transcendentals.rs
?? crates/zerogravity-cpu/src/original_hf/transcendentals/
?? crates/zerogravity-cpu/src/original_hf/transcendentals_silu_golden.txt
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-typed-20261020`

```text
 M crates/zerogravity-core/src/accelerator/program.rs
 M crates/zerogravity-cpu/src/original_hf.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-model-qwen35/src/forward.rs
 M crates/zerogravity-model-qwen35/src/host_math.rs
 M crates/zerogravity-model-qwen35/src/lib.rs
?? crates/zerogravity-core/src/accelerator/program/
?? crates/zerogravity-gpu/src/resident_program/typed_refusal_tests.rs
?? crates/zerogravity-model-qwen35/src/operator_slice.rs
?? crates/zerogravity-model-qwen35/src/operator_slice_tests.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-native-prefix-20261020`

```text
 M crates/zerogravity-cli/src/serve/gpu_batch.rs
 M crates/zerogravity-cli/src/serve/gpu_engine.rs
 M crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
 M crates/zerogravity-cli/src/serve/mod.rs
 M crates/zerogravity-cli/tests/serve_gpu.rs
 M crates/zerogravity-gpu/src/device_activation.rs
 M crates/zerogravity-gpu/src/lib.rs
 M crates/zerogravity-gpu/src/resident_inference/fused.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
?? crates/zerogravity-cli/src/serve/gpu_engine/qualification/
?? crates/zerogravity-cli/tests/serve_gpu/
?? crates/zerogravity-gpu/src/activation_quant_fused.rs
?? crates/zerogravity-gpu/src/activation_quant_fused/
?? crates/zerogravity-gpu/src/resident_inference/prefix_qualification.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-parameter-window-20261020`

```text
 M crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
 M crates/zerogravity-artifacts/tests/tensor_window.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-qwen-window-resolver-20261020`

```text
 M crates/zerogravity-artifacts/src/model/qwen_owned.rs
 M crates/zerogravity-artifacts/src/source_weights.rs
 M crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
 M crates/zerogravity-artifacts/tests/tensor_window.rs
 M crates/zerogravity-core/src/accelerator/program.rs
 M crates/zerogravity-core/src/accelerator/program/typed.rs
 M crates/zerogravity-core/src/accelerator/program/typed_tests.rs
 M crates/zerogravity-cpu/Cargo.toml
 M crates/zerogravity-cpu/src/original_hf.rs
 M crates/zerogravity-cpu/src/original_hf/transcendentals.rs
 M crates/zerogravity-cpu/src/weights.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-model-qwen35/Cargo.toml
 M crates/zerogravity-model-qwen35/src/forward.rs
 M crates/zerogravity-model-qwen35/src/host_math.rs
 M crates/zerogravity-model-qwen35/src/operator_slice.rs
 M crates/zerogravity-model-qwen35/src/operator_slice_tests.rs
 M crates/zerogravity-model-qwen35/src/original_binding.rs
?? crates/zerogravity-cpu/src/original_hf/transcendentals/softmax.rs
?? crates/zerogravity-cpu/src/original_hf/transcendentals/softmax/
?? crates/zerogravity-cpu/src/original_hf/window_embedding.rs
?? crates/zerogravity-cpu/src/original_hf/window_embedding/
?? crates/zerogravity-cpu/src/original_hf/window_projection.rs
?? crates/zerogravity-cpu/src/original_hf/window_projection/
?? crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
?? crates/zerogravity-model-qwen35/src/original_binding/window.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-ragged-20261019`

```text
 M crates/zerogravity-cli/src/serve/gpu_batch.rs
 M crates/zerogravity-cli/src/serve/gpu_engine.rs
 M crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
 M crates/zerogravity-cli/src/serve/gpu_engine/qualification.rs
 M crates/zerogravity-cli/src/serve/mod.rs
 M crates/zerogravity-cli/tests/serve_gpu.rs
 M crates/zerogravity-gpu/src/attention_value.rs
 M crates/zerogravity-gpu/src/resident_batch.rs
 M crates/zerogravity-gpu/src/resident_bindings.rs
 M crates/zerogravity-gpu/src/resident_fused_greedy_qualification.rs
 M crates/zerogravity-gpu/src/resident_fused_qualification.rs
 M crates/zerogravity-gpu/src/resident_inference.rs
 M crates/zerogravity-gpu/src/resident_inference/cohort_qualification.rs
 M crates/zerogravity-gpu/src/resident_inference/fused.rs
 M crates/zerogravity-gpu/src/resident_leases.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-gpu/src/resident_provider.rs
 M crates/zerogravity-gpu/src/rope_kv.rs
?? crates/zerogravity-gpu/src/resident_inference/admission_qualification.rs
?? crates/zerogravity-gpu/src/resident_inference/ragged_qualification.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-serialized-bench-20261020`

```text
 M tools/serve-bench.py
 M tools/serve-compare.py
?? tools/serve-lifecycle.py
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-tensor-window-20261020`

```text
 M crates/zerogravity-artifacts/src/artifact.rs
 M crates/zerogravity-artifacts/src/source_weights.rs
 M crates/zerogravity-artifacts/src/ssd_extent.rs
 M crates/zerogravity-artifacts/src/verified_shards.rs
 M crates/zerogravity-artifacts/tests/ssd_extent.rs
 M crates/zerogravity-core/src/accelerator/program.rs
?? crates/zerogravity-artifacts/src/source_weights/
?? crates/zerogravity-artifacts/tests/tensor_window.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-token-metrics-20261020`

```text
 M crates/zerogravity-cli/src/serve/engine.rs
 M crates/zerogravity-cli/src/serve/gpu_batch.rs
 M crates/zerogravity-cli/src/serve/gpu_engine.rs
 M crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
 M crates/zerogravity-cli/src/serve/gpu_engine/qualification.rs
 M crates/zerogravity-cli/src/serve/http.rs
 M crates/zerogravity-cli/src/serve/mod.rs
 M crates/zerogravity-cli/tests/serve_gpu.rs
 M crates/zerogravity-gpu/src/attention_value.rs
 M crates/zerogravity-gpu/src/resident_batch.rs
 M crates/zerogravity-gpu/src/resident_bindings.rs
 M crates/zerogravity-gpu/src/resident_fused_greedy_qualification.rs
 M crates/zerogravity-gpu/src/resident_fused_qualification.rs
 M crates/zerogravity-gpu/src/resident_inference.rs
 M crates/zerogravity-gpu/src/resident_inference/cohort_qualification.rs
 M crates/zerogravity-gpu/src/resident_inference/fused.rs
 M crates/zerogravity-gpu/src/resident_leases.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-gpu/src/resident_provider.rs
 M crates/zerogravity-gpu/src/rope_kv.rs
 M tools/serve-bench.py
 M tools/serve-compare.py
?? crates/zerogravity-cli/src/serve/token_metrics.rs
?? crates/zerogravity-gpu/src/resident_inference/admission_qualification.rs
?? crates/zerogravity-gpu/src/resident_inference/ragged_qualification.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-torch-gdn-reduce-20261020`

```text
 M crates/zerogravity-cpu/src/original_hf.rs
?? crates/zerogravity-cpu/src/original_hf/
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`

```text
 M crates/zerogravity-core/src/accelerator/program.rs
 M crates/zerogravity-cpu/src/original_hf.rs
 M crates/zerogravity-cpu/src/original_hf/norm.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-model-qwen35/src/attention_slice_tests.rs
 M crates/zerogravity-model-qwen35/src/forward.rs
 M crates/zerogravity-model-qwen35/src/host_math.rs
 M crates/zerogravity-model-qwen35/src/operator_slice.rs
 M crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
 M crates/zerogravity-model-qwen35/src/original_binding/window.rs
?? crates/zerogravity-core/src/accelerator/program/cached_gdn.rs
?? crates/zerogravity-core/src/accelerator/program/coefficient_preparation.rs
?? crates/zerogravity-core/src/accelerator/program/gated_rms.rs
?? crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs
?? crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma.rs
?? crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma/
?? crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs
?? crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs
?? crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
?? crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
?? crates/zerogravity-model-qwen35/src/gdn_cached_slice.rs
?? crates/zerogravity-model-qwen35/src/gdn_gated_rms_slice.rs
?? crates/zerogravity-model-qwen35/src/host_math/cached_gdn.rs
?? crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs
?? crates/zerogravity-model-qwen35/src/host_math/gated_rms.rs
?? crates/zerogravity-model-qwen35/src/operator_slice/
?? crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs
?? crates/zerogravity-model-qwen35/src/original_binding/window/gated_rms.rs
?? crates/zerogravity-model-qwen35/src/original_binding/window/pure.rs
```

### `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-default-bulk-20261020`

```text
 M crates/zerogravity-core/src/accelerator/program.rs
 M crates/zerogravity-cpu/src/math.rs
 M crates/zerogravity-gpu/src/resident_program.rs
 M crates/zerogravity-gpu/src/resident_program/batch.rs
 M crates/zerogravity-model-qwen35/src/forward.rs
 M crates/zerogravity-model-qwen35/src/host_math.rs
 M crates/zerogravity-model-qwen35/src/operator_slice.rs
?? crates/zerogravity-core/src/accelerator/program/bulk_f32.rs
?? crates/zerogravity-core/src/accelerator/program/bulk_f32/
?? crates/zerogravity-cpu/src/math/
?? crates/zerogravity-model-qwen35/src/bulk_f32_slice.rs
?? crates/zerogravity-model-qwen35/src/forward/
?? crates/zerogravity-model-qwen35/src/host_math/bulk_f32.rs
```

## Existing CUDA paths (existence only)

- `/home/pvl/spaces/abird/src/zerogravity/.agents/runs/p12-full-fusion-root-2026-10-19/cuda-view`: exists=True.
- `/nix/store/cs50fahgnln1k5cx7zs0dnyikzqzf4iq-cuda12.9-cuda_nvrtc-12.9.86-lib/lib`: exists=True.
- `/run/opengl-driver/lib`: exists=True.

Use absolute MAIN CUDA_PATH when a future authorized run works in an isolated tree. Clear inherited LD_PRELOAD/LD_AUDIT. Rootless Podman must use its existing full UID-map/private-loopback route, not native-gpu.py single-UID nesting.
