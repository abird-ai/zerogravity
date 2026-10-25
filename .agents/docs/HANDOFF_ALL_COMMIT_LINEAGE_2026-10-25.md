# ZeroGravity — lossless source-commit lineage snapshot (oldest first)

Current HEAD at generation: `d84853337bbd2830f50a9c4e704d266990fd2ce4`.
Total reachable commits: **245**.

> Git metadata may carry author/committer dates later than the repository review date (especially imported historical work). This index deliberately uses **Git ancestry order** and subjects, not timestamps, to establish implementation order. Each SHA resolves the exact source diff; proposals and benchmark claims must be independently checked in code/evidence.

See [main handoff](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md) for interpretation, exact gate status, architecture, and next steps. This index is historical, never an independent acceptance record.

| # | Commit | Recorded subject |
|---:|---|---|
| 1 | `be95424837de` | docs(design): product, architecture, prior-art and research plans |
| 2 | `1e10b5cc1d5c` | feat(core): immutable source identity, state roots and physical resource ownership |
| 3 | `af9a5d3969c9` | feat(artifacts): sharded source artifacts and model identity |
| 4 | `b276fd01e421` | test(artifacts): cache transactions |
| 5 | `586d440b1447` | test(artifacts): qwen text contract |
| 6 | `782e6fc48f09` | feat(cpu): portable CPU math and weights |
| 7 | `1fd12da78401` | feat(model-qwen35): source-verified Qwen3.5 forward and tokenizer |
| 8 | `901ba9ae2f2e` | test(model-qwen35): source layout contract |
| 9 | `e6ecf02bb7d9` | feat(model-gemma4): Gemma contract, MoE and 26B variant |
| 10 | `81642a9af4d8` | test(model-gemma4): e2b contract |
| 11 | `7aa19c268026` | test(model-gemma4): family contract |
| 12 | `65b026f90250` | test(model-gemma4): variant 26b contract |
| 13 | `b03c16451d65` | feat(data): vendored CubeCL license and provenance |
| 14 | `f491e34fe55e` | feat(data): vendor CubeCL cubecl-ir 0.11.0 |
| 15 | `627c58ec80ff` | feat(data): vendor CubeCL cubecl-cpp 0.11.0 |
| 16 | `3664f975e792` | feat(data): vendor CubeCL cubecl-cuda 0.11.0 |
| 17 | `f77ae37ec031` | feat(data): vendor CubeCL cubecl-hip 0.11.0 |
| 18 | `7d768a080a84` | feat(gpu): source-exact resident GPU execution and physical ownership |
| 19 | `a9723bceb8ad` | feat(runtime): facade, semantic program, task drivers and state publication |
| 20 | `c3d0d0b06fb1` | test(runtime): gemma contract |
| 21 | `4add50a023d4` | test(runtime): layout api |
| 22 | `b4a36b829031` | test(runtime): multimodal mm0 |
| 23 | `a691e21c040d` | test(runtime): qwen real model |
| 24 | `25ec3e27e3c2` | test(runtime): qwen shared model |
| 25 | `047e3b70bbcb` | feat(cli): single-binary inspect and run |
| 26 | `52e37313bff5` | test(cli): CLI contract |
| 27 | `22e0cc5abb4a` | feat(tooling): workspace GPU qualification crate |
| 28 | `4d82c6704dad` | test(tooling): independent reference oracle llama_reference_layers |
| 29 | `a2d6a02f1577` | test(tooling): independent reference oracle llama_reference_logits |
| 30 | `a8ea8926b53d` | test(tooling): independent reference oracle llama_reference_qk |
| 31 | `6a0079eb167d` | test(tooling): independent reference oracle llama_reference_recurrent |
| 32 | `be88e30733f9` | test(tooling): independent reference oracle llama_reference_rope |
| 33 | `77778a2c2587` | test(tooling): independent reference oracle llama_reference_softmax |
| 34 | `6015241a5f5c` | test(tooling): independent reference oracle llama_reference_value |
| 35 | `e190aff1fdd3` | test(tooling): independent reference oracle llama_reference_value_padding |
| 36 | `0aea8de66e3e` | test(tooling): build-release-artifacts.sh |
| 37 | `057987203d5d` | test(tooling): check.sh |
| 38 | `c74450a27b1e` | test(tooling): compare-attention-value.py |
| 39 | `9f54d2435fe6` | test(tooling): compare-qwen-layers.py |
| 40 | `7458ffc286f7` | test(tooling): cpu-qualification-gate.py |
| 41 | `73cb6e55e179` | test(tooling): native-gpu.py |
| 42 | `80145a7f9172` | test(tooling): test-install.sh |
| 43 | `ca1138b8cf86` | test(tooling): test-native-gpu.py |
| 44 | `5a0636c3f9c7` | test(tooling): test-numeric-resume.py |
| 45 | `9a6c504ebef0` | chore(workspace): restore the locked full workspace manifest |
| 46 | `57a9ad7fe78b` | test(tooling): independent reference probe notes |
| 47 | `1af2ac434279` | fix(nix): give module check a complete base system |
| 48 | `0452c594d99d` | docs(plan): record module eval check fix |
| 49 | `2afa32aa74af` | docs(readme): reorder engineering and FAQ sections |
| 50 | `a054d396e1ec` | docs(agents): document plan layout and TODO spec |
| 51 | `cc3cab27a271` | docs(contributing): align guide and AI disclosure |
| 52 | `7e61cb63efd0` | docs(agents): gate pushes and destructive actions |
| 53 | `9239719c2512` | docs: retire historical archive bodies |
| 54 | `6deebf222283` | refactor(cache): namespace model cache under abird/zerogravity |
| 55 | `fa6affa42508` | feat(artifacts): add pinned model pulls |
| 56 | `3c97c5207556` | feat(artifacts): resume interrupted pulls |
| 57 | `baa9d8818061` | feat(artifacts): load pinned cache components |
| 58 | `20f449d20320` | feat(artifacts): recover missing pull sizes |
| 59 | `f44264c01785` | feat(cache): add protected snapshot removal |
| 60 | `ea700f67fca7` | feat(pull)!: add capped auxiliary streaming |
| 61 | `03b69482a5fb` | feat(pull): pin multi-asset release components |
| 62 | `e44f4ee8020d` | feat(cache): import local HF snapshots |
| 63 | `fce1353e649d` | feat(cache): qualify cross-filesystem imports |
| 64 | `e0073a5593e5` | feat(pull): add pinned HTTPS objects |
| 65 | `e82d76b8f8e9` | feat(pull): add pinned OCI GGUF layers |
| 66 | `f74a525366e6` | feat(pull): add version-pinned GCS objects |
| 67 | `833d68f71678` | feat(pull): add versioned S3 and SigV4 |
| 68 | `e62222276e0d` | feat(pull): add pinned Azure blobs |
| 69 | `87f6454fb7c4` | feat(pull): add pinned ModelScope files |
| 70 | `223ca794c9d4` | feat(pull): add Xet hash and chunk primitives |
| 71 | `a774ded154ed` | feat(pull): add cloud content contracts |
| 72 | `837e77f5080c` | fix(pull): enforce offline acquisition rules |
| 73 | `52e42fcdf0e7` | feat(pull): add native Xet reconstruction |
| 74 | `46cf5a5830e7` | feat(pull): add conditional cloud reads |
| 75 | `4db5efbcce97` | feat(pull): recover pinned ModelScope sizes |
| 76 | `f22f75972e63` | feat(pull): list pinned file candidates |
| 77 | `f197c9cf4a44` | feat(pull): select explicit GGUF quant families |
| 78 | `1a22597d9fda` | docs: preserve complete October 17 takeover and paused sources |
| 79 | `04363c3e8336` | feat(pull): qualify bounded full-path filters and exclusion closure |
| 80 | `8195c18e43bd` | feat(pull): reconstruct verified required native Xet indexes |
| 81 | `f56b57400eb9` | feat(pull): authenticate versioned AWS FULL_OBJECT SHA256 |
| 82 | `d5aa5cb0ba10` | feat(pull): finish v1 original-model companion closure |
| 83 | `fee0bb2e2d69` | feat(pull): close official Qwen and Gemma source metadata |
| 84 | `4bf624e30025` | feat(pipeline): scaffold shared multimodel routing and opt-in chat context |
| 85 | `a9984a47052b` | chore(gitignore): share one canonical ignore block |
| 86 | `ac44e836b46f` | feat(pipeline): compose source-scoped context filters and typed decisions |
| 87 | `f28a62ceb472` | feat(context): snapshot authorized committed history for pipelines |
| 88 | `f0746d1d7e22` | feat(pipeline): finish YAML-first typed multimodal scaffolding |
| 89 | `93a562288f0e` | feat(source): bind verified immutable safetensors shards |
| 90 | `b796a2e17a4f` | feat(qwen): bind original source tensors to shared CPU value operators |
| 91 | `b11e58cb9ef7` | feat(qwen): qualify original hybrid layer CPU proposals |
| 92 | `1b761119c2de` | docs(agents): unify shared conventions and plan model |
| 93 | `631e59fd614f` | docs(plans): migrate historical plans to the unified model |
| 94 | `6329cffbb3e1` | feat(qwen): admit source-bound original layer states atomically |
| 95 | `e43ba4c63106` | feat(qwen): compose original full-model FP32 diagnostic forward |
| 96 | `28e47f5a4728` | docs(agents): add terse-writing guidance |
| 97 | `7e328411a8de` | docs(agents): strengthen ownership and commit-scope rules |
| 98 | `aaf9ba6ec94a` | feat(qwen): qualify source-bound BF16 dtype diagnostics |
| 99 | `1c0bb41f2e7b` | feat(qwen): bind pinned source bundles to numerical reference traces |
| 100 | `e90ff8868c46` | docs(agents): add terse-output guidance |
| 101 | `35b90932f9bd` | feat(serve): add bounded OpenAI CPU router and verified SSD extents |
| 102 | `9790fa2ef6c4` | feat(serve): admit resident GPU requests with fair wave scheduling |
| 103 | `114bba566bb2` | feat(gpu): qualify native multi-row F32 projection on CUDA |
| 104 | `2869f8784d98` | feat(gpu): qualify batched attention GDN KV and elementwise operators |
| 105 | `08be9c64ea60` | feat(gpu): qualify resident B2-B8 state cohort publication |
| 106 | `3ec431a2df62` | feat(gpu): qualify native batched GGUF quantized projections |
| 107 | `c85ad250334f` | docs(agents): preserve complete Zerogravity history and P12 handoff |
| 108 | `35ebc5b343fd` | feat(gpu): fuse native resident model batches |
| 109 | `904a190cf39a` | docs(agents): record native batch acceptance |
| 110 | `54fd580815db` | feat(gpu): keep batch greedy feedback on device |
| 111 | `570d865696ab` | docs(agents): record batch feedback acceptance |
| 112 | `2677d8b664c7` | feat(serve): run native CUDA B2 waves |
| 113 | `7fc9ad475a30` | docs(serve): bound native B2 acceptance |
| 114 | `f26232e8b03b` | fix(qwen): honor tied original output heads |
| 115 | `6bfcdbe72f01` | feat(bench): account for offered serving work |
| 116 | `d9fa829ceba0` | feat(serve): qualify native CUDA rows 4 and 8 |
| 117 | `9815504a6975` | feat(qwen): bind original HF diagnostic profiles |
| 118 | `dbb5f4f4cc42` | feat(serve): fuse unequal decode frontiers |
| 119 | `cadcbd9d1216` | feat(qwen): qualify pinned CPU HF operators |
| 120 | `1b599df06d8d` | feat(bench): isolate owned provider lifecycles |
| 121 | `d41163432fd6` | feat(artifacts): own verified tensor windows |
| 122 | `2a32aff615fc` | feat(qwen): compose typed source operators |
| 123 | `c5eb1f8e717c` | feat(cpu): qualify source sigmoid and RMS |
| 124 | `d0a60844217f` | perf(gpu): fuse source activation preparation |
| 125 | `5350a4d40b53` | feat(cpu): qualify typed source windows |
| 126 | `f78892957f09` | feat(gpu): fuse native teacher-prefix serving |
| 127 | `bfe62df68d76` | feat(cpu): execute leased source-window plans |
| 128 | `fe2a34213009` | feat(cpu): qualify original attention fragments |
| 129 | `9a199017f18f` | feat(gpu): qualify batched graph replay |
| 130 | `fff1842594f8` | feat: qualify trained BF16 operator slices |
| 131 | `fb68d8fa165d` | perf: combine GPU state scatter certification |
| 132 | `bb089823653f` | feat(core): add run policy request validation |
| 133 | `bd277cf76673` | feat(core): add reserved domains and workspace |
| 134 | `35c9962d8633` | feat(core): admit device backing views |
| 135 | `eb49fa0e945b` | feat(core): record plan topology and source programs |
| 136 | `ec30569bc6b9` | refactor(cpu): share reference math lanes |
| 137 | `9dd1ebfb1862` | feat(cpu): add original torch CPU operators |
| 138 | `a3716fd0de50` | feat(artifacts): select cached sources by locator |
| 139 | `77c48b315fee` | feat(artifacts): authenticate reusable source access |
| 140 | `7ba9278b0a1a` | feat(cpu): evaluate reusable text source windows |
| 141 | `dbd836351d40` | feat(cpu): add haswell and zen bulk kernels |
| 142 | `b026d0f9d502` | feat(tools): add qwen27b output-tail reference |
| 143 | `85d86b3c5863` | feat(tools): add qwen27b attention-tail reference |
| 144 | `0c7a4a9770f6` | feat(tools): add qwen27b full-window reference |
| 145 | `af5aa8498550` | feat(tools): add qwen27b layer9-prefix reference |
| 146 | `6dd5970776b7` | feat(tools): add qwen27b initial-core reference |
| 147 | `2c70ccb66c78` | feat(tools): add gemma4 tokenizer reference |
| 148 | `c870725898c9` | feat(tools): add gemma4-e2b text reference |
| 149 | `8b0cc32c9510` | feat(tools): add gemma4-e2b CPU matrix |
| 150 | `283765cff8bd` | feat(tools): add portable BF16 slice reference |
| 151 | `b5015e7a77e3` | feat(gemma4): admit dense 31B text metadata |
| 152 | `bebfeaf2d817` | feat(gemma4): add E2B text reference and tokenizer |
| 153 | `3002089997fc` | feat(gemma4): add E2B learned-source diagnostics |
| 154 | `225fc5e1a520` | feat(qwen35): record window, bulk and CPU tail sources |
| 155 | `9af407bf3ff5` | feat(qwen35): add qwen27b window replays |
| 156 | `c4cbff17e6c4` | feat(cuda): observe device metadata |
| 157 | `d9d4f8b9bb50` | feat(cuda): own compiler, source and cubin actor paths |
| 158 | `4bc7c429d585` | feat(hip): expose availability and prepared native sets |
| 159 | `248bcec54c6e` | build(gpu): add qualification feature surface |
| 160 | `68c2f5c05013` | feat(gpu): add shared kernel source primitives |
| 161 | `abca70fdebda` | feat(gpu): add original source arithmetic kernels |
| 162 | `63583681ae8c` | feat(gpu): add native policy and physical identity |
| 163 | `fe95ecdde780` | feat(gpu): add vLLM input RMS qualification kernels |
| 164 | `405e2462ff31` | feat(gpu): add vLLM GDN instruction kernels |
| 165 | `71c59964149f` | feat(gpu): add Qwen2B GDN instruction assets |
| 166 | `00d4bbabde04` | feat(gpu): add .8B short-HF GDN call catalog |
| 167 | `d031166b726a` | feat(gpu): add vLLM CUDA qualification scaffolding |
| 168 | `8dbe361c1e53` | feat(gpu): add layer, attention and GDN cubin checks |
| 169 | `13ae2ffb7d05` | feat(gpu): add canonical47 model qualifications |
| 170 | `bbafb1de8c76` | feat(gpu): add Qwen2B qualification leaves |
| 171 | `f619f7eed837` | feat(gpu): add short-HF and full-model checks |
| 172 | `3e89f1919993` | feat(gpu): wire resident leases, program and native sets |
| 173 | `136a4a8692d9` | feat(facade): expose CPU routing preparation |
| 174 | `08b30b571e4b` | feat(facade): add canonical47 original-HF loader |
| 175 | `bc66207ce9c5` | feat(facade): add short-HF serving source loader |
| 176 | `e44a53f75320` | test(facade): add short-HF physical suite |
| 177 | `08fc5935e124` | test(facade): add short-HF cohort suite |
| 178 | `a4a5b6322f34` | feat(facade): add Gemma4 E2B CPU candidate |
| 179 | `2058dcb1ea3f` | feat(facade): add qwen27b HIP teacher source |
| 180 | `71f496b190dc` | feat(facade): add Gemma4 HIP drivers |
| 181 | `752642361cb4` | feat(facade): add shared HIP lifecycle driver |
| 182 | `2bd8167db2c1` | build(cli): gate finite HF serving and run features |
| 183 | `b3c5a532d72a` | feat(cli): route run requests and add gemma4 candidate |
| 184 | `45570bdb3d7f` | perf(serve): plan atomic source prefill waves |
| 185 | `e6b07efdbb2d` | feat(serve): route and serve canonical47 sources |
| 186 | `9999adb1a66d` | feat(serve): serve short-HF BF16 B4 cohorts |
| 187 | `ef9cffd413df` | chore(scripts): extend gate for HF features |
| 188 | `2225af34b82e` | docs(agents): codify parallel execution cadence |
| 189 | `abb6b3b3b233` | docs(research): compare optimization mechanisms |
| 190 | `4b44d313bede` | docs(policy): adopt reuse-first provenance rule |
| 191 | `46996b66e39e` | docs(plans): design composition and batch state |
| 192 | `6ef74b048e83` | docs(cpu-gpu): record shared memory and parity |
| 193 | `fb7ea9a9a77b` | docs(numerics): record F32 closure checkpoints |
| 194 | `424c672768b3` | docs(models): record source20 and Qwen audit |
| 195 | `da28f851448b` | docs(acquisition): record Gemma E2B acquisition |
| 196 | `0c966799d783` | docs(cpu-gpu): record current checkpoints |
| 197 | `c57648ac54ff` | docs(memory): adopt shared pools and paged state |
| 198 | `b552e2198d66` | docs(composition): record shared HIP checkpoints |
| 199 | `6075a4386589` | docs(serving): certify transport and add batching |
| 200 | `3b9da530d922` | docs(distributed): defer mechanism reconciliation |
| 201 | `38e3d77eba85` | docs(platform): add Vulkan parity and reuse gates |
| 202 | `a765f221279c` | docs(cli): add device routing plan |
| 203 | `309b17d81bfb` | docs(handoff): archive history and run notes |
| 204 | `ac17d49c07b2` | docs(handoff): preserve paused continuation text |
| 205 | `7115754b639a` | docs(handoff): preserve draft patches |
| 206 | `11a47c74ecc5` | docs(handoff): preserve takeover and audit reports |
| 207 | `b5d814f8fcab` | docs(handoff): preserve canonical sourcebooks |
| 208 | `e11fb986091f` | docs(handoff): index the 20 Oct handoff packet |
| 209 | `4bf36731cfc4` | docs(plans): activate CLI, composition and compute |
| 210 | `8f01635305af` | feat(core): portable admission and thread-control roots |
| 211 | `5d40cf507a2e` | feat(cli): shared pending queue and ingress credit |
| 212 | `52f76a0e0dc3` | fix(cli): short-HF tokenizer credit and HOST8 |
| 213 | `efd7a34220ce` | fix(gpu): CUDA fragment inventory and host bound |
| 214 | `19ff0598dd79` | feat(gpu): Vulkan native catalogue via cubecl-wgpu |
| 215 | `29a98c1c7c08` | feat(gpu): HIP source-page events and witness |
| 216 | `e8db11115899` | feat(examples): Qwen27B HIP matrix, natural tools |
| 217 | `332038ea78ec` | feat(examples): Gemma4 HIP final-owner and matrix |
| 218 | `68e13a8114e1` | chore(cargo): lock assembled five-lane graph |
| 219 | `74d0c7199870` | docs(plans): record assembly, C16 and handoff |
| 220 | `1269b8eae908` | docs(plans): retire five landed lane trees |
| 221 | `06b89da45565` | fix(gpu): HIP probe witness page id typing |
| 222 | `c217c35d480e` | fix(qualification): V state base parents, attempt-2 bases |
| 223 | `3941a3c9fe0e` | fix(qualification): Gemma resident report-phase envelope 4GiB |
| 224 | `e6ab074e3118` | fix(qualification): H/G source census skips build dirs |
| 225 | `1565683442e8` | docs(plans): record queue GPU acceptance results |
| 226 | `2304f8b4d207` | fix(cli): precedence, ambiguity, pre-client hold |
| 227 | `d3d21a35c5b7` | feat(gpu): B4 grouped transport and certification |
| 228 | `6f1e0947cdf7` | docs(plans): record main checkpoint and B4 merge |
| 229 | `9fbe15e9e45c` | docs(handoff): record checkpoint, HGV pass 3, successor queue |
| 230 | `5749b732dc75` | docs(handoff): record Qwen27B capture lane yield detail |
| 231 | `488e0ea2c21e` | fix: repair native qualification contracts and GPU feature isolation |
| 232 | `93b611acaa65` | fix: prevent Gemma HIP source-control payload shadowing |
| 233 | `6edc6e2c99d9` | docs: pin refreshed H G packages and completed GPU-free controls |
| 234 | `9e67b74000bc` | docs(audit): independently validate post-4bf GPU boundaries and resume holds |
| 235 | `03f2e662f219` | fix(gpu): certify grouped B4 CUDA ABI and exact fixture retirement |
| 236 | `3d9866dfd6cf` | docs(audit): distinguish 23 model-dependent B4 scopes |
| 237 | `a39ca967ea20` | fix(gpu): qualify grouped transport from independent fresh state and seal KV cache views |
| 238 | `7362699b7023` | fix(serve): preserve prepublication cancellations and bound decode context |
| 239 | `5229e53d7b46` | docs(review): record architectural and correctness findings and qualification gates |
| 240 | `83a317476793` | refactor(gpu): one typed closed CUDA ABI for metadata and bound handles |
| 241 | `095bf8a4b97d` | refactor(core): collocate Qwen3.5 recipes without changing type identity |
| 242 | `2a47d25181f1` | feat(cli): concise run placement aliases and observed-device inventory |
| 243 | `d9c5872c73ee` | docs: scope ABI family and run UX refactor acceptance |
| 244 | `5399ab93929e` | refactor(cli): use device selection without redundant on alias |
| 245 | `d84853337bbd` | feat(cli): resolve Linux GPU device nodes to verified physical PCI identities |
