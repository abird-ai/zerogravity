# Draft Patches — volume 19

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-repack-transport-followup/REPACK-TRANSPORT-FOLLOWUP.patch

Original UTF-8 bytes: 35207; SHA256 `0da1f69b1bb31ff9b830e1d3a0d3d9d0a6fb41b925fc68e2b1c924bda92c36e6`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- before-repack-transport/native_window.cpp
+++ repack-transport/native_window.cpp
@@ -4,6 +4,8 @@
 #include "../numerics-and-context-long-context-audit-20261020/native_raw.cpp"
 #undef main
 #include <map>
+#include <cstdint>
+#include <dlfcn.h>
 
 constexpr size_t BULK_CAP = 112 * 1024 * 1024;
 constexpr size_t AR1_CAP = 112 * 1024 * 1024;
@@ -119,11 +121,90 @@
     {"ssm_conv1d.weight",GGML_TYPE_F32,4,6144}, {"ssm_dt.bias",GGML_TYPE_F32,16,1},
     {"ssm_norm.weight",GGML_TYPE_F32,128,1}, {"ssm_out.weight",GGML_TYPE_Q8_0,2048,2048},
 };
-struct Learned { const ggml_tensor *tensor; size_t bytes,index; void *data; ggml_backend_buffer_t buffer; };
+constexpr const char *PINNED_CPU_DSO="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so";
+static ggml_backend_buffer_type_t cpu_raw_type=nullptr,cpu_mapped_type=nullptr,cpu_repack_type=nullptr;
+static void bind_learned_buffer_types() {
+    const auto reg=ggml_backend_reg_by_name("CPU");
+    require(reg && std::strcmp(ggml_backend_reg_name(reg),"CPU")==0
+        && ggml_backend_reg_dev_count(reg)==1,"closed original CPU registry");
+    const auto dev=ggml_backend_reg_dev_get(reg,0);
+    require(dev && ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_CPU,"closed original CPU device");
+    cpu_raw_type=ggml_backend_cpu_buffer_type();
+    require(cpu_raw_type && ggml_backend_dev_buffer_type(dev)==cpu_raw_type
+        && std::strcmp(ggml_backend_buft_name(cpu_raw_type),"CPU")==0
+        && ggml_backend_buft_is_host(cpu_raw_type),"closed original CPU raw type");
+    // Public descriptor-only type query; never aliases a model allocation.
+    alignas(64) unsigned char local[64]={};
+    const auto mapped=ggml_backend_cpu_buffer_from_ptr(local,sizeof(local));
+    require(mapped,"public mapped-buffer type query");
+    cpu_mapped_type=ggml_backend_buffer_get_type(mapped);
+    const bool mapped_ok=cpu_mapped_type && std::strcmp(ggml_backend_buft_name(cpu_mapped_type),"CPU_Mapped")==0
+        && ggml_backend_buft_is_host(cpu_mapped_type);
+    ggml_backend_buffer_free(mapped);
+    require(mapped_ok,"closed original CPU mapped type");
+    void *address=ggml_backend_reg_get_proc_address(reg,"ggml_backend_dev_get_extra_bufts");
+    Dl_info info{};
+    require(address && dladdr(address,&info)!=0 && info.dli_fname
+        && fs::canonical(info.dli_fname)==fs::canonical(PINNED_CPU_DSO),"extra-buffer public callback pinned Haswell DSO");
+    const auto extra=reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(address)(dev);
+    require(extra,"public null-terminated extra-buffer list");
+    size_t count=0;
+    while(count<16 && extra[count]) {
+        const auto type=extra[count++];
+        require(ggml_backend_buft_name(type),"named original extra-buffer type");
+        if(std::strcmp(ggml_backend_buft_name(type),"CPU_REPACK")==0) {
+            require(!cpu_repack_type && ggml_backend_buft_get_device(type)==dev,"unique exact original CPU_REPACK type");
+            cpu_repack_type=type;
+        }
+    }
+    require(count<16 && cpu_repack_type && !ggml_backend_buft_is_host(cpu_repack_type),"closed bounded CPU_REPACK type list");
+}
+struct PhysicalRead {
+    ggml_backend_buffer_type_t type;
+    void *base;
+    size_t size,offset;
+    bool opaque;
+};
+static PhysicalRead physical_read(const ggml_tensor *t,size_t bytes) {
+    require(t && !t->view_src && t->data && t->buffer && bytes==ggml_nbytes(t),"nonview learned physical owner");
+    const auto type=ggml_backend_buffer_get_type(t->buffer);
+    const bool opaque=type==cpu_repack_type;
+    require(type && (type==cpu_raw_type || type==cpu_mapped_type || opaque),"unknown learned buffer type refuses");
+    require(!opaque || t->type==GGML_TYPE_Q4_0 || t->type==GGML_TYPE_Q4_1 || t->type==GGML_TYPE_Q8_0,"closed opaque learned quantized kinds");
+    require(std::strcmp(ggml_backend_buffer_name(t->buffer),opaque ? "CPU_REPACK" : type==cpu_raw_type ? "CPU" : "CPU_Mapped")==0
+        && ggml_backend_buffer_get_usage(t->buffer)==GGML_BACKEND_BUFFER_USAGE_WEIGHTS,"exact learned buffer name/weight usage");
+    void *base=ggml_backend_buffer_get_base(t->buffer);
+    const size_t size=ggml_backend_buffer_get_size(t->buffer);
+    const auto first=reinterpret_cast<uintptr_t>(base),data=reinterpret_cast<uintptr_t>(t->data);
+    require(base && size && size<=1296764000ULL && size<=std::numeric_limits<uintptr_t>::max()-first
+        && data>=first && data-first<=size && bytes<=size-size_t(data-first),"learned complete span within original public CPU allocation");
+    return {type,base,size,size_t(data-first),opaque};
+}
+static bool same_physical_read(const PhysicalRead &a,const PhysicalRead &b) {
+    return a.type==b.type && a.base==b.base && a.size==b.size && a.offset==b.offset && a.opaque==b.opaque;
+}
+static void read_learned(const ggml_tensor *t,const PhysicalRead &owner,size_t bytes,size_t at,void *out,size_t length) {
+    check_time();
+    require(length && length<=LEARNED_CHUNK && at<=bytes && length<=bytes-at
+        && same_physical_read(owner,physical_read(t,bytes)),"same-owner bounded learned read");
+    // CPU_REPACK clears get_tensor while retaining the original CPU allocation.
+    // These bytes are opaque, never interpreted as the advertised GGUF codec.
+    if(owner.opaque)std::memcpy(out,static_cast<const unsigned char*>(t->data)+at,length);
+    else ggml_backend_tensor_get(t,out,at,length);
+}
+struct Learned {
+    const ggml_tensor *tensor;
+    size_t bytes,index;
+    void *data;
+    ggml_backend_buffer_t buffer;
+    PhysicalRead owner;
+    bool bulk_seen;
+    uint64_t ar_seen;
+};
 static std::map<std::string,Learned> learned;
 static fs::path learned_folder;
 static std::ofstream learned_records;
-static size_t learned_bytes=0;
+static size_t learned_bytes=0,learned_opaque_count=0,learned_verified_boundaries=0;
 static void capture_learned(const ggml_tensor *t) {
     const int layer=t ? tensor_layer(t->name) : -1;
     if(layer<0)return;
@@ -136,23 +217,36 @@
     const size_t bytes=ggml_nbytes(t);
     require(bytes==ggml_row_size(role->type,role->columns)*size_t(role->rows)
         && bytes<=8*1024*1024,"exact original packed learned bytes");
+    const auto owner=physical_read(t,bytes);
+    require(provider_mode=="bulk64" || (provider_mode=="ar1" && call_index>=0 && call_index<64),"learned actual call binding");
     const auto found=learned.find(name);
     if(found!=learned.end()) {
         require(found->second.tensor==t && found->second.bytes==bytes && found->second.data==t->data
-            && found->second.buffer==t->buffer,"all contexts share same immutable learned owner");return;
+            && found->second.buffer==t->buffer && same_physical_read(found->second.owner,owner),"all contexts share same immutable learned owner");
+        if(provider_mode=="bulk64")found->second.bulk_seen=true;
+        else found->second.ar_seen|=uint64_t(1)<<call_index;
+        return;
     }
     require(provider_mode=="bulk64" && learned.size()<28 && learned_bytes<=LEARNED_CAP
         && bytes<=LEARNED_CAP-learned_bytes,"bounded complete shared learned operands");
     const size_t index=learned.size()+1;
-    learned.emplace(name,Learned{t,bytes,index,t->data,t->buffer});learned_bytes+=bytes;
+    learned.emplace(name,Learned{t,bytes,index,t->data,t->buffer,owner,true,0});learned_bytes+=bytes;
+    if(owner.opaque)learned_opaque_count++;
     learned_records<<"{\"name\":"<<quote(name)<<",\"type\":"<<int(t->type)
         <<",\"type_name\":"<<quote(ggml_type_name(t->type))<<",\"bytes\":"<<bytes<<",\"ne\":[";
     for(int i=0;i<4;i++)learned_records<<(i?",":"")<<t->ne[i];
     learned_records<<"],\"nb\":[";for(int i=0;i<4;i++)learned_records<<(i?",":"")<<t->nb[i];
-    learned_records<<"],\"chunks\":[";
+    learned_records<<"],\"encoding\":"<<quote(owner.opaque ? "CPU_REPACK_UNDECODED" : "GGUF_RAW")
+        <<",\"buffer_name\":"<<quote(ggml_backend_buffer_name(t->buffer))
+        <<",\"physical_owner\":{\"tensor\":"<<reinterpret_cast<uintptr_t>(t)
+        <<",\"buffer\":"<<reinterpret_cast<uintptr_t>(t->buffer)<<",\"buffer_type\":"<<reinterpret_cast<uintptr_t>(owner.type)
+        <<",\"base\":"<<reinterpret_cast<uintptr_t>(owner.base)<<",\"allocation_bytes\":"<<owner.size
+        <<",\"data\":"<<reinterpret_cast<uintptr_t>(t->data)<<",\"offset\":"<<owner.offset
+        <<"},\"source_codec_equality\":"<<quote(owner.opaque ? "UNQUALIFIED_OPAQUE_REPACK" : "REQUIRES_INDEPENDENT_GGUF_BYTE_VALIDATION")
+        <<",\"chunks\":[";
     for(size_t at=0;at<bytes;at+=LEARNED_CHUNK) {
         const size_t length=std::min(LEARNED_CHUNK,bytes-at);
-        std::vector<unsigned char> raw(length);ggml_backend_tensor_get(t,raw.data(),at,length);
+        std::vector<unsigned char> raw(length);read_learned(t,owner,bytes,at,raw.data(),length);
         const std::string file="learned-"+std::to_string(index)+"-"+std::to_string(at/LEARNED_CHUNK)+".bin";
         write_bytes(learned_folder/file,raw.data(),raw.size());
         if(at)learned_records<<',';
@@ -161,19 +255,21 @@
     learned_records<<"]}\n";learned_records.flush();
     require(bool(learned_records) && learned_records.tellp()>=0 && uint64_t(learned_records.tellp())<=128*1024,"learned metadata cap/write");
 }
-static void verify_learned_unchanged() {
+static void verify_learned_unchanged(bool final) {
     require(learned.size()==28,"complete original2B first-two-layer learned closure");
     for(const auto &entry:learned) {
         const auto &r=entry.second;
-        require(r.tensor->data==r.data && r.tensor->buffer==r.buffer,"retained learned physical owner");
+        require(r.tensor->data==r.data && r.tensor->buffer==r.buffer && r.bulk_seen
+            && (!final || r.ar_seen==std::numeric_limits<uint64_t>::max()),"all28 retained owners in bulk and every AR1 call");
         for(size_t at=0;at<r.bytes;at+=LEARNED_CHUNK) {
             const size_t length=std::min(LEARNED_CHUNK,r.bytes-at);
-            std::vector<unsigned char> actual(length);ggml_backend_tensor_get(r.tensor,actual.data(),at,length);
+            std::vector<unsigned char> actual(length);read_learned(r.tensor,r.owner,r.bytes,at,actual.data(),length);
             const std::string file="learned-"+std::to_string(r.index)+"-"+std::to_string(at/LEARNED_CHUNK)+".bin";
             const auto before=read_bounded(learned_folder/file,LEARNED_CHUNK);
             require(before.size()==length && actual==before,"original learned operand changed during actual computations");
         }
     }
+    learned_verified_boundaries++;
 }
 
 static bool window_observer(ggml_tensor *t,bool ask,void*) {
@@ -285,7 +381,7 @@
         window_start=int(start);window_count=64;
         const int threads=integer(argv[4],8),seconds=integer(argv[5],7200);
         deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
-        const uint32_t endian=1;require(*reinterpret_cast<const unsigned char*>(&endian)==1 && sizeof(float)==4,"little-endian F32 host");
+        const uint32_t endian=1;require(*reinterpret_cast<const unsigned char*>(&endian)==1 && sizeof(float)==4 && sizeof(uintptr_t)==8,"little-endian F32/64-bit-pointer host");
         const auto raw=read_bounded(argv[2],8192*4);require(raw.size()==8192*4,"exact common8192 teacher input");
         std::vector<llama_token> ids(raw.size()/4);
         for(size_t i=0;i<ids.size();i++){uint32_t token;std::memcpy(&token,raw.data()+i*4,4);require(token<248320,"source token bound");ids[i]=llama_token(token);}
@@ -294,6 +390,7 @@
         llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
         std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
         require(bool(model) && llama_model_n_layer(model.get())==24 && llama_model_n_embd(model.get())==2048 && llama_vocab_n_tokens(llama_model_get_vocab(model.get()))==248320,"original2B geometry");
+        bind_learned_buffer_types();
         auto params=llama_context_default_params();params.n_ctx=8320;params.n_batch=64;params.n_ubatch=64;
         params.n_threads=threads;params.n_threads_batch=threads;params.type_k=GGML_TYPE_F32;params.type_v=GGML_TYPE_F32;
         params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;params.offload_kqv=false;params.op_offload=false;params.no_perf=true;params.cb_eval=nullptr;
@@ -323,6 +420,7 @@
         begin_observation(base,"bulk64");enabled=true;call_index=0;call_count=window_count;occurrences.clear();tensor_ids.clear();
         decode(observed_bulk.get(),ids,window_start,window_count,false);enabled=false;require_current_coverage();
         const size_t bulk_bytes=end_observation();require(state(observed_bulk.get())==after_bulk,"observer changed unchanged-width bulk complete state");
+        verify_learned_unchanged(false);
         guard_next(bulk.get(),observed_bulk.get(),ids,window_start+window_count,base/"bulk.next-logits.f32le");
         serial_params.cb_eval=window_observer;Context observed_ar(llama_init_from_model(model.get(),serial_params),llama_free);require(bool(observed_ar),"observed AR1 context");
         require(llama_state_set_data(observed_ar.get(),before_serial.data(),before_serial.size())==before_serial.size(),"same-source AR1 prior restore");
@@ -330,7 +428,8 @@
         for(int at=0;at<window_count;at++){call_index=at;occurrences.clear();tensor_ids.clear();decode(observed_ar.get(),ids,window_start+at,1,false);require_current_coverage();}
         enabled=false;const size_t ar_bytes=end_observation();require(state(observed_ar.get())==after_serial,"observer changed AR1 complete state");
         guard_next(serial.get(),observed_ar.get(),ids,window_start+window_count,base/"ar1.next-logits.f32le");
-        verify_learned_unchanged();learned_records.close();
+        verify_learned_unchanged(true);learned_records.close();
+        require(learned_verified_boundaries==2,"complete bulk and AR1 learned readset checks");
         size_t artifact_bytes=0;
         for(const auto &entry:fs::recursive_directory_iterator(base)) {
             require(!entry.is_symlink(),"no capture artifact symlink");
@@ -343,6 +442,8 @@
         std::ofstream done(base/"complete.json");require(bool(done),"window complete record open");
         done<<"{\"status\":\"TRANSPARENT_WINDOW_CAPTURE_NOT_CAUSE_QUALIFICATION\",\"model_path\":"<<quote(fs::canonical(argv[1]).string())
             <<",\"hidden_width\":2048,\"selected_layers\":[0,1],\"learned_operand_count\":28,\"learned_operand_bytes\":"<<learned_bytes<<",\"learned_unchanged_after_compute\":true,\"window_start\":"<<window_start<<",\"window_tokens\":"<<window_count<<",\"original_case_cut\":8192,\"threads\":"<<threads
+            <<",\"learned_verified_boundaries\":2,\"all28_owners_seen_bulk_and_all64_AR1_calls\":true,\"learned_opaque_repack_count\":"<<learned_opaque_count
+            <<",\"opaque_repack_source_codec_equality\":\"UNQUALIFIED\",\"cpu_extra_buffer_callback_dso\":"<<quote(PINNED_CPU_DSO)
             <<",\"bulk_actual_ubatch\":64,\"ar1_actual_ubatch\":1,\"common_prior_exact\":true,\"bulk_instrumentation_complete_state_exact\":true,\"ar1_instrumentation_complete_state_exact\":true"
             <<",\"both_next_AR1_full248320_and_whole_state_exact\":true,\"post_state_bulk_vs_ar1_equal\":"<<(after_bulk==after_serial?"true":"false")
             <<",\"bulk_payload_bytes\":"<<bulk_bytes<<",\"ar1_payload_bytes\":"<<ar_bytes
--- before-repack-transport/compare_window.py
+++ repack-transport/compare_window.py
@@ -13,6 +13,7 @@
 
 MODE_CAPS = {"bulk64": 112 * 1024 * 1024, "ar1": 112 * 1024 * 1024}
 FILE_DIGESTS = {}
+CPU_DSO = "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so"
 
 
 def require(ok, message):
@@ -172,6 +173,27 @@
     return {"bytes": before.st_size, "sha256": h.hexdigest()}
 
 
+def physical_owner(row, model_bytes):
+    encoding, name = row["encoding"], row["buffer_name"]
+    require((encoding == "GGUF_RAW" and name in ["CPU", "CPU_Mapped"]
+             and row["source_codec_equality"] == "REQUIRES_INDEPENDENT_GGUF_BYTE_VALIDATION")
+            or (encoding == "CPU_REPACK_UNDECODED" and name == "CPU_REPACK" and row["type"] in [2, 3, 8]
+                and row["source_codec_equality"] == "UNQUALIFIED_OPAQUE_REPACK"),
+            "closed raw/opaque buffer encoding; unknown buffer refuses")
+    owner = row["physical_owner"]
+    require(type(owner) is dict and set(owner) == {"tensor", "buffer", "buffer_type", "base", "allocation_bytes", "data", "offset"},
+            "complete actual physical owner descriptor")
+    require(all(type(value) is int and 0 <= value < 2**64 for value in owner.values())
+            and all(owner[key] > 0 for key in ["tensor", "buffer", "buffer_type", "base", "allocation_bytes", "data"]),
+            "bounded nonnull original physical owner identity")
+    require(owner["allocation_bytes"] <= model_bytes and owner["base"] + owner["allocation_bytes"] < 2**64
+            and owner["offset"] <= owner["allocation_bytes"]
+            and row["bytes"] <= owner["allocation_bytes"] - owner["offset"]
+            and owner["data"] == owner["base"] + owner["offset"],
+            "whole learned span inside original public CPU allocation")
+    return owner, encoding == "CPU_REPACK_UNDECODED"
+
+
 def verify_learned(capture, done):
     here = Path(__file__).resolve().parent
     descriptors = jread(here / "learned-source-headers.json")
@@ -188,15 +210,38 @@
             bounded(capture / "learned-operands/records.jsonl", 128 * 1024).splitlines()]
     require(len(rows) == 28, "all actual learned operand descriptors")
     seen, files, total, hashes = set(), set(), 0, {}
+    owners, tensors, intervals, opaque_count, opaque_bytes = {}, set(), {}, 0, 0
     with os.fdopen(os.open(model, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
         for row in rows:
             name = row["name"]
             require(name in expected and name not in seen, "unique original learned role")
             seen.add(name)
             original = expected[name]
-            require(row["type"] == original["type"] and row["ne"] == original["ne"]
+            require(type(row["type"]) is int and row["type"] == original["type"]
+                    and type(row["ne"]) is list and all(type(value) is int for value in row["ne"])
+                    and row["ne"] == original["ne"] and type(row["bytes"]) is int
                     and row["bytes"] == original["bytes"] and 0 < row["bytes"] <= 8 * 1024**2,
                     "exact original learned dtype/shape/byte geometry")
+            block, unit, kind = {0: (1, 4, "f32"), 2: (32, 18, "q4_0"),
+                                 3: (32, 20, "q4_1"), 8: (32, 34, "q8_0")}[row["type"]]
+            stride = row["ne"][0] // block * unit
+            require(row["ne"][0] % block == 0 and row["type_name"] == kind
+                    and type(row["nb"]) is list and all(type(value) is int for value in row["nb"])
+                    and row["nb"] == [unit, stride, stride * row["ne"][1], stride * row["ne"][1]],
+                    "original type-labelled contiguous metadata; opaque encoding stays undecoded")
+            owner, opaque = physical_owner(row, descriptors["model_bytes"])
+            require(owner["tensor"] not in tensors, "distinct original learned tensor identities")
+            tensors.add(owner["tensor"])
+            allocation = (owner["buffer_type"], owner["base"], owner["allocation_bytes"], row["buffer_name"], row["encoding"])
+            require(owner["buffer"] not in owners or owners[owner["buffer"]] == allocation,
+                    "one consistent original physical allocation descriptor per buffer")
+            owners[owner["buffer"]] = allocation
+            ranges = intervals.setdefault(owner["buffer"], [])
+            require(all(owner["offset"] + row["bytes"] <= lo or hi <= owner["offset"] for lo, hi in ranges),
+                    "distinct learned roles do not overlap in their physical owner")
+            ranges.append((owner["offset"], owner["offset"] + row["bytes"]))
+            opaque_count += int(opaque)
+            opaque_bytes += row["bytes"] if opaque else 0
             at, content = 0, hashlib.sha256()
             for chunk in row["chunks"]:
                 require(re.fullmatch(r"learned-[1-9][0-9]?-[0-9]+\.bin", chunk["file"])
@@ -206,8 +251,9 @@
                 files.add(chunk["file"])
                 raw = bounded(capture / "learned-operands" / chunk["file"], 1024**2)
                 require(len(raw) == chunk["bytes"], "exact raw source chunk")
-                source.seek(original["absolute_offset"] + at)
-                require(source.read(len(raw)) == raw, "actual physical learned operand equals approved GGUF bytes")
+                if not opaque:
+                    source.seek(original["absolute_offset"] + at)
+                    require(source.read(len(raw)) == raw, "actual raw physical learned operand equals approved GGUF bytes")
                 if row["type"] == 0:
                     require(all(bits & 0x7f800000 != 0x7f800000 for (bits,) in struct.iter_unpack("<I", raw)),
                             "finite actual F32 learned source")
@@ -215,11 +261,25 @@
                 at += len(raw)
             require(at == row["bytes"], "complete actual packed learned operand")
             total += at
-            hashes[name] = {"bytes": at, "sha256": content.hexdigest(), "type": row["type"], "ne": row["ne"]}
+            hashes[name] = {"bytes": at, "sha256": content.hexdigest(), "type": row["type"], "ne": row["ne"],
+                            "encoding": row["encoding"], "physical_owner": owner,
+                            "approved_GGUF_absolute_offset": original["absolute_offset"],
+                            "source_codec_equality": "UNQUALIFIED_OPAQUE_REPACK" if opaque else "EXACT_RAW_GGUF_BYTES"}
     require(seen == set(expected) and total == descriptors["selected_raw_bytes"] == done["learned_operand_bytes"]
             and total <= 80 * 1024**2, "all28 actual raw learned operands accounted")
+    require(type(done["learned_opaque_repack_count"]) is int and done["learned_opaque_repack_count"] == opaque_count
+            and type(done["learned_verified_boundaries"]) is int and done["learned_verified_boundaries"] == 2
+            and done["all28_owners_seen_bulk_and_all64_AR1_calls"] is True
+            and done["opaque_repack_source_codec_equality"] == "UNQUALIFIED"
+            and done["cpu_extra_buffer_callback_dso"] == CPU_DSO,
+            "complete owner/readset invariance at both real computation boundaries")
     require(digest_file(model) == expected_source, "original GGUF unchanged after actual readset validation")
-    return {"all28_equal_original_source_bytes": True, "bytes": total, "raw_operands": hashes,
+    return {"all28_equal_original_source_bytes": None if opaque_count else True,
+            "source_codec_equality": "UNQUALIFIED_OPAQUE_REPACK_PRESENT" if opaque_count else "EXACT_RAW_GGUF_BYTES",
+            "all28_same_owner_and_bytes_after_bulk_and_AR1": True,
+            "ordinary_raw_operand_count": 28 - opaque_count, "ordinary_raw_bytes": total - opaque_bytes,
+            "opaque_repack_operand_count": opaque_count, "opaque_repack_bytes": opaque_bytes,
+            "approved_GGUF_model": expected_source, "bytes": total, "raw_operands": hashes,
             "weight_dequantization_or_arithmetic_executed": False}
 
 
--- before-repack-transport/source-pins.json
+++ repack-transport/source-pins.json
@@ -1,5 +1,5 @@
 {
-  "scope": "source-only first-two-layer original2B bulk64/AR1 causal observation preparation; no model/math execution",
+  "scope": "source-only original2B causal observer including public CPU_REPACK_UNDECODED physical-read transport; opaque codec equivalence remains UNQUALIFIED",
   "model": {
     "path": "/home/pvl/spaces/abird/src/zerogravity/target/models/qwen35-2b/model.gguf",
     "bytes": 1296764000,
@@ -161,6 +161,26 @@
       "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/learned-source-headers.json",
       "bytes": 6898,
       "sha256": "5a0e2c8ee280d08c06252b25ddcc38fab300ccbaf56186104b3564ca864b9615"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/repack.cpp",
+      "bytes": 213604,
+      "sha256": "5fa9f24cc2631609f40fa47a64078f77dccbe9f327711af91af671cd29b8b87d"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.cpp",
+      "bytes": 25005,
+      "sha256": "ccf382d3fbf126b5ddb5850a0a970656fb24e00587da4003419439277c91cd01"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-impl.h",
+      "bytes": 25821,
+      "sha256": "43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/src/llama-model.cpp",
+      "bytes": 163123,
+      "sha256": "c29d7f29c3e760f90855755748edc56fcdf62f0628b0c572f06ffb4c377db5f5"
     }
   ]
 }
--- before-repack-transport/README.md
+++ repack-transport/README.md
@@ -36,15 +36,31 @@
   ne[67,6144,1,1]. Every AR1 input ne[4,6144,1,1]. Comparator byte-selects each
   actual four-tap causal slice and compares all4×6144 raw words for every row.
   It computes no convolution or expected numerical result.
-- All28 original learned inputs for these two layers are captured as their exact
-  packed source bytes, including real Q4_0/Q4_1/Q8_0 matrix encodings and F32
-  norms/kernel/A/dt. Header-only inspection independently records actual source
-  ne/type/relative+absolute offsets: total72,582,400B, largest7,864,320B.
-  One shared source owner/tensor/data/buffer identity must remain the same for
-  every context/callback. Copy once in≤1MiB chunks; re-read/compare all bytes
-  after both observed computations. Comparator checks all28 raw operands against
-  the full-hash-pinned original GGUF payload, not dequantized/candidate values.
-  Learned source invariance is checked at captured start/end; no per-FMA trace.
+- All28 learned inputs retain actual original tensor/type/shape/byte metadata,
+  separately labelled `GGUF_RAW` or `CPU_REPACK_UNDECODED`. Header-only source
+  geometry covers 72,582,400B, largest7,864,320B. Actual repacked bytes are NOT
+  interpreted/dequantized as the metadata's GGUF codec. Ordinary CPU/CPU_Mapped
+  payloads still must equal independently pinned original GGUF bytes exactly;
+  opaque repack source-codec equality remains UNQUALIFIED even after a passing
+  diagnostic. Full model hash/header binding is preserved for both transports.
+- Exact CPU_REPACK type pointer is selected through the public CPU registry's
+  documented extra-bufts callback, dladdr-bound to the pinned Haswell DSO. Known
+  raw CPU and CPU_Mapped singleton types are also checked. CPU_Mapped's private
+  singleton is obtained through a public temporary descriptor over an aligned
+  local64B array, immediately freed; it never aliases a model allocation.
+  Unknown buffers refuse before any data access. No private iface access.
+- The pinned source allocates CPU_REPACK using an ordinary CPU allocation,
+  retaining get_base/free/size while replacing set_tensor and clearing get_tensor
+  (repack.cpp5168–5180). The original public tensor_get would call NULL. Opaque
+  reads therefore use original public buffer base+size, nonview data uintptr
+  and complete ggml_nbytes bounds, then <=1MiB memcpy. No native repack setting,
+  operation, arithmetic or callback is changed. Raw CPU uses public tensor_get.
+- Each role must retain the same tensor/data/buffer/type/base/size/offset through
+  bulk and all64 AR1 calls. Copy once in <=1MiB chunks; re-read/compare all28
+  complete payloads after bulk and again after AR1. These are actual captured
+  start/boundary readsets, not per-FMA observations. Numerical transparency
+  against each own plain context remains mandatory. Physical-owner addresses
+  are diagnostic identity within this one process, not cross-run identities.
 - No attention3/cache-capacity read is selected in this two-layer slice. F32
   copier retains exact ne/nb/checked physical span, reads only occupied inner
   runs and refuses spans>16MiB. No private strided-cache exception/cap increase.
@@ -86,18 +102,19 @@
 TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
 TASK_GLIBC=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
 python3 "$TASK_2B_PROBE/verify_preparation.py"
+python3 "$TASK_2B_PROBE/test_transport_validation.py"
 c++ -std=c++17 -O2 -Wall -Wextra -Werror \
   -I "$TASK_LLAMA_SRC/include" -I "$TASK_LLAMA_SRC/ggml/include" \
   "$TASK_2B_PROBE/native_window.cpp" -L "$TASK_LLAMA_LIB" -L "$TASK_GLIBC" \
   -Wl,-rpath,"$TASK_LLAMA_LIB:$TASK_GLIBC" \
   -Wl,--dynamic-linker,"$TASK_GLIBC/ld-linux-x86-64.so.2" \
-  -lllama -lggml -lggml-base -lm -o "$TASK_2B_PROBE/native_window"
+  -lllama -lggml -lggml-base -lm -ldl -o "$TASK_2B_PROBE/native_window"
 # Only ROOT under existing bounded supervisor; fresh output and binary hash.
 "$TASK_2B_PROBE/native_window" target/models/qwen35-2b/model.gguf \
   .agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le \
-  0 8 900 "$TASK_2B_PROBE/root-first64"
-python3 "$TASK_2B_PROBE/compare_window.py" --capture "$TASK_2B_PROBE/root-first64" \
-  --output "$TASK_2B_PROBE/root-first64-comparison.json"
+  0 8 900 "$TASK_2B_PROBE/root-first64-repack-transport"
+python3 "$TASK_2B_PROBE/compare_window.py" --capture "$TASK_2B_PROBE/root-first64-repack-transport" \
+  --output "$TASK_2B_PROBE/root-first64-repack-transport-comparison.json"
 python3 "$TASK_2B_PROBE/verify_preparation.py"
 ```
 
@@ -105,3 +122,16 @@
 means a complete diagnostic was emitted, even if result rows differ. Source or
 internal operator dispatch, then source-correct shared-engine implementation and
 full independent42-state/logit long cases remain separate required work.
+
+## Preserved failure and additive transport follow-up
+
+The first actual ROOT attempt exited SIGSEGV(-11), before complete observation.
+Root GDB/source isolated the NULL CPU_REPACK get_tensor callback; that attempt is
+UNQUALIFIED. Original frozen-prepared/, compiled binary, partial root-first64/
+and failure/GDB logs are retained unchanged. frozen-before-repack-transport/
+retains the exact live pre-repair sources (including earlier row-binding fix).
+The additive transport patch must be reviewed and freshly compiled by ROOT.
+No compilation, native call, operator/model/container/GPU execution by author.
+Prepared six descriptor-validation fixture methods have not been executed here.
+Passing transport/transparency establishes diagnostic read/owner scope only;
+original repack codec equality and first numerical causal operator remain open.
--- before-repack-transport/TODO.md
+++ repack-transport/TODO.md
@@ -13,3 +13,11 @@
 - [ ] Locate first observed gap; require every actual causal input/dispatch
   before attributing an operator or implementing a source-correct fix.
 - [ ] Full long-context independent42-domain/logit acceptance remains open.
+
+- [x] Preserve first failed actual attempt and pre-repair bytes; source-isolate
+  public CPU_REPACK get_tensor=NULL; prepare closed opaque physical transport,
+  explicit codec-UNQUALIFIED boundary, owner/full-readset checks after both modes.
+- [ ] Independent additive transport review, descriptor fixtures, strict fresh
+  ROOT build and actual complete unchanged-window capture.
+- [ ] Any opaque repack-to-original codec proof requires a separate reviewed
+  unchanged original repack control; this diagnostic does not claim it.
--- /dev/null
+++ repack-transport/test_transport_validation.py
@@ -0,0 +1,70 @@
+#!/usr/bin/env python3
+"""Source-free descriptor refusals only; ROOT runs, no native/model calls."""
+import copy
+import importlib.util
+from pathlib import Path
+import unittest
+
+SPEC = importlib.util.spec_from_file_location("window_transport", Path(__file__).with_name("compare_window.py"))
+MODULE = importlib.util.module_from_spec(SPEC)
+SPEC.loader.exec_module(MODULE)
+
+
+def raw_row():
+    return {"encoding": "GGUF_RAW", "buffer_name": "CPU_Mapped", "type": 2, "bytes": 32,
+            "source_codec_equality": "REQUIRES_INDEPENDENT_GGUF_BYTE_VALIDATION",
+            "physical_owner": {"tensor": 4096, "buffer": 8192, "buffer_type": 12288,
+                               "base": 16384, "allocation_bytes": 128, "data": 16416, "offset": 32}}
+
+
+class TransportValidation(unittest.TestCase):
+    def test_actual_raw_geometry(self):
+        row = raw_row()
+        owner, opaque = MODULE.physical_owner(row, 1024)
+        self.assertIs(owner, row["physical_owner"])
+        self.assertFalse(opaque)
+
+    def test_opaque_stays_unqualified(self):
+        row = raw_row()
+        row.update(encoding="CPU_REPACK_UNDECODED", buffer_name="CPU_REPACK",
+                   source_codec_equality="UNQUALIFIED_OPAQUE_REPACK")
+        self.assertTrue(MODULE.physical_owner(row, 1024)[1])
+        row["source_codec_equality"] = "EXACT_RAW_GGUF_BYTES"
+        with self.assertRaises(ValueError):
+            MODULE.physical_owner(row, 1024)
+
+    def test_unknown_or_mislabelled_buffer_refuses(self):
+        for key, value in [("buffer_name", "CUDA"), ("encoding", "CPU_REPACK"),
+                           ("buffer_name", "CPU_REPACK")]:
+            with self.subTest(key=key, value=value):
+                row = raw_row()
+                row[key] = value
+                with self.assertRaises(ValueError):
+                    MODULE.physical_owner(row, 1024)
+
+    def test_entire_span_and_integer_bounds(self):
+        for changed in [{"offset": 120, "data": 16504}, {"base": 2**64 - 64},
+                        {"offset": True}, {"allocation_bytes": 2048},
+                        {"data": 16417}, {"buffer": 0}]:
+            row = raw_row()
+            row["physical_owner"].update(changed)
+            with self.subTest(changed=changed), self.assertRaises(ValueError):
+                MODULE.physical_owner(row, 1024)
+
+    def test_f32_is_not_opaque_repack(self):
+        row = raw_row()
+        row.update(encoding="CPU_REPACK_UNDECODED", buffer_name="CPU_REPACK", type=0,
+                   source_codec_equality="UNQUALIFIED_OPAQUE_REPACK")
+        with self.assertRaises(ValueError):
+            MODULE.physical_owner(row, 1024)
+
+    def test_owner_fields_complete(self):
+        for key in raw_row()["physical_owner"]:
+            row = copy.deepcopy(raw_row())
+            del row["physical_owner"][key]
+            with self.subTest(key=key), self.assertRaises(ValueError):
+                MODULE.physical_owner(row, 1024)
+
+
+if __name__ == "__main__":
+    unittest.main()

````

## Source: .agents/runs/numerics-and-context-av-replay-20261020/FOLLOWUP_CLEANUP.patch

Original UTF-8 bytes: 11426; SHA256 `5f25361fc30955df3dc82ceef14c3e4e9de57c20da44dc090c02ee3f87b3c31b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/run_replay.py
+++ b/run_replay.py
@@ -12,6 +12,73 @@
 
 LOADER = '/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib/ld-linux-x86-64.so.2'
 LIB = '/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib'
+
+
+def group_exists(pgid):
+    try:
+        os.killpg(pgid, 0)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def signal_group(pgid, sig):
+    try:
+        os.killpg(pgid, sig)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def leader_exit_without_reaping(process, seconds):
+    # Keep the owned leader PID reserved until group cleanup begins.
+    deadline = time.monotonic() + seconds
+    while time.monotonic() < deadline:
+        observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+        if observed is not None:
+            return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+        time.sleep(0.05)
+    raise TimeoutError('owned leader exit deadline')
+
+
+def cleanup_owned_group(process, receipt):
+    pgid = process.pid  # start_new_session=True; never an inherited/shared group.
+    cleanup = receipt['cleanup'] = {'pgid': pgid, 'term_sent': False, 'kill_sent': False,
+                                    'leader_reaped': False, 'group_absent': False}
+    cleanup['term_sent'] = signal_group(pgid, signal.SIGTERM)
+    try:
+        process.wait(timeout=3)
+    except subprocess.TimeoutExpired:
+        cleanup['kill_sent'] = signal_group(pgid, signal.SIGKILL)
+        process.wait(timeout=5)
+    cleanup['leader_reaped'] = True
+    # A reaped leader does not establish that its owned descendants retired.
+    if group_exists(pgid):
+        cleanup['kill_sent'] = signal_group(pgid, signal.SIGKILL) or cleanup['kill_sent']
+    deadline = time.monotonic() + 5
+    while group_exists(pgid) and time.monotonic() < deadline:
+        time.sleep(0.05)
+    cleanup['group_absent'] = not group_exists(pgid)
+    require(cleanup['group_absent'], 'owned process group remains after cleanup')
+
+
+def finish_pipe(process, log, log_bytes, receipt):
+    # Group absence alone does not prove that no writer retained this pipe.
+    os.set_blocking(process.stdout.fileno(), False)
+    deadline = time.monotonic() + 1
+    while time.monotonic() < deadline:
+        try:
+            raw = os.read(process.stdout.fileno(), 65536)
+        except BlockingIOError:
+            time.sleep(0.02)
+            continue
+        if not raw:
+            receipt['cleanup']['pipe_eof'] = True
+            return
+        log_bytes += len(raw)
+        require(log_bytes <= 8 * 1024 * 1024, 'bounded cleanup child log')
+        log.write(raw)
+    raise TimeoutError('owned stdout pipe EOF deadline')
 
 
 def main():
@@ -45,13 +112,16 @@
                'exit_code': None, 'status': 'UNQUALIFIED'}
     with a.new_receipt.open('x') as receipt_file, a.new_log.open('xb') as log:
         process = None
+        selector = None
+        log_bytes = 0
+        primary_error = None
+        cleanup_error = None
         try:
             process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, start_new_session=True)
             receipt['owned_pid'] = process.pid
             selector = selectors.DefaultSelector()
             selector.register(process.stdout, selectors.EVENT_READ)
-            log_bytes = 0
             while selector.get_map():
                 require(time.monotonic() - started < 300, 'finite wall deadline')
                 for key, _ in selector.select(0.2):
@@ -62,7 +132,7 @@
                     log_bytes += len(raw)
                     require(log_bytes <= 8 * 1024 * 1024, 'bounded child log')
                     log.write(raw)
-            receipt['exit_code'] = process.wait(timeout=5)
+            receipt['exit_code'] = leader_exit_without_reaping(process, 5)
             require(receipt['exit_code'] == 0, 'public AV process refused')
             require(digest(a.binary) == before_binary, 'binary changed during run')
             require(verify_fixtures(a.fixtures, pins) == before_fixtures, 'fixtures changed during run')
@@ -70,23 +140,40 @@
             receipt['status'] = 'DEFAULT_GATE_COMPLETED_CONTROLS_DIAGNOSTIC_VALIDATOR_REQUIRED'
         except BaseException as error:
             receipt['error'] = f'{type(error).__name__}: {error}'
-            raise
+            primary_error = error
         finally:
-            if process is not None:
-                if process.poll() is None:
+            try:
+                if process is not None:
+                    cleanup_owned_group(process, receipt)
+                    finish_pipe(process, log, log_bytes, receipt)
+            except BaseException as error:
+                cleanup_error = error
+                receipt['cleanup_error'] = f'{type(error).__name__}: {error}'
+                receipt['status'] = 'UNQUALIFIED'
+            finally:
+                closers = []
+                if selector is not None:
+                    closers.append(('selector_closed', selector.close))
+                if process is not None:
+                    closers.append(('pipe_closed', process.stdout.close))
+                for label, close in closers:
                     try:
-                        os.killpg(process.pid, signal.SIGTERM)
-                        process.wait(timeout=3)
-                    except subprocess.TimeoutExpired:
-                        os.killpg(process.pid, signal.SIGKILL)
-                        process.wait(timeout=5)
-                    except ProcessLookupError:
-                        process.wait(timeout=5)
-                receipt['exit_code'] = process.returncode
+                        close()
+                        receipt.setdefault('cleanup', {})[label] = True
+                    except BaseException as error:
+                        cleanup_error = cleanup_error or error
+                        receipt.setdefault('cleanup_errors', []).append(f'{label}: {type(error).__name__}: {error}')
+                        receipt['status'] = 'UNQUALIFIED'
+                if process is not None:
+                    receipt['exit_code'] = process.returncode
             receipt['elapsed_seconds'] = time.monotonic() - started
             json.dump(receipt, receipt_file, indent=2, allow_nan=False)
             receipt_file.write('\n')
             receipt_file.flush()
+        if primary_error is not None:
+            raise primary_error
+        if cleanup_error is not None:
+            raise cleanup_error
 
 
 if __name__ == '__main__':
--- a/validate.py
+++ b/validate.py
@@ -1,6 +1,7 @@
 #!/usr/bin/env python3
 """ROOT ONLY: authenticate public AV replay; controls remain diagnostic."""
 import argparse
+import hashlib
 import json
 from pathlib import Path
 import struct
@@ -20,6 +21,7 @@
     done = jread(a.output / 'complete.json', 128 * 1024)
     require(done['status'] == 'PUBLIC_AV_REPLAY_DEFAULT_EXACT_CONTROL_DIAGNOSTIC'
             and done['default_cases'] == done['control_cases'] == 65 and done['default_words'] == 262144
+            and type(done['threads']) is int and done['threads'] == 8
             and done['cause_qualified'] is False and done['production_math_changed'] is False,
             'complete default gate and diagnostic-only controls')
     cases = [json.loads(line) for line in bounded(a.output / 'cases.jsonl', 128 * 1024).splitlines()]
@@ -50,6 +52,8 @@
         actual_path = a.output / f'{"ref-control" if control else "default"}-{key}.actual.f32le'
         actual = bounded(actual_path, 8 * 1024 * 1024)
         require(len(actual) == len(expected) == row['words'] * 4, 'complete raw output')
+        actual_receipt = {'bytes': len(actual), 'sha256': hashlib.sha256(actual).hexdigest()}
+        require(digest(actual_path) == actual_receipt, 'unchanged complete output read')
         unequal = sum(left != right for (left,), (right,) in
                       zip(struct.iter_unpack('<I', actual), struct.iter_unpack('<I', expected)))
         require(unequal == row['actual_capture_unequal_words'], 'independent mismatch accounting')
@@ -61,12 +65,17 @@
             ref_bulk_ar = row['ref_bulk_vs_all_default_ar1_unequal_words']
             independently = 0
             for at in range(64):
-                ar = bounded(a.output / f'default-ar1-{at}.actual.f32le', 8192)
+                ar_path = a.output / f'default-ar1-{at}.actual.f32le'
+                ar = bounded(ar_path, 8192)
+                require(len(ar) == 8192
+                        and hashlib.sha256(ar).hexdigest() == receipts[ar_path.name]['sha256']
+                        and digest(ar_path) == receipts[ar_path.name],
+                        'complete unchanged previously validated default AR1 output')
                 selected = b''.join(actual[(head * 64 + at) * 1024:(head * 64 + at + 1) * 1024] for head in range(8))
                 independently += sum(left != right for (left,), (right,) in
                                      zip(struct.iter_unpack('<I', selected), struct.iter_unpack('<I', ar)))
             require(independently == ref_bulk_ar, 'independent ref bulk versus original AR1')
-        receipts[actual_path.name] = digest(actual_path)
+        receipts[actual_path.name] = actual_receipt
     require(defaults == 262144, 'all original output words covered')
     cpu_path = pins['cpu_backend']
     for phase in ['before', 'after']:
--- a/README.md
+++ b/README.md
@@ -68,9 +68,16 @@
   actual capture,300s wall,8GiB address-space limit, finite CPU/file/log caps,
   no NVIDIA exposure and only owned child-process-group cleanup. These are
   configured process ceilings, not ResourceArbiter/physical-RAM qualification.
+  The leader is inspected without reaping before cleanup. Every spawned group
+  receives bounded TERM/KILL cleanup even after leader exit; receipt requires
+  leader reaped, group absent, stdout EOF and closed selector/pipe. Cleanup has
+  separate3s TERM/5s leader-KILL/5s group/1s pipe deadlines; failures stay
+  UNQUALIFIED and retain both primary and cleanup errors.
 - Fixtures/output≤512MiB combined, native live backing≤32MiB/case, separate
   bounded8MiB log. Default and control payloads total2MiB. Python validates
   original descriptor/byte binding, full raw words and before/after pins.
+  Repeated default AR1 reads require all8192bytes and the previously validated
+  content hash; accepted completion explicitly requires8threads.
 - CPU backend maps and public control origin are observed. Internal
   microkernel dispatch remains unobserved; no automatic claim that use_ref
   equality proves every long-context failure or a production fix.
--- a/TODO.md
+++ b/TODO.md
@@ -7,6 +7,9 @@
 - [x] Source-audit original Haswell public dispatch and reduction candidates.
 - [x] Prepare bounded original public ggml AV default/ref-control replay;
   no model load or arithmetic replacement.
+- [x] Independent source review; repair owned-group cleanup after leader exit,
+  re-read output coverage/hash and explicit8thread validation. Original frozen
+  preparation retained; follow-up unexecuted.
 - [ ] Root compile/run/validator and independent source review.
 - [ ] Full long-context causal closure/production implementation remains open.
 

````

## Source: .agents/runs/numerics-and-context-av-replay-20261020/frozen-cleanup-followup/FOLLOWUP_CLEANUP.patch

Original UTF-8 bytes: 11426; SHA256 `5f25361fc30955df3dc82ceef14c3e4e9de57c20da44dc090c02ee3f87b3c31b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/run_replay.py
+++ b/run_replay.py
@@ -12,6 +12,73 @@
 
 LOADER = '/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib/ld-linux-x86-64.so.2'
 LIB = '/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib'
+
+
+def group_exists(pgid):
+    try:
+        os.killpg(pgid, 0)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def signal_group(pgid, sig):
+    try:
+        os.killpg(pgid, sig)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def leader_exit_without_reaping(process, seconds):
+    # Keep the owned leader PID reserved until group cleanup begins.
+    deadline = time.monotonic() + seconds
+    while time.monotonic() < deadline:
+        observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+        if observed is not None:
+            return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+        time.sleep(0.05)
+    raise TimeoutError('owned leader exit deadline')
+
+
+def cleanup_owned_group(process, receipt):
+    pgid = process.pid  # start_new_session=True; never an inherited/shared group.
+    cleanup = receipt['cleanup'] = {'pgid': pgid, 'term_sent': False, 'kill_sent': False,
+                                    'leader_reaped': False, 'group_absent': False}
+    cleanup['term_sent'] = signal_group(pgid, signal.SIGTERM)
+    try:
+        process.wait(timeout=3)
+    except subprocess.TimeoutExpired:
+        cleanup['kill_sent'] = signal_group(pgid, signal.SIGKILL)
+        process.wait(timeout=5)
+    cleanup['leader_reaped'] = True
+    # A reaped leader does not establish that its owned descendants retired.
+    if group_exists(pgid):
+        cleanup['kill_sent'] = signal_group(pgid, signal.SIGKILL) or cleanup['kill_sent']
+    deadline = time.monotonic() + 5
+    while group_exists(pgid) and time.monotonic() < deadline:
+        time.sleep(0.05)
+    cleanup['group_absent'] = not group_exists(pgid)
+    require(cleanup['group_absent'], 'owned process group remains after cleanup')
+
+
+def finish_pipe(process, log, log_bytes, receipt):
+    # Group absence alone does not prove that no writer retained this pipe.
+    os.set_blocking(process.stdout.fileno(), False)
+    deadline = time.monotonic() + 1
+    while time.monotonic() < deadline:
+        try:
+            raw = os.read(process.stdout.fileno(), 65536)
+        except BlockingIOError:
+            time.sleep(0.02)
+            continue
+        if not raw:
+            receipt['cleanup']['pipe_eof'] = True
+            return
+        log_bytes += len(raw)
+        require(log_bytes <= 8 * 1024 * 1024, 'bounded cleanup child log')
+        log.write(raw)
+    raise TimeoutError('owned stdout pipe EOF deadline')
 
 
 def main():
@@ -45,13 +112,16 @@
                'exit_code': None, 'status': 'UNQUALIFIED'}
     with a.new_receipt.open('x') as receipt_file, a.new_log.open('xb') as log:
         process = None
+        selector = None
+        log_bytes = 0
+        primary_error = None
+        cleanup_error = None
         try:
             process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, start_new_session=True)
             receipt['owned_pid'] = process.pid
             selector = selectors.DefaultSelector()
             selector.register(process.stdout, selectors.EVENT_READ)
-            log_bytes = 0
             while selector.get_map():
                 require(time.monotonic() - started < 300, 'finite wall deadline')
                 for key, _ in selector.select(0.2):
@@ -62,7 +132,7 @@
                     log_bytes += len(raw)
                     require(log_bytes <= 8 * 1024 * 1024, 'bounded child log')
                     log.write(raw)
-            receipt['exit_code'] = process.wait(timeout=5)
+            receipt['exit_code'] = leader_exit_without_reaping(process, 5)
             require(receipt['exit_code'] == 0, 'public AV process refused')
             require(digest(a.binary) == before_binary, 'binary changed during run')
             require(verify_fixtures(a.fixtures, pins) == before_fixtures, 'fixtures changed during run')
@@ -70,23 +140,40 @@
             receipt['status'] = 'DEFAULT_GATE_COMPLETED_CONTROLS_DIAGNOSTIC_VALIDATOR_REQUIRED'
         except BaseException as error:
             receipt['error'] = f'{type(error).__name__}: {error}'
-            raise
+            primary_error = error
         finally:
-            if process is not None:
-                if process.poll() is None:
+            try:
+                if process is not None:
+                    cleanup_owned_group(process, receipt)
+                    finish_pipe(process, log, log_bytes, receipt)
+            except BaseException as error:
+                cleanup_error = error
+                receipt['cleanup_error'] = f'{type(error).__name__}: {error}'
+                receipt['status'] = 'UNQUALIFIED'
+            finally:
+                closers = []
+                if selector is not None:
+                    closers.append(('selector_closed', selector.close))
+                if process is not None:
+                    closers.append(('pipe_closed', process.stdout.close))
+                for label, close in closers:
                     try:
-                        os.killpg(process.pid, signal.SIGTERM)
-                        process.wait(timeout=3)
-                    except subprocess.TimeoutExpired:
-                        os.killpg(process.pid, signal.SIGKILL)
-                        process.wait(timeout=5)
-                    except ProcessLookupError:
-                        process.wait(timeout=5)
-                receipt['exit_code'] = process.returncode
+                        close()
+                        receipt.setdefault('cleanup', {})[label] = True
+                    except BaseException as error:
+                        cleanup_error = cleanup_error or error
+                        receipt.setdefault('cleanup_errors', []).append(f'{label}: {type(error).__name__}: {error}')
+                        receipt['status'] = 'UNQUALIFIED'
+                if process is not None:
+                    receipt['exit_code'] = process.returncode
             receipt['elapsed_seconds'] = time.monotonic() - started
             json.dump(receipt, receipt_file, indent=2, allow_nan=False)
             receipt_file.write('\n')
             receipt_file.flush()
+        if primary_error is not None:
+            raise primary_error
+        if cleanup_error is not None:
+            raise cleanup_error
 
 
 if __name__ == '__main__':
--- a/validate.py
+++ b/validate.py
@@ -1,6 +1,7 @@
 #!/usr/bin/env python3
 """ROOT ONLY: authenticate public AV replay; controls remain diagnostic."""
 import argparse
+import hashlib
 import json
 from pathlib import Path
 import struct
@@ -20,6 +21,7 @@
     done = jread(a.output / 'complete.json', 128 * 1024)
     require(done['status'] == 'PUBLIC_AV_REPLAY_DEFAULT_EXACT_CONTROL_DIAGNOSTIC'
             and done['default_cases'] == done['control_cases'] == 65 and done['default_words'] == 262144
+            and type(done['threads']) is int and done['threads'] == 8
             and done['cause_qualified'] is False and done['production_math_changed'] is False,
             'complete default gate and diagnostic-only controls')
     cases = [json.loads(line) for line in bounded(a.output / 'cases.jsonl', 128 * 1024).splitlines()]
@@ -50,6 +52,8 @@
         actual_path = a.output / f'{"ref-control" if control else "default"}-{key}.actual.f32le'
         actual = bounded(actual_path, 8 * 1024 * 1024)
         require(len(actual) == len(expected) == row['words'] * 4, 'complete raw output')
+        actual_receipt = {'bytes': len(actual), 'sha256': hashlib.sha256(actual).hexdigest()}
+        require(digest(actual_path) == actual_receipt, 'unchanged complete output read')
         unequal = sum(left != right for (left,), (right,) in
                       zip(struct.iter_unpack('<I', actual), struct.iter_unpack('<I', expected)))
         require(unequal == row['actual_capture_unequal_words'], 'independent mismatch accounting')
@@ -61,12 +65,17 @@
             ref_bulk_ar = row['ref_bulk_vs_all_default_ar1_unequal_words']
             independently = 0
             for at in range(64):
-                ar = bounded(a.output / f'default-ar1-{at}.actual.f32le', 8192)
+                ar_path = a.output / f'default-ar1-{at}.actual.f32le'
+                ar = bounded(ar_path, 8192)
+                require(len(ar) == 8192
+                        and hashlib.sha256(ar).hexdigest() == receipts[ar_path.name]['sha256']
+                        and digest(ar_path) == receipts[ar_path.name],
+                        'complete unchanged previously validated default AR1 output')
                 selected = b''.join(actual[(head * 64 + at) * 1024:(head * 64 + at + 1) * 1024] for head in range(8))
                 independently += sum(left != right for (left,), (right,) in
                                      zip(struct.iter_unpack('<I', selected), struct.iter_unpack('<I', ar)))
             require(independently == ref_bulk_ar, 'independent ref bulk versus original AR1')
-        receipts[actual_path.name] = digest(actual_path)
+        receipts[actual_path.name] = actual_receipt
     require(defaults == 262144, 'all original output words covered')
     cpu_path = pins['cpu_backend']
     for phase in ['before', 'after']:
--- a/README.md
+++ b/README.md
@@ -68,9 +68,16 @@
   actual capture,300s wall,8GiB address-space limit, finite CPU/file/log caps,
   no NVIDIA exposure and only owned child-process-group cleanup. These are
   configured process ceilings, not ResourceArbiter/physical-RAM qualification.
+  The leader is inspected without reaping before cleanup. Every spawned group
+  receives bounded TERM/KILL cleanup even after leader exit; receipt requires
+  leader reaped, group absent, stdout EOF and closed selector/pipe. Cleanup has
+  separate3s TERM/5s leader-KILL/5s group/1s pipe deadlines; failures stay
+  UNQUALIFIED and retain both primary and cleanup errors.
 - Fixtures/output≤512MiB combined, native live backing≤32MiB/case, separate
   bounded8MiB log. Default and control payloads total2MiB. Python validates
   original descriptor/byte binding, full raw words and before/after pins.
+  Repeated default AR1 reads require all8192bytes and the previously validated
+  content hash; accepted completion explicitly requires8threads.
 - CPU backend maps and public control origin are observed. Internal
   microkernel dispatch remains unobserved; no automatic claim that use_ref
   equality proves every long-context failure or a production fix.
--- a/TODO.md
+++ b/TODO.md
@@ -7,6 +7,9 @@
 - [x] Source-audit original Haswell public dispatch and reduction candidates.
 - [x] Prepare bounded original public ggml AV default/ref-control replay;
   no model load or arithmetic replacement.
+- [x] Independent source review; repair owned-group cleanup after leader exit,
+  re-read output coverage/hash and explicit8thread validation. Original frozen
+  preparation retained; follow-up unexecuted.
 - [ ] Root compile/run/validator and independent source review.
 - [ ] Full long-context causal closure/production implementation remains open.
 

````

## Source: .agents/runs/numerics-and-context-bulk-causality-20261020/FOLLOWUP-SPAN-COVERAGE.patch

Original UTF-8 bytes: 20503; SHA256 `d11cf225a3fa52064a6a07397f6f728d911c29a0647bc6dcec1e0a2ee4b1cdac`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/native_window.cpp
+++ b/native_window.cpp
@@ -13,6 +13,54 @@
 static std::map<std::pair<int,std::string>,int> occurrences;
 static std::set<int> observed_layers;
 static int gdn_calls = 0;
+static size_t actual_context_capacity = 0;
+
+static std::vector<float> window_dense(const ggml_tensor *t,const char *site="selected_or_operand") {
+    size_t count=1,span=sizeof(float);
+    try {
+        require(t && t->type==GGML_TYPE_F32 && t->nb[0]==sizeof(float),"window dense F32 inner geometry");
+        for(int i=0;i<4;i++) {
+            require(t->ne[i]>0 && size_t(t->ne[i])<=2*1024*1024/count,"window logical element cap");
+            count*=size_t(t->ne[i]);
+            const size_t extent=size_t(t->ne[i]-1);
+            require(t->nb[i]%sizeof(float)==0 && (!extent || t->nb[i]<=std::numeric_limits<size_t>::max()/extent),"window stride multiplication overflow");
+            const size_t add=extent*t->nb[i];
+            require(add<=std::numeric_limits<size_t>::max()-span,"window span addition overflow");span+=add;
+        }
+        require(count==size_t(ggml_nelements(t)) && span==ggml_nbytes(t),"window exact physical span");
+        if(span>16*1024*1024) {
+            // Pinned V storage is [capacity,head*dimension], then viewed and
+            // permuted. Read occupied logical rows, never the capacity gaps.
+            require(std::string(site)=="attention_V_cache_operand" && std::string(t->name).rfind("cache_v_l3",0)==0
+                && t->op==GGML_OP_PERMUTE && actual_context_capacity==8448
+                && t->ne[0]>=256 && t->ne[0]<=1024 && t->ne[0]%256==0
+                && t->ne[1]==256 && t->ne[2]==2 && t->ne[3]==1
+                && t->nb[1]==actual_context_capacity*4
+                && t->nb[2]==actual_context_capacity*256*4
+                && t->nb[3]==actual_context_capacity*512*4
+                && span==((512-1)*actual_context_capacity+size_t(t->ne[0]))*4
+                && span<=actual_context_capacity*512*4,"unsupported large window V-cache span");
+        }
+        require(t->data && (t->view_src ? t->view_src->buffer : t->buffer),"window tensor physical buffer");
+        std::vector<float> result(count);size_t written=0;
+        const size_t run=size_t(t->ne[0])*sizeof(float);
+        for(int64_t d=0;d<t->ne[3];d++)for(int64_t c=0;c<t->ne[2];c++)for(int64_t b=0;b<t->ne[1];b++) {
+            const size_t offset=size_t(d)*t->nb[3]+size_t(c)*t->nb[2]+size_t(b)*t->nb[1];
+            require(offset<=span && run<=span-offset && size_t(t->ne[0])<=count-written,"window bounded logical read run");
+            ggml_backend_tensor_get(t,result.data()+written,offset,run);written+=size_t(t->ne[0]);
+        }
+        require(written==count,"window complete logical copy");return result;
+    } catch(const std::exception &) {
+        std::cerr<<"WINDOW_COPY_CONTEXT site="<<site<<" mode="<<provider_mode<<" call="<<call_index
+                 <<" n_ctx="<<actual_context_capacity<<" logical_elements="<<count<<" checked_span="<<span;
+        if(t) {
+            std::cerr<<" name="<<t->name<<" op="<<ggml_op_name(t->op)<<" ne=";
+            for(int i=0;i<4;i++)std::cerr<<(i?",":"")<<t->ne[i];
+            std::cerr<<" nb=";for(int i=0;i<4;i++)std::cerr<<(i?",":"")<<t->nb[i];
+        }
+        std::cerr<<'\n';throw;
+    }
+}
 
 static int tensor_layer(const std::string &name) {
     if(name.rfind("blk.",0)!=0)return -1;
@@ -77,35 +125,35 @@
         // Q/K/V labels occur on both original projection nodes and later
         // head-shaped views. Preserve the actual source geometry for each.
         const int axis=t->op==GGML_OP_MUL_MAT ? 1 : pick.second;
-        record(pick.first,t->name,dense(t),t,axis);
+        record(pick.first,t->name,window_dense(t),t,axis);
     }
     if(projection>=0) {
-        record(projection,"projection_input::"+std::string(t->src[0]->name),dense(t->src[1]),t->src[1],1);
-        record(projection,"projection_output::"+std::string(t->src[0]->name),dense(t),t,1);
+        record(projection,"projection_input::"+std::string(t->src[0]->name),window_dense(t->src[1]),t->src[1],1);
+        record(projection,"projection_output::"+std::string(t->src[0]->name),window_dense(t),t,1);
     }
     if(named_layer(t->name,"kq")==3) {
         require(t->op==GGML_OP_MUL_MAT && t->src[0] && t->src[1],"actual QK operator");
-        record(3,"attention_K_cache_operand",dense(t->src[0]),t->src[0],-1,"full_actual_padded_cache_raw; causal_rows_only_for_per_query_comparison");
-        record(3,"attention_Q_operand",dense(t->src[1]),t->src[1],1);
+        record(3,"attention_K_cache_operand",window_dense(t->src[0]),t->src[0],-1,"full_actual_padded_cache_raw; causal_rows_only_for_per_query_comparison");
+        record(3,"attention_Q_operand",window_dense(t->src[1]),t->src[1],1);
     }
     if(named_layer(t->name,"kq_soft_max")==3) {
         require(t->op==GGML_OP_SOFT_MAX && t->src[0] && t->src[1],"actual softmax/mask operator");
-        record(3,"attention_softmax_scores_operand",dense(t->src[0]),t->src[0],1);
-        record(3,"attention_mask_operand",dense(t->src[1]),t->src[1],1,"actual_mask",true);
+        record(3,"attention_softmax_scores_operand",window_dense(t->src[0]),t->src[0],1);
+        record(3,"attention_mask_operand",window_dense(t->src[1]),t->src[1],1,"actual_mask",true);
     }
     if(named_layer(t->name,"kqv")==3) {
         require(t->op==GGML_OP_MUL_MAT && t->src[0] && t->src[1],"actual AV operator");
-        record(3,"attention_V_cache_operand",dense(t->src[0]),t->src[0],-1,"full_actual_padded_cache_raw; causal_rows_only_for_per_query_comparison");
-        record(3,"attention_probabilities_operand",dense(t->src[1]),t->src[1],1);
+        record(3,"attention_V_cache_operand",window_dense(t->src[0],"attention_V_cache_operand"),t->src[0],-1,"full_actual_padded_cache_raw; causal_rows_only_for_per_query_comparison");
+        record(3,"attention_probabilities_operand",window_dense(t->src[1]),t->src[1],1);
     }
     if(gdn>=0 && gdn<=4) {
         require(gdn!=3 && t->src[0] && t->src[1] && t->src[2] && t->src[3] && t->src[4] && t->src[5],"actual GDN source binding");
         require(t->src[2]->ne[0]==128 && t->src[2]->ne[1]==16 && t->src[2]->ne[2]==call_count && t->src[2]->ne[3]==1,"actual unchanged GDN token width");
         int32_t snapshots=0;std::memcpy(&snapshots,t->op_params,4);require(snapshots==1,"original GDN final-state policy");
         const std::vector<std::string> stages={"gdn_q_operand","gdn_k_operand","gdn_v_operand","gdn_gate_operand","gdn_beta_operand"};
-        for(int i=0;i<5;i++)record(gdn,stages[i],dense(t->src[i]),t->src[i],2);
-        if(provider_mode=="bulk64" || call_index==0)record(gdn,"gdn_prior_state",dense(t->src[5]),t->src[5],-1,"actual_window_initial_state");
-        const auto packed=dense(t);const size_t core=16*128*size_t(call_count),state_words=16*128*128;
+        for(int i=0;i<5;i++)record(gdn,stages[i],window_dense(t->src[i]),t->src[i],2);
+        if(provider_mode=="bulk64" || call_index==0)record(gdn,"gdn_prior_state",window_dense(t->src[5]),t->src[5],-1,"actual_window_initial_state");
+        const auto packed=window_dense(t);const size_t core=16*128*size_t(call_count),state_words=16*128*128;
         require(packed.size()==core+state_words,"complete actual GDN packed output");
         record(gdn,"gdn_core_output",std::vector<float>(packed.begin(),packed.begin()+core),t,2,"actual_packed_output_prefix",false,{128,16,call_count,1});
         if(provider_mode=="bulk64" || call_index+1==window_count)record(gdn,"gdn_proposed_state",std::vector<float>(packed.begin()+core,packed.end()),t,-1,"actual_packed_output_suffix_window_final_state",false,{128,128,16,1},core);
@@ -117,6 +165,26 @@
     folder=base/mode;require(fs::create_directory(folder),"new mode output");require(chmod(folder.c_str(),0700)==0,"mode output permissions");
     provider_mode=mode;window_bytes=window_records=0;observed_layers.clear();gdn_calls=0;
     records.open(folder/"records.jsonl");require(bool(records),"window records open");require(chmod((folder/"records.jsonl").c_str(),0600)==0,"record file mode");
+}
+static void require_current_coverage() {
+    auto one=[](int layer,const std::string &stage) {
+        require(occurrences[{layer,stage}]==1,"missing/duplicate required window field: "+stage);
+    };
+    one(-1,"model.input_embed");
+    for(int layer=0;layer<=4;layer++) {
+        for(const std::string &base:std::vector<std::string>{"attn_norm","attn_residual","attn_post_norm","ffn_out","l_out"})one(layer,base+"-"+std::to_string(layer));
+        auto roles=std::vector<std::string>{"ffn_gate.weight","ffn_up.weight","ffn_down.weight"};
+        const auto attention=layer==3 ? std::vector<std::string>{"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight"}
+            : std::vector<std::string>{"attn_qkv.weight","ssm_alpha.weight","ssm_beta.weight","attn_gate.weight","ssm_out.weight"};
+        roles.insert(roles.end(),attention.begin(),attention.end());
+        for(const auto &role:roles)for(const std::string &prefix:std::vector<std::string>{"projection_input::","projection_output::"})one(layer,prefix+"blk."+std::to_string(layer)+"."+role);
+        if(layer!=3) {
+            for(const std::string &stage:std::vector<std::string>{"gdn_q_operand","gdn_k_operand","gdn_v_operand","gdn_gate_operand","gdn_beta_operand","gdn_core_output"})one(layer,stage);
+            if(provider_mode=="bulk64" || call_index==0)one(layer,"gdn_prior_state");
+            if(provider_mode=="bulk64" || call_index+1==window_count)one(layer,"gdn_proposed_state");
+        }
+    }
+    for(const std::string &stage:std::vector<std::string>{"Qcur_normed-3","Kcur_normed-3","kq-3","kq_soft_max-3","kqv-3","attn_pregate-3","attn_gated-3","attention_K_cache_operand","attention_Q_operand","attention_V_cache_operand","attention_softmax_scores_operand","attention_mask_operand","attention_probabilities_operand"})one(3,stage);
 }
 static size_t end_observation() {
     require(observed_layers==std::set<int>({0,1,2,3,4}),"missing first-five-layer observations");
@@ -156,6 +224,7 @@
         auto serial_params=params;serial_params.n_batch=1;serial_params.n_ubatch=1;
         Context serial(llama_init_from_model(model.get(),serial_params),llama_free);require(bool(serial),"plain AR1 context");
         require(llama_n_ctx(bulk.get())==llama_n_ctx(serial.get()) && llama_n_batch(bulk.get())==64 && llama_n_ubatch(bulk.get())==64 && llama_n_batch(serial.get())==1 && llama_n_ubatch(serial.get())==1,"actual unchanged width configuration");
+        actual_context_capacity=llama_n_ctx(bulk.get());require(actual_context_capacity==8448,"pinned8320 context padded to256");
         for(int at=0;at<window_start;at+=64)decode(bulk.get(),ids,at,64,false);
         for(int at=0;at<window_start;at++)decode(serial.get(),ids,at,1,false);
         const auto before_bulk=state(bulk.get()),before_serial=state(serial.get());
@@ -168,13 +237,13 @@
         params.cb_eval=window_observer;Context observed_bulk(llama_init_from_model(model.get(),params),llama_free);require(bool(observed_bulk),"observed bulk context");
         require(llama_state_set_data(observed_bulk.get(),before_bulk.data(),before_bulk.size())==before_bulk.size(),"same-source bulk prior restore");
         begin_observation(base,"bulk64");enabled=true;call_index=0;call_count=window_count;occurrences.clear();
-        decode(observed_bulk.get(),ids,window_start,window_count,false);enabled=false;
+        decode(observed_bulk.get(),ids,window_start,window_count,false);enabled=false;require_current_coverage();
         const size_t bulk_bytes=end_observation();require(state(observed_bulk.get())==after_bulk,"observer changed unchanged-width bulk complete state");
         guard_next(bulk.get(),observed_bulk.get(),ids,window_start+window_count,base/"bulk.next-logits.f32le");
         serial_params.cb_eval=window_observer;Context observed_ar(llama_init_from_model(model.get(),serial_params),llama_free);require(bool(observed_ar),"observed AR1 context");
         require(llama_state_set_data(observed_ar.get(),before_serial.data(),before_serial.size())==before_serial.size(),"same-source AR1 prior restore");
         begin_observation(base,"ar1");call_count=1;enabled=true;
-        for(int at=0;at<window_count;at++){call_index=at;occurrences.clear();decode(observed_ar.get(),ids,window_start+at,1,false);}
+        for(int at=0;at<window_count;at++){call_index=at;occurrences.clear();decode(observed_ar.get(),ids,window_start+at,1,false);require_current_coverage();}
         enabled=false;const size_t ar_bytes=end_observation();require(state(observed_ar.get())==after_serial,"observer changed AR1 complete state");
         guard_next(serial.get(),observed_ar.get(),ids,window_start+window_count,base/"ar1.next-logits.f32le");
         std::ofstream done(base/"complete.json");require(bool(done),"window complete record open");
--- a/compare_window.py
+++ b/compare_window.py
@@ -68,7 +68,39 @@
         total += row["elements"] * 4
         require(total <= CAP, "aggregate observation payload cap")
         fields[key] = row
+    require_coverage(fields, mode, count)
     return fields, hashlib.sha256(raw).hexdigest()
+
+
+def require_coverage(fields, mode, count):
+    lookup = {}
+    for call, layer, stage, occurrence in fields:
+        lookup.setdefault((call, layer, stage), []).append(occurrence)
+    def one(call, layer, stage):
+        require(lookup.get((call, layer, stage)) == [0], f"missing/duplicate required field: {call}/{layer}/{stage}")
+    for call in range(1 if mode == "bulk64" else count):
+        one(call, -1, "model.input_embed")
+        for layer in range(5):
+            for stage in ["attn_norm", "attn_residual", "attn_post_norm", "ffn_out", "l_out"]:
+                one(call, layer, f"{stage}-{layer}")
+            roles = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
+            roles += (["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"] if layer == 3 else
+                      ["attn_qkv.weight", "ssm_alpha.weight", "ssm_beta.weight", "attn_gate.weight", "ssm_out.weight"])
+            for role in roles:
+                for prefix in ["projection_input::", "projection_output::"]:
+                    one(call, layer, f"{prefix}blk.{layer}.{role}")
+            if layer != 3:
+                for stage in ["gdn_q_operand", "gdn_k_operand", "gdn_v_operand", "gdn_gate_operand", "gdn_beta_operand", "gdn_core_output"]:
+                    one(call, layer, stage)
+                if mode == "bulk64" or call == 0:
+                    one(call, layer, "gdn_prior_state")
+                if mode == "bulk64" or call + 1 == count:
+                    one(call, layer, "gdn_proposed_state")
+        for stage in ["Qcur_normed-3", "Kcur_normed-3", "kq-3", "kq_soft_max-3", "kqv-3",
+                      "attn_pregate-3", "attn_gated-3", "attention_K_cache_operand", "attention_Q_operand",
+                      "attention_V_cache_operand", "attention_softmax_scores_operand", "attention_mask_operand",
+                      "attention_probabilities_operand"]:
+            one(call, 3, stage)
 
 
 def field_bytes(folder, row):
--- a/AUDIT.md
+++ b/AUDIT.md
@@ -4,6 +4,9 @@
 Owner: `/root/spine_review`. Source-only preparation; root alone builds/runs.
 No production/frozen-helper/capture modifications. No Cargo/C++/native/model/
 operator/runtime/GPU execution by author.
+
+First64 root observation refused the old16MiB backing-span cap before completion;
+see `SPAN_FIX.md` for preserved failure and exact V-cache logical-run follow-up.
 
 ## Current actual evidence
 
--- a/TODO.md
+++ b/TODO.md
@@ -9,6 +9,10 @@
   source arithmetic and cannot identify the original first divergent row.
 - [x] Prepare source-only unchanged-block/native-AR1 observer + bit comparator;
   all input/result rows and independent state/next-logit transparency guards.
+- [x] Preserve root first64 span refusal; prepare local logical-run V-cache
+  copier with exact context/head/stride/span guards and refusal context.
+- [x] Require projection/attention/GDN semantic families per actual call in
+  both emitter and comparator; no silently partial observation scope.
 - [ ] Root/independent review, compile and actual first64-row bounded control.
 - [ ] Actual first causal row/operator with complete equal inputs/prior.
 - [ ] Source-correct existing-engine fix; actual operator/full-state/long gates.
--- a/source-pins.json
+++ b/source-pins.json
@@ -115,6 +115,16 @@
       "path": "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so",
       "bytes": 1459208,
       "sha256": "ca75af8b6281c57114f411d8883120a4ee595d8cb05c998a0b5e40921417c935"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml.c",
+      "bytes": 261112,
+      "sha256": "7e8bf2b60598b50d718f46aa3c36c9293b10f4e7dfde6844f0693e6fc976d505"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend.cpp",
+      "bytes": 105561,
+      "sha256": "cdb79f2937aa147084a76f949adc0eb760ad1883e0ccc015bb4e0f599c89bd5f"
     }
   ]
 }
--- a/SPAN_FIX.md
+++ b/SPAN_FIX.md
@@ -0,0 +1,47 @@
+# First64 observation span refusal: narrow follow-up
+
+Initial frozen manifest564ab0e0/helper639fc525 and root-first64 output remain
+unchanged. Root compiled that helper strictly; four source-free comparator
+fixtures passed. Actual run exited3 before transparency completion, therefore
+establishes no accepted native window/model inference result.
+
+- Last completed field152: actual `kqv-3`, ne[256,64,8,1]. Next field reads its
+  V-cache input. Previously inherited `dense()` refused any backing span>16MiB.
+- Pinned `llama-context.cpp:292` pads requested8320 context to256→8448. Actual
+  common contexts and earlier captured K-cache nb3=17301504 independently agree.
+- `llama-kv-cache.cpp:1329/:1514`: transposed V storage is
+  `[capacity,head*dimension]`; view+permute gives ne[visibleKV,256,2,1] with
+  nb[4,8448*4,8448*256*4,8448*512*4]. At visibleKV256, exact minimum backing
+  span is `((512-1)*8448+256)*4=17268736B`, while logical data is524288B.
+  `ggml.c:1298` computes exactly4+sum((ne-1)*nb). No larger tensor was requested.
+
+`window_dense()` is local to this new diagnostic only:
+
+- Existing native_raw.cpp/dense() bytes and all frozen/captured attempts unchanged.
+- F32 innerstride4; positive checked dimensions/product≤2M words; checked stride
+  products/span additions; computed span must equal actual public ggml_nbytes.
+- Ordinary spans retain≤16MiB policy. The only larger accepted span is exactly
+  the named attention3 transposed V-cache operand with actual context8448,
+  visibleKV256/512/768/1024, heads2×dim256 and exact above strides/formula.
+  Its range remains inside the actual full cache allocation17301504B.
+- Allocate only logical≤8MiB result. Public `ggml_backend_tensor_get` copies each
+  inner contiguous run into old dense order; read offsets/counts checked against
+  the exact tensor span, no capacity-gap read/copy and no scalar math operation.
+- Refusals now print bounded call/site/name/op/ne/nb/context/logical/span details.
+  Original exact full-state/next-token guards, limits and source math unchanged.
+
+Independent review also found mandatory-field coverage should be explicit.
+New C++ checks each actual call; comparator independently requires every call's
+embedding, first5-layer broad boundaries, exact original projection role
+inputs/outputs, attention3 QK/mask/softmax/AV/cache inputs and all GDN0/1/2/4
+q/k/v/g/b/core plus initial/final states at their actual boundaries. No omitted
+attention field may silently reduce scope. Runtime callback labels still must
+match actual source; no guessed expected value or numerical exception.
+
+Conv source history/kernel inputs remain unobserved. A convolution-result gap
+would still need a narrow actual operand capture before causal attribution.
+The comparator stays diagnostic-only and never auto-promotes a cause/profile.
+
+Author performed source/hash/AST checks only. Root independently reviews,
+compiles a fresh binary and runs a **new** exclusive attempt directory; no
+native/model/operator/runtime/Cargo/C++compile/GPU execution by author.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 22333; SHA256 `e1460ed2f9a5db11f82a57cb456091d9072ef0cd87d782e2729171c2a8a03b29`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs b/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs
@@ -0,0 +1,257 @@
+use super::*;
+use std::sync::Arc;
+
+fn budget(limit: u64) -> (ResourceArbiter, PhysicalDeviceKey, ResourceOwner) {
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "bulk-f32-unit").unwrap();
+    arbiter.register(key.clone(), limit).unwrap();
+    (
+        arbiter,
+        key,
+        ResourceOwner::new("source-free", "unit").unwrap(),
+    )
+}
+fn request<'a>(source: SourceIdentity, a: &'a [f32], b: &'a [f32]) -> BulkF32Request<'a> {
+    BulkF32Request {
+        source,
+        profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
+        left: F32View {
+            source,
+            name: "left",
+            values: a,
+            offset: 0,
+            shape: [8, 4, 1, 1],
+            strides: [1, 8, 32, 32],
+        },
+        right: F32View {
+            source,
+            name: "right",
+            values: b,
+            offset: 0,
+            shape: [8, 2, 1, 1],
+            strides: [1, 8, 16, 16],
+        },
+    }
+}
+
+#[test]
+fn source_one_chain_and_horizontal_order_have_independent_counterexamples() {
+    let a = [1.0; 32];
+    let mut b = [0.0; 32];
+    b[0] = 16_777_216.0;
+    b[8] = 1.0;
+    b[16] = -16_777_216.0;
+    b[24] = 1.0;
+    // Four independent Vec8 partials yield2; the original one-chain yields1.
+    assert_eq!(dot(&a, &b).to_bits(), 1.0f32.to_bits());
+    let h = [67_108_864.0, 1.0, -67_108_864.0, 1.0, 0.0, 0.0, 0.0, 0.0];
+    // Even/odd horizontal pairing yields2; adjacent-pair horizontal gives0.
+    assert_eq!(dot(&[1.0; 8], &h).to_bits(), 2.0f32.to_bits());
+}
+
+#[test]
+fn real_zero_filled_tail_fmas_preserve_source_signed_zero_semantics() {
+    let tiny = f32::from_bits(1);
+    assert_eq!(dot(&[-tiny; 8], &[tiny; 8]).to_bits(), (-0.0f32).to_bits());
+    // Original load_partial fills BOTH unused operands with +0 before FMA.
+    assert_eq!(dot(&[-tiny; 9], &[tiny; 9]).to_bits(), 0.0f32.to_bits());
+    assert_eq!(dot(&[2.0; 3], &[3.0; 3]).to_bits(), 18.0f32.to_bits());
+}
+
+#[test]
+fn strided_offset_broadcast_canaries_and_output_lease_are_real() {
+    let source = SourceIdentity::for_new_allocation().unwrap();
+    let mut a = vec![f32::NAN; 128];
+    for head in 0..2 {
+        for row in 0..4 {
+            a[3 + head * 64 + row * 12..3 + head * 64 + row * 12 + 8]
+                .fill((head * 4 + row + 1) as f32);
+        }
+    }
+    let mut b = vec![f32::NAN; 66];
+    b[1..65].fill(1.0);
+    let mut r = request(source, &a, &b);
+    r.left.offset = 3;
+    r.left.strides = [1, 12, 64, 128];
+    r.left.shape = [8, 4, 2, 1];
+    r.right.offset = 1;
+    r.right.shape = [8, 2, 4, 1];
+    r.right.strides = [1, 8, 16, 64];
+    let (arbiter, key, owner) = budget(4096);
+    let result = execute(
+        r,
+        BulkF32Admission {
+            arbiter: &arbiter,
+            resource: &key,
+            owner: &owner,
+        },
+        None,
+    )
+    .unwrap();
+    assert_eq!(result.shape(), [4, 2, 4, 1]);
+    for head in 0..4 {
+        for column in 0..2 {
+            for row in 0..4 {
+                assert_eq!(
+                    result.values()[row + 4 * (column + 2 * head)],
+                    (head / 2 * 4 + row + 1) as f32 * 8.0
+                );
+            }
+        }
+    }
+    let live = arbiter.snapshot(&key).unwrap();
+    assert_eq!(live.reservations, 1);
+    assert_eq!(live.per_class[&AllocationClass::Output], 128);
+    assert_eq!(live.per_class[&AllocationClass::Scratch], SCRATCH_BYTES);
+    assert!(a[0].is_nan() && a[11].is_nan() && b[0].is_nan() && b[65].is_nan());
+    assert_eq!(result.source(), source);
+    drop(result);
+    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+}
+
+#[test]
+fn foreign_source_stride_span_overflow_and_ar1_are_closed() {
+    let source = SourceIdentity::for_new_allocation().unwrap();
+    let a = [1.0; 32];
+    let b = [1.0; 16];
+    let (arbiter, key, owner) = budget(4096);
+    let valid = request(source, &a, &b);
+    let mut cases = Vec::new();
+    let mut r = valid;
+    r.right.source = SourceIdentity::for_new_allocation().unwrap();
+    cases.push(r);
+    let mut r = valid;
+    r.left.strides[1] = 7;
+    cases.push(r);
+    let mut r = valid;
+    r.left.offset = 1;
+    cases.push(r);
+    let mut r = valid;
+    r.left.offset = usize::MAX;
+    cases.push(r);
+    let mut r = valid;
+    r.left.strides[3] = usize::MAX;
+    r.left.shape[3] = 2;
+    cases.push(r);
+    let mut r = valid;
+    r.right.shape[1] = 1;
+    r.right.strides = [1, 8, 8, 8];
+    cases.push(r);
+    let mut r = valid;
+    r.right.strides[3] = 17;
+    cases.push(r);
+    for r in cases {
+        assert!(
+            execute(
+                r,
+                BulkF32Admission {
+                    arbiter: &arbiter,
+                    resource: &key,
+                    owner: &owner
+                },
+                None
+            )
+            .is_err()
+        );
+        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+    }
+}
+
+#[test]
+fn actual_capacity_pre_cancel_and_mid_output_cancel_release_only_private_storage() {
+    let source = SourceIdentity::for_new_allocation().unwrap();
+    let a = [1.0; 32];
+    let b = [1.0; 16];
+    let (arbiter, key, owner) = budget(32 + SCRATCH_BYTES - 1);
+    assert!(matches!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            None
+        ),
+        Err(Error::Limit(_))
+    ));
+    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+    let (arbiter, key, owner) = budget(4096);
+    let flag = Arc::new(AtomicBool::new(true));
+    assert!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            Some(&flag)
+        )
+        .is_err()
+    );
+    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+    flag.store(false, Ordering::Release);
+    let observed_arbiter = arbiter.clone();
+    let observed_key = key.clone();
+    let observed_flag = flag.clone();
+    AFTER_FIRST_OUTPUT.with(|hook| {
+        *hook.borrow_mut() = Some(Box::new(move || {
+            assert_eq!(
+                observed_arbiter.snapshot(&observed_key).unwrap().reserved,
+                32 + SCRATCH_BYTES
+            );
+            observed_flag.store(true, Ordering::Release);
+        }))
+    });
+    assert!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            Some(&flag)
+        )
+        .is_err()
+    );
+    let final_state = arbiter.snapshot(&key).unwrap();
+    assert_eq!(final_state.reserved, 0);
+    assert!(!final_state.fenced);
+}
+
+#[test]
+fn nonfinite_used_words_and_finite_input_overflow_refuse_without_outputs() {
+    let source = SourceIdentity::for_new_allocation().unwrap();
+    let (arbiter, key, owner) = budget(4096);
+    let mut a = [1.0; 32];
+    let b = [2.0; 16];
+    a[19] = f32::INFINITY;
+    assert!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            None
+        )
+        .is_err()
+    );
+    a.fill(f32::MAX);
+    assert!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            None
+        )
+        .is_err()
+    );
+    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+}
diff --git a/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs b/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs
@@ -0,0 +1,364 @@
+//! Closed GGML d812350 Haswell tinyBLAS F32 schedule, no model/plan selection.
+//!
+//! One Vec8 FMA chain per output and the original even/odd horizontal tree.
+//! Default GGUF four-Vec8 projections/attention and source profiles are unchanged.
+//! Source: ggml/src/ggml-cpu/llamafile/sgemm.cpp:252-270,559-730,3908-3934.
+
+/*
+MIT License
+
+Copyright (c) 2023-2026 The ggml authors
+
+Permission is hereby granted, free of charge, to any person obtaining a copy
+of this software and associated documentation files (the "Software"), to deal
+in the Software without restriction, including without limitation the rights
+to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
+copies of the Software, and to permit persons to whom the Software is
+furnished to do so, subject to the following conditions:
+
+The above copyright notice and this permission notice shall be included in all
+copies or substantial portions of the Software.
+
+THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
+IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
+FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
+AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
+LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
+OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
+SOFTWARE.
+*/
+
+use crate::accelerator::SourceIdentity;
+use crate::{Error, Result};
+use std::sync::atomic::{AtomicBool, Ordering};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
+    ResourceOwner,
+};
+
+const MAX_SPAN_WORDS: usize = 8 * 1024 * 1024;
+const MAX_OUTPUT_WORDS: usize = 1024 * 1024;
+const MAX_FMAS: usize = 128 * 1024 * 1024;
+const SCRATCH_BYTES: u64 = 256;
+
+#[cfg(test)]
+std::thread_local! {
+    static AFTER_FIRST_OUTPUT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
+        const { std::cell::RefCell::new(None) };
+}
+
+/// Distinct closed arithmetic contract; no executable model/profile capability.
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum BulkF32Profile {
+    GgmlD812350HaswellTinyBlas,
+}
+
+/// Original F32 axes `[inner,rows,head,sequence]`; strides/offset are in words.
+/// Borrowed raw activations/weights retain their caller's owner and admission.
+/// This descriptor validates a view, not its model-file SHA or source issuer.
+#[derive(Clone, Copy)]
+pub struct F32View<'a> {
+    pub source: SourceIdentity,
+    pub name: &'a str,
+    pub values: &'a [f32],
+    pub offset: usize,
+    pub shape: [usize; 4],
+    pub strides: [usize; 4],
+}
+
+/// Same source for both operands. B is the original contiguous RHS; A may
+/// retain a strided V-cache view. All broadcast heads are independently read.
+#[derive(Clone, Copy)]
+pub struct BulkF32Request<'a> {
+    pub profile: BulkF32Profile,
+    pub source: SourceIdentity,
+    pub left: F32View<'a>,
+    pub right: F32View<'a>,
+}
+
+/// Existing host ledger, not a new pool or a claim about all host RAM.
+pub struct BulkF32Admission<'a> {
+    pub arbiter: &'a ResourceArbiter,
+    pub resource: &'a PhysicalDeviceKey,
+    pub owner: &'a ResourceOwner,
+}
+
+/// Payload drops before its original output/scratch ticket. No unleased Vec
+/// escape or state publication; synchronous borrowed inputs outlive execution.
+pub struct BulkF32Completed {
+    values: Vec<f32>,
+    source: SourceIdentity,
+    shape: [usize; 4],
+    _lease: GlobalResourceLease,
+}
+impl BulkF32Completed {
+    pub fn values(&self) -> &[f32] {
+        &self.values
+    }
+    pub fn source(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn shape(&self) -> [usize; 4] {
+        self.shape
+    }
+    pub fn profile(&self) -> BulkF32Profile {
+        BulkF32Profile::GgmlD812350HaswellTinyBlas
+    }
+}
+
+fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
+    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        return Err(Error::State("Haswell bulk F32 cancelled".into()));
+    }
+    Ok(())
+}
+
+fn checked_product(values: &[usize]) -> Result<usize> {
+    values.iter().try_fold(1usize, |acc, &next| {
+        acc.checked_mul(next)
+            .ok_or_else(|| Error::Limit("Haswell bulk F32 geometry overflow".into()))
+    })
+}
+
+impl F32View<'_> {
+    fn validate(self, source: SourceIdentity, cancelled: Option<&AtomicBool>) -> Result<()> {
+        not_cancelled(cancelled)?;
+        if !source.is_bound()
+            || self.source != source
+            || self.name.is_empty()
+            || self.name.len() > 256
+            || self.name.contains('\0')
+            || self.shape.contains(&0)
+            || self.shape[0] > 8192
+            || self.shape[1] > 4096
+            || self.shape[2] > 64
+            || self.shape[3] > 64
+            || self.strides[0] != 1
+        {
+            return Err(Error::Malformed(
+                "Haswell bulk F32 source/view geometry".into(),
+            ));
+        }
+        // Original matrix rows do not overlap. Axis3's declared stride is
+        // checked even when that dimension has size1, preserving raw geometry.
+        let mut previous = self.shape[0];
+        for (&stride, &size) in self.strides.iter().zip(&self.shape).skip(1) {
+            if stride < previous {
+                return Err(Error::Malformed(
+                    "Haswell bulk F32 overlapping strides".into(),
+                ));
+            }
+            previous = stride
+                .checked_mul(size)
+                .ok_or_else(|| Error::Limit("Haswell bulk F32 stride overflow".into()))?;
+        }
+        let end = self
+            .shape
+            .iter()
+            .zip(self.strides)
+            .try_fold(self.offset, |at, (&size, stride)| {
+                (size - 1)
+                    .checked_mul(stride)
+                    .and_then(|n| at.checked_add(n))
+                    .ok_or_else(|| Error::Limit("Haswell bulk F32 view span overflow".into()))
+            })?
+            .checked_add(1)
+            .ok_or_else(|| Error::Limit("Haswell bulk F32 view end overflow".into()))?;
+        if end > self.values.len() || self.values.len() > MAX_SPAN_WORDS {
+            return Err(Error::Limit(
+                "Haswell bulk F32 view exceeds backing/bound".into(),
+            ));
+        }
+        let words = checked_product(&self.shape)?;
+        if words > MAX_OUTPUT_WORDS {
+            return Err(Error::Limit(
+                "Haswell bulk F32 logical operand bound".into(),
+            ));
+        }
+        // Only actual logical words are read; storage gaps/canaries remain
+        // outside the arithmetic and may contain arbitrary IEEE754 words.
+        for sequence in 0..self.shape[3] {
+            for head in 0..self.shape[2] {
+                for row in 0..self.shape[1] {
+                    not_cancelled(cancelled)?;
+                    let start = self.at(0, row, head, sequence);
+                    if self.values[start..start + self.shape[0]]
+                        .iter()
+                        .any(|x| !x.is_finite())
+                    {
+                        return Err(Error::Malformed(
+                            "nonfinite Haswell bulk F32 operand".into(),
+                        ));
+                    }
+                }
+            }
+        }
+        Ok(())
+    }
+    fn at(self, inner: usize, row: usize, head: usize, sequence: usize) -> usize {
+        // Whole view's final index and products were checked before access.
+        self.offset
+            + inner
+            + row * self.strides[1]
+            + head * self.strides[2]
+            + sequence * self.strides[3]
+    }
+}
+
+fn validate(
+    request: BulkF32Request<'_>,
+    cancelled: Option<&AtomicBool>,
+) -> Result<([usize; 4], usize)> {
+    not_cancelled(cancelled)?;
+    let BulkF32Profile::GgmlD812350HaswellTinyBlas = request.profile;
+    let a = request.left;
+    let b = request.right;
+    a.validate(request.source, cancelled)?;
+    b.validate(request.source, cancelled)?;
+    if a.shape[0] != b.shape[0]
+        || !(2..=64).contains(&b.shape[1])
+        || !a.shape[1].is_multiple_of(4)
+        || !b.shape[2].is_multiple_of(a.shape[2])
+        || !b.shape[3].is_multiple_of(a.shape[3])
+    {
+        // n<2 selects the original vec-dot path, never this bulk schedule.
+        return Err(Error::Unsupported(
+            "unsupported Haswell tinyBLAS F32 shape".into(),
+        ));
+    }
+    let mut stride = 1usize;
+    for (&actual, &size) in b.strides.iter().zip(&b.shape) {
+        if actual != stride {
+            return Err(Error::Malformed(
+                "Haswell tinyBLAS requires original contiguous RHS".into(),
+            ));
+        }
+        stride = stride
+            .checked_mul(size)
+            .ok_or_else(|| Error::Limit("Haswell bulk F32 contiguous span overflow".into()))?;
+    }
+    let shape = [a.shape[1], b.shape[1], b.shape[2], b.shape[3]];
+    let count = checked_product(&shape)?;
+    if count > MAX_OUTPUT_WORDS || count.checked_mul(a.shape[0]).is_none_or(|n| n > MAX_FMAS) {
+        return Err(Error::Limit("Haswell bulk F32 output/work bound".into()));
+    }
+    not_cancelled(cancelled)?;
+    Ok((shape, count))
+}
+
+fn dot(a: &[f32], b: &[f32]) -> f32 {
+    let mut lanes = [0.0f32; 8];
+    let full = a.len() & !7usize;
+    for base in (0..full).step_by(8) {
+        for (lane, sum) in lanes.iter_mut().enumerate() {
+            *sum = a[base + lane].mul_add(b[base + lane], *sum);
+        }
+    }
+    if full != a.len() {
+        // tinyBLAS load_partial zero-fills BOTH remaining Vec8 operands.
+        // Execute those +0*+0 FMAs too; skipping them changes zero signs.
+        for (lane, sum) in lanes.iter_mut().enumerate() {
+            let at = full + lane;
+            let (x, y) = if at < a.len() {
+                (a[at], b[at])
+            } else {
+                (0.0, 0.0)
+            };
+            *sum = x.mul_add(y, *sum);
+        }
+    }
+    let low = [
+        lanes[0] + lanes[4],
+        lanes[1] + lanes[5],
+        lanes[2] + lanes[6],
+        lanes[3] + lanes[7],
+    ];
+    // Actual hsum(__m128): add(movehl), then add_ss(movehdup).
+    // This is (0+2)+(1+3), not vec_dot's (0+1)+(2+3).
+    (low[0] + low[2]) + (low[1] + low[3])
+}
+
+/// Pure source-derived primitive. Provider/model/session/checkpoint eligibility
+/// is unchanged. Caller authenticates operands and retains input reservations;
+/// this function admits only its own private result and bounded stack scratch.
+pub fn execute(
+    request: BulkF32Request<'_>,
+    admission: BulkF32Admission<'_>,
+    cancelled: Option<&AtomicBool>,
+) -> Result<BulkF32Completed> {
+    let (shape, count) = validate(request, cancelled)?;
+    if admission.resource.backend != "host" {
+        return Err(Error::Unsupported(
+            "Haswell bulk F32 requires existing host admission".into(),
+        ));
+    }
+    let output_bytes = count
+        .checked_mul(4)
+        .ok_or_else(|| Error::Limit("Haswell bulk F32 output bytes overflow".into()))?;
+    let lease = admission.arbiter.admit(
+        admission.resource,
+        admission.owner.clone(),
+        &[
+            AllocationRequest {
+                kind: AllocationClass::Output,
+                bytes: output_bytes as u64,
+            },
+            AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: SCRATCH_BYTES,
+            },
+        ],
+    )?;
+    not_cancelled(cancelled)?;
+    let mut values = Vec::new();
+    values
+        .try_reserve_exact(count)
+        .map_err(|_| Error::Limit("Haswell bulk F32 output allocation".into()))?;
+    if values.capacity() != count {
+        return Err(Error::Limit(
+            "Haswell bulk F32 unexpected output capacity".into(),
+        ));
+    }
+    let a = request.left;
+    let b = request.right;
+    let ratio2 = b.shape[2] / a.shape[2];
+    let ratio3 = b.shape[3] / a.shape[3];
+    for sequence in 0..shape[3] {
+        for head in 0..shape[2] {
+            for column in 0..shape[1] {
+                not_cancelled(cancelled)?;
+                let br = b.at(0, column, head, sequence);
+                for row in 0..shape[0] {
+                    not_cancelled(cancelled)?;
+                    let ar = a.at(0, row, head / ratio2, sequence / ratio3);
+                    let value = dot(
+                        &a.values[ar..ar + a.shape[0]],
+                        &b.values[br..br + b.shape[0]],
+                    );
+                    if !value.is_finite() {
+                        return Err(Error::Limit("Haswell bulk F32 nonfinite result".into()));
+                    }
+                    values.push(value);
+                    #[cfg(test)]
+                    if values.len() == 1 {
+                        AFTER_FIRST_OUTPUT.with(|hook| {
+                            if let Some(hook) = hook.borrow_mut().take() {
+                                hook();
+                            }
+                        });
+                    }
+                }
+            }
+        }
+    }
+    not_cancelled(cancelled)?;
+    Ok(BulkF32Completed {
+        values,
+        source: request.source,
+        shape,
+        _lease: lease,
+    })
+}
+
+#[cfg(test)]
+mod tests;
diff --git a/crates/zerogravity-cpu/src/math.rs b/crates/zerogravity-cpu/src/math.rs
--- a/crates/zerogravity-cpu/src/math.rs
+++ b/crates/zerogravity-cpu/src/math.rs
@@ -368,3 +368,6 @@
         assert_eq!(ggml_source_silu(0.0), 0.0);
     }
 }
+
+/// Explicit pinned GGML bulk-F32 diagnostic; default source math is unchanged.
+pub mod haswell_bulk_f32;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/followup-attribution/ATTRIBUTION_ONLY.patch

Original UTF-8 bytes: 1534; SHA256 `ce21ffea1afe0851004dba001c3d239d1522d79be84155ac14f424a8c56706b1`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs
+++ b/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs
@@ -3,6 +3,28 @@
 //! One Vec8 FMA chain per output and the original even/odd horizontal tree.
 //! Default GGUF four-Vec8 projections/attention and source profiles are unchanged.
 //! Source: ggml/src/ggml-cpu/llamafile/sgemm.cpp:252-270,559-730,3908-3934.
+
+// Copyright 2024 Mozilla Foundation
+//
+// Permission is hereby granted, free of charge, to any person obtaining
+// a copy of this software and associated documentation files (the
+// "Software"), to deal in the Software without restriction, including
+// without limitation the rights to use, copy, modify, merge, publish,
+// distribute, sublicense, and/or sell copies of the Software, and to
+// permit persons to whom the Software is furnished to do so, subject to
+// the following conditions:
+//
+// The above copyright notice and this permission notice shall be
+// included in all copies or substantial portions of the Software.
+//
+// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
+// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
+// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
+// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
+// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
+// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
+// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
+// SOFTWARE.
 
 /*
 MIT License

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/NEW_DELTA.patch

Original UTF-8 bytes: 22333; SHA256 `e1460ed2f9a5db11f82a57cb456091d9072ef0cd87d782e2729171c2a8a03b29`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs b/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs
@@ -0,0 +1,257 @@
+use super::*;
+use std::sync::Arc;
+
+fn budget(limit: u64) -> (ResourceArbiter, PhysicalDeviceKey, ResourceOwner) {
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "bulk-f32-unit").unwrap();
+    arbiter.register(key.clone(), limit).unwrap();
+    (
+        arbiter,
+        key,
+        ResourceOwner::new("source-free", "unit").unwrap(),
+    )
+}
+fn request<'a>(source: SourceIdentity, a: &'a [f32], b: &'a [f32]) -> BulkF32Request<'a> {
+    BulkF32Request {
+        source,
+        profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
+        left: F32View {
+            source,
+            name: "left",
+            values: a,
+            offset: 0,
+            shape: [8, 4, 1, 1],
+            strides: [1, 8, 32, 32],
+        },
+        right: F32View {
+            source,
+            name: "right",
+            values: b,
+            offset: 0,
+            shape: [8, 2, 1, 1],
+            strides: [1, 8, 16, 16],
+        },
+    }
+}
+
+#[test]
+fn source_one_chain_and_horizontal_order_have_independent_counterexamples() {
+    let a = [1.0; 32];
+    let mut b = [0.0; 32];
+    b[0] = 16_777_216.0;
+    b[8] = 1.0;
+    b[16] = -16_777_216.0;
+    b[24] = 1.0;
+    // Four independent Vec8 partials yield2; the original one-chain yields1.
+    assert_eq!(dot(&a, &b).to_bits(), 1.0f32.to_bits());
+    let h = [67_108_864.0, 1.0, -67_108_864.0, 1.0, 0.0, 0.0, 0.0, 0.0];
+    // Even/odd horizontal pairing yields2; adjacent-pair horizontal gives0.
+    assert_eq!(dot(&[1.0; 8], &h).to_bits(), 2.0f32.to_bits());
+}
+
+#[test]
+fn real_zero_filled_tail_fmas_preserve_source_signed_zero_semantics() {
+    let tiny = f32::from_bits(1);
+    assert_eq!(dot(&[-tiny; 8], &[tiny; 8]).to_bits(), (-0.0f32).to_bits());
+    // Original load_partial fills BOTH unused operands with +0 before FMA.
+    assert_eq!(dot(&[-tiny; 9], &[tiny; 9]).to_bits(), 0.0f32.to_bits());
+    assert_eq!(dot(&[2.0; 3], &[3.0; 3]).to_bits(), 18.0f32.to_bits());
+}
+
+#[test]
+fn strided_offset_broadcast_canaries_and_output_lease_are_real() {
+    let source = SourceIdentity::for_new_allocation().unwrap();
+    let mut a = vec![f32::NAN; 128];
+    for head in 0..2 {
+        for row in 0..4 {
+            a[3 + head * 64 + row * 12..3 + head * 64 + row * 12 + 8]
+                .fill((head * 4 + row + 1) as f32);
+        }
+    }
+    let mut b = vec![f32::NAN; 66];
+    b[1..65].fill(1.0);
+    let mut r = request(source, &a, &b);
+    r.left.offset = 3;
+    r.left.strides = [1, 12, 64, 128];
+    r.left.shape = [8, 4, 2, 1];
+    r.right.offset = 1;
+    r.right.shape = [8, 2, 4, 1];
+    r.right.strides = [1, 8, 16, 64];
+    let (arbiter, key, owner) = budget(4096);
+    let result = execute(
+        r,
+        BulkF32Admission {
+            arbiter: &arbiter,
+            resource: &key,
+            owner: &owner,
+        },
+        None,
+    )
+    .unwrap();
+    assert_eq!(result.shape(), [4, 2, 4, 1]);
+    for head in 0..4 {
+        for column in 0..2 {
+            for row in 0..4 {
+                assert_eq!(
+                    result.values()[row + 4 * (column + 2 * head)],
+                    (head / 2 * 4 + row + 1) as f32 * 8.0
+                );
+            }
+        }
+    }
+    let live = arbiter.snapshot(&key).unwrap();
+    assert_eq!(live.reservations, 1);
+    assert_eq!(live.per_class[&AllocationClass::Output], 128);
+    assert_eq!(live.per_class[&AllocationClass::Scratch], SCRATCH_BYTES);
+    assert!(a[0].is_nan() && a[11].is_nan() && b[0].is_nan() && b[65].is_nan());
+    assert_eq!(result.source(), source);
+    drop(result);
+    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+}
+
+#[test]
+fn foreign_source_stride_span_overflow_and_ar1_are_closed() {
+    let source = SourceIdentity::for_new_allocation().unwrap();
+    let a = [1.0; 32];
+    let b = [1.0; 16];
+    let (arbiter, key, owner) = budget(4096);
+    let valid = request(source, &a, &b);
+    let mut cases = Vec::new();
+    let mut r = valid;
+    r.right.source = SourceIdentity::for_new_allocation().unwrap();
+    cases.push(r);
+    let mut r = valid;
+    r.left.strides[1] = 7;
+    cases.push(r);
+    let mut r = valid;
+    r.left.offset = 1;
+    cases.push(r);
+    let mut r = valid;
+    r.left.offset = usize::MAX;
+    cases.push(r);
+    let mut r = valid;
+    r.left.strides[3] = usize::MAX;
+    r.left.shape[3] = 2;
+    cases.push(r);
+    let mut r = valid;
+    r.right.shape[1] = 1;
+    r.right.strides = [1, 8, 8, 8];
+    cases.push(r);
+    let mut r = valid;
+    r.right.strides[3] = 17;
+    cases.push(r);
+    for r in cases {
+        assert!(
+            execute(
+                r,
+                BulkF32Admission {
+                    arbiter: &arbiter,
+                    resource: &key,
+                    owner: &owner
+                },
+                None
+            )
+            .is_err()
+        );
+        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+    }
+}
+
+#[test]
+fn actual_capacity_pre_cancel_and_mid_output_cancel_release_only_private_storage() {
+    let source = SourceIdentity::for_new_allocation().unwrap();
+    let a = [1.0; 32];
+    let b = [1.0; 16];
+    let (arbiter, key, owner) = budget(32 + SCRATCH_BYTES - 1);
+    assert!(matches!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            None
+        ),
+        Err(Error::Limit(_))
+    ));
+    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+    let (arbiter, key, owner) = budget(4096);
+    let flag = Arc::new(AtomicBool::new(true));
+    assert!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            Some(&flag)
+        )
+        .is_err()
+    );
+    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+    flag.store(false, Ordering::Release);
+    let observed_arbiter = arbiter.clone();
+    let observed_key = key.clone();
+    let observed_flag = flag.clone();
+    AFTER_FIRST_OUTPUT.with(|hook| {
+        *hook.borrow_mut() = Some(Box::new(move || {
+            assert_eq!(
+                observed_arbiter.snapshot(&observed_key).unwrap().reserved,
+                32 + SCRATCH_BYTES
+            );
+            observed_flag.store(true, Ordering::Release);
+        }))
+    });
+    assert!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            Some(&flag)
+        )
+        .is_err()
+    );
+    let final_state = arbiter.snapshot(&key).unwrap();
+    assert_eq!(final_state.reserved, 0);
+    assert!(!final_state.fenced);
+}
+
+#[test]
+fn nonfinite_used_words_and_finite_input_overflow_refuse_without_outputs() {
+    let source = SourceIdentity::for_new_allocation().unwrap();
+    let (arbiter, key, owner) = budget(4096);
+    let mut a = [1.0; 32];
+    let b = [2.0; 16];
+    a[19] = f32::INFINITY;
+    assert!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            None
+        )
+        .is_err()
+    );
+    a.fill(f32::MAX);
+    assert!(
+        execute(
+            request(source, &a, &b),
+            BulkF32Admission {
+                arbiter: &arbiter,
+                resource: &key,
+                owner: &owner
+            },
+            None
+        )
+        .is_err()
+    );
+    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
+}
diff --git a/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs b/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs
@@ -0,0 +1,364 @@
+//! Closed GGML d812350 Haswell tinyBLAS F32 schedule, no model/plan selection.
+//!
+//! One Vec8 FMA chain per output and the original even/odd horizontal tree.
+//! Default GGUF four-Vec8 projections/attention and source profiles are unchanged.
+//! Source: ggml/src/ggml-cpu/llamafile/sgemm.cpp:252-270,559-730,3908-3934.
+
+/*
+MIT License
+
+Copyright (c) 2023-2026 The ggml authors
+
+Permission is hereby granted, free of charge, to any person obtaining a copy
+of this software and associated documentation files (the "Software"), to deal
+in the Software without restriction, including without limitation the rights
+to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
+copies of the Software, and to permit persons to whom the Software is
+furnished to do so, subject to the following conditions:
+
+The above copyright notice and this permission notice shall be included in all
+copies or substantial portions of the Software.
+
+THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
+IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
+FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
+AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
+LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
+OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
+SOFTWARE.
+*/
+
+use crate::accelerator::SourceIdentity;
+use crate::{Error, Result};
+use std::sync::atomic::{AtomicBool, Ordering};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
+    ResourceOwner,
+};
+
+const MAX_SPAN_WORDS: usize = 8 * 1024 * 1024;
+const MAX_OUTPUT_WORDS: usize = 1024 * 1024;
+const MAX_FMAS: usize = 128 * 1024 * 1024;
+const SCRATCH_BYTES: u64 = 256;
+
+#[cfg(test)]
+std::thread_local! {
+    static AFTER_FIRST_OUTPUT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
+        const { std::cell::RefCell::new(None) };
+}
+
+/// Distinct closed arithmetic contract; no executable model/profile capability.
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum BulkF32Profile {
+    GgmlD812350HaswellTinyBlas,
+}
+
+/// Original F32 axes `[inner,rows,head,sequence]`; strides/offset are in words.
+/// Borrowed raw activations/weights retain their caller's owner and admission.
+/// This descriptor validates a view, not its model-file SHA or source issuer.
+#[derive(Clone, Copy)]
+pub struct F32View<'a> {
+    pub source: SourceIdentity,
+    pub name: &'a str,
+    pub values: &'a [f32],
+    pub offset: usize,
+    pub shape: [usize; 4],
+    pub strides: [usize; 4],
+}
+
+/// Same source for both operands. B is the original contiguous RHS; A may
+/// retain a strided V-cache view. All broadcast heads are independently read.
+#[derive(Clone, Copy)]
+pub struct BulkF32Request<'a> {
+    pub profile: BulkF32Profile,
+    pub source: SourceIdentity,
+    pub left: F32View<'a>,
+    pub right: F32View<'a>,
+}
+
+/// Existing host ledger, not a new pool or a claim about all host RAM.
+pub struct BulkF32Admission<'a> {
+    pub arbiter: &'a ResourceArbiter,
+    pub resource: &'a PhysicalDeviceKey,
+    pub owner: &'a ResourceOwner,
+}
+
+/// Payload drops before its original output/scratch ticket. No unleased Vec
+/// escape or state publication; synchronous borrowed inputs outlive execution.
+pub struct BulkF32Completed {
+    values: Vec<f32>,
+    source: SourceIdentity,
+    shape: [usize; 4],
+    _lease: GlobalResourceLease,
+}
+impl BulkF32Completed {
+    pub fn values(&self) -> &[f32] {
+        &self.values
+    }
+    pub fn source(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn shape(&self) -> [usize; 4] {
+        self.shape
+    }
+    pub fn profile(&self) -> BulkF32Profile {
+        BulkF32Profile::GgmlD812350HaswellTinyBlas
+    }
+}
+
+fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
+    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        return Err(Error::State("Haswell bulk F32 cancelled".into()));
+    }
+    Ok(())
+}
+
+fn checked_product(values: &[usize]) -> Result<usize> {
+    values.iter().try_fold(1usize, |acc, &next| {
+        acc.checked_mul(next)
+            .ok_or_else(|| Error::Limit("Haswell bulk F32 geometry overflow".into()))
+    })
+}
+
+impl F32View<'_> {
+    fn validate(self, source: SourceIdentity, cancelled: Option<&AtomicBool>) -> Result<()> {
+        not_cancelled(cancelled)?;
+        if !source.is_bound()
+            || self.source != source
+            || self.name.is_empty()
+            || self.name.len() > 256
+            || self.name.contains('\0')
+            || self.shape.contains(&0)
+            || self.shape[0] > 8192
+            || self.shape[1] > 4096
+            || self.shape[2] > 64
+            || self.shape[3] > 64
+            || self.strides[0] != 1
+        {
+            return Err(Error::Malformed(
+                "Haswell bulk F32 source/view geometry".into(),
+            ));
+        }
+        // Original matrix rows do not overlap. Axis3's declared stride is
+        // checked even when that dimension has size1, preserving raw geometry.
+        let mut previous = self.shape[0];
+        for (&stride, &size) in self.strides.iter().zip(&self.shape).skip(1) {
+            if stride < previous {
+                return Err(Error::Malformed(
+                    "Haswell bulk F32 overlapping strides".into(),
+                ));
+            }
+            previous = stride
+                .checked_mul(size)
+                .ok_or_else(|| Error::Limit("Haswell bulk F32 stride overflow".into()))?;
+        }
+        let end = self
+            .shape
+            .iter()
+            .zip(self.strides)
+            .try_fold(self.offset, |at, (&size, stride)| {
+                (size - 1)
+                    .checked_mul(stride)
+                    .and_then(|n| at.checked_add(n))
+                    .ok_or_else(|| Error::Limit("Haswell bulk F32 view span overflow".into()))
+            })?
+            .checked_add(1)
+            .ok_or_else(|| Error::Limit("Haswell bulk F32 view end overflow".into()))?;
+        if end > self.values.len() || self.values.len() > MAX_SPAN_WORDS {
+            return Err(Error::Limit(
+                "Haswell bulk F32 view exceeds backing/bound".into(),
+            ));
+        }
+        let words = checked_product(&self.shape)?;
+        if words > MAX_OUTPUT_WORDS {
+            return Err(Error::Limit(
+                "Haswell bulk F32 logical operand bound".into(),
+            ));
+        }
+        // Only actual logical words are read; storage gaps/canaries remain
+        // outside the arithmetic and may contain arbitrary IEEE754 words.
+        for sequence in 0..self.shape[3] {
+            for head in 0..self.shape[2] {
+                for row in 0..self.shape[1] {
+                    not_cancelled(cancelled)?;
+                    let start = self.at(0, row, head, sequence);
+                    if self.values[start..start + self.shape[0]]
+                        .iter()
+                        .any(|x| !x.is_finite())
+                    {
+                        return Err(Error::Malformed(
+                            "nonfinite Haswell bulk F32 operand".into(),
+                        ));
+                    }
+                }
+            }
+        }
+        Ok(())
+    }
+    fn at(self, inner: usize, row: usize, head: usize, sequence: usize) -> usize {
+        // Whole view's final index and products were checked before access.
+        self.offset
+            + inner
+            + row * self.strides[1]
+            + head * self.strides[2]
+            + sequence * self.strides[3]
+    }
+}
+
+fn validate(
+    request: BulkF32Request<'_>,
+    cancelled: Option<&AtomicBool>,
+) -> Result<([usize; 4], usize)> {
+    not_cancelled(cancelled)?;
+    let BulkF32Profile::GgmlD812350HaswellTinyBlas = request.profile;
+    let a = request.left;
+    let b = request.right;
+    a.validate(request.source, cancelled)?;
+    b.validate(request.source, cancelled)?;
+    if a.shape[0] != b.shape[0]
+        || !(2..=64).contains(&b.shape[1])
+        || !a.shape[1].is_multiple_of(4)
+        || !b.shape[2].is_multiple_of(a.shape[2])
+        || !b.shape[3].is_multiple_of(a.shape[3])
+    {
+        // n<2 selects the original vec-dot path, never this bulk schedule.
+        return Err(Error::Unsupported(
+            "unsupported Haswell tinyBLAS F32 shape".into(),
+        ));
+    }
+    let mut stride = 1usize;
+    for (&actual, &size) in b.strides.iter().zip(&b.shape) {
+        if actual != stride {
+            return Err(Error::Malformed(
+                "Haswell tinyBLAS requires original contiguous RHS".into(),
+            ));
+        }
+        stride = stride
+            .checked_mul(size)
+            .ok_or_else(|| Error::Limit("Haswell bulk F32 contiguous span overflow".into()))?;
+    }
+    let shape = [a.shape[1], b.shape[1], b.shape[2], b.shape[3]];
+    let count = checked_product(&shape)?;
+    if count > MAX_OUTPUT_WORDS || count.checked_mul(a.shape[0]).is_none_or(|n| n > MAX_FMAS) {
+        return Err(Error::Limit("Haswell bulk F32 output/work bound".into()));
+    }
+    not_cancelled(cancelled)?;
+    Ok((shape, count))
+}
+
+fn dot(a: &[f32], b: &[f32]) -> f32 {
+    let mut lanes = [0.0f32; 8];
+    let full = a.len() & !7usize;
+    for base in (0..full).step_by(8) {
+        for (lane, sum) in lanes.iter_mut().enumerate() {
+            *sum = a[base + lane].mul_add(b[base + lane], *sum);
+        }
+    }
+    if full != a.len() {
+        // tinyBLAS load_partial zero-fills BOTH remaining Vec8 operands.
+        // Execute those +0*+0 FMAs too; skipping them changes zero signs.
+        for (lane, sum) in lanes.iter_mut().enumerate() {
+            let at = full + lane;
+            let (x, y) = if at < a.len() {
+                (a[at], b[at])
+            } else {
+                (0.0, 0.0)
+            };
+            *sum = x.mul_add(y, *sum);
+        }
+    }
+    let low = [
+        lanes[0] + lanes[4],
+        lanes[1] + lanes[5],
+        lanes[2] + lanes[6],
+        lanes[3] + lanes[7],
+    ];
+    // Actual hsum(__m128): add(movehl), then add_ss(movehdup).
+    // This is (0+2)+(1+3), not vec_dot's (0+1)+(2+3).
+    (low[0] + low[2]) + (low[1] + low[3])
+}
+
+/// Pure source-derived primitive. Provider/model/session/checkpoint eligibility
+/// is unchanged. Caller authenticates operands and retains input reservations;
+/// this function admits only its own private result and bounded stack scratch.
+pub fn execute(
+    request: BulkF32Request<'_>,
+    admission: BulkF32Admission<'_>,
+    cancelled: Option<&AtomicBool>,
+) -> Result<BulkF32Completed> {
+    let (shape, count) = validate(request, cancelled)?;
+    if admission.resource.backend != "host" {
+        return Err(Error::Unsupported(
+            "Haswell bulk F32 requires existing host admission".into(),
+        ));
+    }
+    let output_bytes = count
+        .checked_mul(4)
+        .ok_or_else(|| Error::Limit("Haswell bulk F32 output bytes overflow".into()))?;
+    let lease = admission.arbiter.admit(
+        admission.resource,
+        admission.owner.clone(),
+        &[
+            AllocationRequest {
+                kind: AllocationClass::Output,
+                bytes: output_bytes as u64,
+            },
+            AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: SCRATCH_BYTES,
+            },
+        ],
+    )?;
+    not_cancelled(cancelled)?;
+    let mut values = Vec::new();
+    values
+        .try_reserve_exact(count)
+        .map_err(|_| Error::Limit("Haswell bulk F32 output allocation".into()))?;
+    if values.capacity() != count {
+        return Err(Error::Limit(
+            "Haswell bulk F32 unexpected output capacity".into(),
+        ));
+    }
+    let a = request.left;
+    let b = request.right;
+    let ratio2 = b.shape[2] / a.shape[2];
+    let ratio3 = b.shape[3] / a.shape[3];
+    for sequence in 0..shape[3] {
+        for head in 0..shape[2] {
+            for column in 0..shape[1] {
+                not_cancelled(cancelled)?;
+                let br = b.at(0, column, head, sequence);
+                for row in 0..shape[0] {
+                    not_cancelled(cancelled)?;
+                    let ar = a.at(0, row, head / ratio2, sequence / ratio3);
+                    let value = dot(
+                        &a.values[ar..ar + a.shape[0]],
+                        &b.values[br..br + b.shape[0]],
+                    );
+                    if !value.is_finite() {
+                        return Err(Error::Limit("Haswell bulk F32 nonfinite result".into()));
+                    }
+                    values.push(value);
+                    #[cfg(test)]
+                    if values.len() == 1 {
+                        AFTER_FIRST_OUTPUT.with(|hook| {
+                            if let Some(hook) = hook.borrow_mut().take() {
+                                hook();
+                            }
+                        });
+                    }
+                }
+            }
+        }
+    }
+    not_cancelled(cancelled)?;
+    Ok(BulkF32Completed {
+        values,
+        source: request.source,
+        shape,
+        _lease: lease,
+    })
+}
+
+#[cfg(test)]
+mod tests;
diff --git a/crates/zerogravity-cpu/src/math.rs b/crates/zerogravity-cpu/src/math.rs
--- a/crates/zerogravity-cpu/src/math.rs
+++ b/crates/zerogravity-cpu/src/math.rs
@@ -368,3 +368,6 @@
         assert_eq!(ggml_source_silu(0.0), 0.0);
     }
 }
+
+/// Explicit pinned GGML bulk-F32 diagnostic; default source math is unchanged.
+pub mod haswell_bulk_f32;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/GPU_REFUSAL.patch

Original UTF-8 bytes: 2158; SHA256 `8d24045e9855533e4793721122c74b0e2d380a0bd83bec2265f5a0b000a4ea2f`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/resident_program/batch.rs b/crates/zerogravity-gpu/src/resident_program/batch.rs
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -928,7 +928,9 @@
         let scratch = |i: usize| &self.inner.scratch[i];
         unsafe {
             match &node.operation {
-                PlanOperation::TypedEmbed { .. }
+                PlanOperation::TypedBulkF32Input { .. }
+                | PlanOperation::TypedBulkF32Matmul { .. }
+                | PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
                 | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
diff --git a/crates/zerogravity-gpu/src/resident_program.rs b/crates/zerogravity-gpu/src/resident_program.rs
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -345,7 +345,9 @@
             || plan.nodes().iter().any(|node| {
                 matches!(
                     node.operation,
-                    PlanOperation::TypedEmbed { .. }
+                    PlanOperation::TypedBulkF32Input { .. }
+                        | PlanOperation::TypedBulkF32Matmul { .. }
+                        | PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
@@ -1428,7 +1430,9 @@
         // bound every offset/count. Parent retains the complete original cohort.
         unsafe {
             match &node.operation {
-                PlanOperation::TypedEmbed { .. }
+                PlanOperation::TypedBulkF32Input { .. }
+                | PlanOperation::TypedBulkF32Matmul { .. }
+                | PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
                 | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 53597; SHA256 `79c065c2669ba31003ab65594af82b8f4dea45d2e652a978db323c0960afab0f`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs b/crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs
@@ -0,0 +1,193 @@
+use super::*;
+fn span() -> TeacherSpan {
+    TeacherSpan::new(
+        SourceIdentity::for_new_allocation().unwrap(),
+        0,
+        8448,
+        248320,
+        &(0..64).collect::<Vec<_>>(),
+    )
+    .unwrap()
+}
+fn matrix(span: &TeacherSpan) -> BulkF32Matrix {
+    let left = BulkF32Operand::external(
+        span.clone(),
+        BulkF32Side::Left,
+        "actual external V",
+        BulkF32Layout::new([256, 256, 2, 1], [1, 8448, 2162688, 4325376], 0, 4317184).unwrap(),
+    )
+    .unwrap();
+    let right = BulkF32Operand::external(
+        span.clone(),
+        BulkF32Side::Right,
+        "actual external P",
+        BulkF32Layout::dense([256, 64, 8, 1]).unwrap(),
+    )
+    .unwrap();
+    BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1).unwrap()
+}
+#[test]
+fn actual_temporal_n_is_not_cohort_width_and_full_k_is_preserved() {
+    let span = span();
+    let operation = matrix(&span);
+    assert_eq!(operation.left().layout().shape(), [256, 256, 2, 1]);
+    assert_eq!(operation.right().layout().shape(), [256, 64, 8, 1]);
+    assert_eq!(operation.output().shape(), [256, 64, 8, 1]);
+    assert_eq!(operation.output().backing_words(), 131072);
+    // Real N2 is a different declared teacher span, never a B2 request label.
+    let wrong = BulkF32Operand::external(
+        span.clone(),
+        BulkF32Side::Right,
+        "P",
+        BulkF32Layout::dense([256, 2, 8, 1]).unwrap(),
+    );
+    assert!(wrong.is_err());
+    assert!(TeacherSpan::new(span.source(), 0, 8448, 248320, &[11]).is_err());
+    assert!(TeacherSpan::new(span.source(), 8447, 8448, 248320, &[11, 353]).is_err());
+}
+#[test]
+fn same_ids_source_names_do_not_merge_issuers_and_wrong_full_k_refuses() {
+    let span = span();
+    let operation = matrix(&span);
+    let foreign = TeacherSpan::new(
+        span.source(),
+        span.base(),
+        span.capacity(),
+        248320,
+        span.ids(),
+    )
+    .unwrap();
+    let right = BulkF32Operand::external(
+        foreign,
+        BulkF32Side::Right,
+        "actual external P",
+        operation.right().layout(),
+    )
+    .unwrap();
+    assert!(
+        BulkF32Matrix::new(
+            operation.left().clone(),
+            right,
+            BulkF32Contract::HaswellTinyBlasF32V1
+        )
+        .is_err()
+    );
+    let right = BulkF32Operand::external(
+        span.clone(),
+        BulkF32Side::Right,
+        "P",
+        BulkF32Layout::dense([128, 64, 8, 1]).unwrap(),
+    )
+    .unwrap();
+    assert!(
+        BulkF32Matrix::new(
+            operation.left().clone(),
+            right,
+            BulkF32Contract::HaswellTinyBlasF32V1
+        )
+        .is_err()
+    );
+}
+#[test]
+fn real_nonzero_offsets_and_gaps_need_full_checked_storage() {
+    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 47).is_ok());
+    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 46).is_err());
+    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 7, 28, 28], 0, 32).is_err());
+    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], usize::MAX, 64).is_err());
+    assert!(BulkF32Layout::dense([8192, 4096, 8, 1]).is_err());
+    let span = span();
+    assert!(
+        BulkF32Operand::external(
+            span,
+            BulkF32Side::Right,
+            "strided P",
+            BulkF32Layout::new([8, 64, 1, 1], [1, 12, 768, 768], 0, 764).unwrap()
+        )
+        .is_err()
+    );
+}
+#[test]
+fn same_builder_three_nodes_stay_pure_and_legacy_use_or_foreign_values_refuse() {
+    let span = span();
+    let operation = matrix(&span);
+    let mut b =
+        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
+            .unwrap();
+    let left = b.bulk_f32_input(operation.left()).unwrap();
+    let right = b.bulk_f32_input(operation.right()).unwrap();
+    let mut other =
+        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
+            .unwrap();
+    let foreign = other.bulk_f32_input(operation.right()).unwrap();
+    assert!(b.bulk_f32_matmul(&operation, &left, &foreign).is_err());
+    assert!(b.silu("legacy", &right, None).is_err());
+    let output = b.bulk_f32_matmul(&operation, &left, &right).unwrap();
+    let slice = b.finish_operator_slice(output).unwrap();
+    assert_eq!(slice.plan().nodes().len(), 3);
+    assert!(slice.plan().required_domains().is_empty());
+    assert!(slice.plan().proposed_domains().is_empty());
+    assert_eq!(slice.plan().frontier_delta(), 0);
+    assert!(slice.plan().logits().is_none());
+    assert!(slice.plan().is_operator_slice());
+    let slot = StateSlot::Conv(ConvDomain(0));
+    let mut full = PlanBuilder::new(span.source(), 8448, vec![slot], false).unwrap();
+    assert!(full.bulk_f32_input(operation.left()).is_err());
+}
+
+struct Original {
+    identity: SourceIdentity,
+    kind: QuantKind,
+    shape: [u64; 2],
+    bytes: Vec<u8>,
+}
+impl TensorSource for Original {
+    fn source_identity(&self) -> SourceIdentity {
+        self.identity
+    }
+    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
+        if name != "original.weight" {
+            return Err(Error::Malformed("unknown original name".into()));
+        }
+        Ok(SourceTensorRef {
+            kind: self.kind,
+            shape: &self.shape,
+            packed_bytes: &self.bytes,
+        })
+    }
+}
+#[test]
+fn whole_raw_source_matrix_keeps_axes_dtype_extent_and_exact_issuer() {
+    let original = Original {
+        identity: SourceIdentity::for_new_allocation().unwrap(),
+        kind: QuantKind::F32,
+        shape: [8, 4],
+        bytes: vec![0; 128],
+    };
+    let span = TeacherSpan::new(original.identity, 0, 8, 248320, &[11, 353]).unwrap();
+    let left = BulkF32Operand::source_matrix(span.clone(), &original, "original.weight").unwrap();
+    assert_eq!(left.layout().shape(), [8, 4, 1, 1]);
+    let BulkF32Origin::SourceMatrix(raw) = left.origin() else {
+        panic!("original raw source expected")
+    };
+    assert_eq!(raw.axes, MatrixAxes::InputOutput);
+    raw.validate_source(&original).unwrap();
+    let foreign = Original {
+        identity: SourceIdentity::for_new_allocation().unwrap(),
+        ..original
+    };
+    assert!(BulkF32Operand::source_matrix(span.clone(), &foreign, "original.weight").is_err());
+    let bf16 = Original {
+        identity: span.source(),
+        kind: QuantKind::Bf16,
+        shape: [8, 4],
+        bytes: vec![0; 64],
+    };
+    assert!(BulkF32Operand::source_matrix(span.clone(), &bf16, "original.weight").is_err());
+    let short = Original {
+        identity: span.source(),
+        kind: QuantKind::F32,
+        shape: [8, 4],
+        bytes: vec![0; 124],
+    };
+    assert!(BulkF32Operand::source_matrix(span, &short, "original.weight").is_err());
+}
diff --git a/crates/zerogravity-core/src/accelerator/program/bulk_f32.rs b/crates/zerogravity-core/src/accelerator/program/bulk_f32.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/bulk_f32.rs
@@ -0,0 +1,410 @@
+//! Closed temporal matrix metadata; no session, model route or producer permit.
+use super::*;
+
+/// ONE consecutive teacher sequence. Clones share one immutable issuer; equal
+/// text/IDs from another constructor do not silently become the same scope.
+#[derive(Clone, Debug)]
+pub struct TeacherSpan(Arc<TeacherSpanData>);
+#[derive(Debug)]
+struct TeacherSpanData {
+    source: SourceIdentity,
+    base: usize,
+    capacity: usize,
+    ids: Arc<[u32]>,
+}
+impl PartialEq for TeacherSpan {
+    fn eq(&self, other: &Self) -> bool {
+        Arc::ptr_eq(&self.0, &other.0)
+    }
+}
+impl Eq for TeacherSpan {}
+impl TeacherSpan {
+    /// Metadata only; these IDs are not a current SessionAuthority/StateRoot.
+    pub fn new(
+        source: SourceIdentity,
+        base: usize,
+        capacity: usize,
+        vocabulary: u64,
+        ids: &[u32],
+    ) -> Result<Self> {
+        if !source.is_bound()
+            || !(2..=64).contains(&ids.len())
+            || capacity == 0
+            || capacity > 8448
+            || base.checked_add(ids.len()).is_none_or(|end| end > capacity)
+            || vocabulary == 0
+            || vocabulary > u64::from(u32::MAX) + 1
+            || ids.iter().any(|&id| u64::from(id) >= vocabulary)
+        {
+            return Err(Error::Malformed(
+                "closed teacher span source/geometry/IDs".into(),
+            ));
+        }
+        Ok(Self(Arc::new(TeacherSpanData {
+            source,
+            base,
+            capacity,
+            ids: Arc::from(ids),
+        })))
+    }
+    pub fn source(&self) -> SourceIdentity {
+        self.0.source
+    }
+    pub fn base(&self) -> usize {
+        self.0.base
+    }
+    pub fn capacity(&self) -> usize {
+        self.0.capacity
+    }
+    pub fn ids(&self) -> &[u32] {
+        &self.0.ids
+    }
+    pub fn len(&self) -> usize {
+        self.0.ids.len()
+    }
+    pub fn is_empty(&self) -> bool {
+        false
+    }
+}
+
+/// Actual F32 storage axes `[full_K,M_or_N,head,sequence]`, word strides.
+/// Offset/backing describe THIS allocation; original capture provenance stays
+/// separate if transport reconstructed gaps. Full K is never split/reduced.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub struct BulkF32Layout {
+    shape: [usize; 4],
+    strides: [usize; 4],
+    offset: usize,
+    backing: usize,
+}
+impl BulkF32Layout {
+    pub fn new(
+        shape: [usize; 4],
+        strides: [usize; 4],
+        offset: usize,
+        backing: usize,
+    ) -> Result<Self> {
+        let value = Self {
+            shape,
+            strides,
+            offset,
+            backing,
+        };
+        value.validate()?;
+        Ok(value)
+    }
+    pub fn dense(shape: [usize; 4]) -> Result<Self> {
+        let mut strides = [1usize; 4];
+        for i in 1..4 {
+            strides[i] = strides[i - 1]
+                .checked_mul(shape[i - 1])
+                .ok_or_else(|| Error::Limit("bulk dense stride".into()))?;
+        }
+        let backing = strides[3]
+            .checked_mul(shape[3])
+            .ok_or_else(|| Error::Limit("bulk dense words".into()))?;
+        Self::new(shape, strides, 0, backing)
+    }
+    pub fn shape(self) -> [usize; 4] {
+        self.shape
+    }
+    pub fn strides(self) -> [usize; 4] {
+        self.strides
+    }
+    pub fn offset(self) -> usize {
+        self.offset
+    }
+    pub fn backing_words(self) -> usize {
+        self.backing
+    }
+    pub fn logical_words(self) -> Result<usize> {
+        self.shape.iter().try_fold(1usize, |n, &x| {
+            n.checked_mul(x)
+                .ok_or_else(|| Error::Limit("bulk logical words".into()))
+        })
+    }
+    pub fn validate(self) -> Result<()> {
+        if self.shape.contains(&0)
+            || self.shape[0] > 8192
+            || self.shape[1] > 4096
+            || self.shape[2] > 64
+            || self.shape[3] != 1
+            || self.strides[0] != 1
+            || self.backing == 0
+            || self.backing > 8 * 1024 * 1024
+        {
+            return Err(Error::Unsupported("closed bulk F32 axes/backing".into()));
+        }
+        let mut previous = self.shape[0];
+        for (&stride, &size) in self.strides.iter().zip(&self.shape).skip(1) {
+            if stride < previous {
+                return Err(Error::Malformed("bulk overlapping rows/axes".into()));
+            }
+            previous = stride
+                .checked_mul(size)
+                .ok_or_else(|| Error::Limit("bulk stride product".into()))?;
+        }
+        let end = self
+            .shape
+            .iter()
+            .zip(self.strides)
+            .try_fold(self.offset, |n, (&size, stride)| {
+                (size - 1)
+                    .checked_mul(stride)
+                    .and_then(|x| n.checked_add(x))
+                    .ok_or_else(|| Error::Limit("bulk full span".into()))
+            })?
+            .checked_add(1)
+            .ok_or_else(|| Error::Limit("bulk span end".into()))?;
+        if end > self.backing || self.logical_words()? > 1024 * 1024 {
+            return Err(Error::Limit("bulk operand span/logical bound".into()));
+        }
+        Ok(())
+    }
+    pub fn is_dense(self) -> bool {
+        Self::dense(self.shape).is_ok_and(|dense| dense == self)
+    }
+}
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum BulkF32Side {
+    Left,
+    Right,
+}
+#[derive(Clone, Debug, PartialEq, Eq)]
+pub enum BulkF32Origin {
+    /// Caller-supplied pure operator input, explicitly NOT a model/state producer.
+    ExternalObserved { name: String },
+    /// Exact whole original raw learned matrix, no dequantization/retag/cache.
+    SourceMatrix(SourceTensor),
+}
+#[derive(Clone, Debug, PartialEq, Eq)]
+pub struct BulkF32Operand {
+    span: TeacherSpan,
+    side: BulkF32Side,
+    layout: BulkF32Layout,
+    origin: BulkF32Origin,
+}
+impl BulkF32Operand {
+    pub fn external(
+        span: TeacherSpan,
+        side: BulkF32Side,
+        name: &str,
+        layout: BulkF32Layout,
+    ) -> Result<Self> {
+        if name.is_empty() || name.len() > 256 || name.contains('\0') {
+            return Err(Error::Malformed("bulk external input label bound".into()));
+        }
+        let value = Self {
+            span,
+            side,
+            layout,
+            origin: BulkF32Origin::ExternalObserved { name: name.into() },
+        };
+        value.validate()?;
+        Ok(value)
+    }
+    pub fn source_matrix(span: TeacherSpan, source: &dyn TensorSource, name: &str) -> Result<Self> {
+        if name.is_empty() || name.len() > 256 || name.contains('\0') {
+            return Err(Error::Malformed("bulk learned input label bound".into()));
+        }
+        let tensor = SourceTensor::from_source(source, name, MatrixAxes::InputOutput)?;
+        if tensor.kind != QuantKind::F32 || tensor.source != span.source() {
+            return Err(Error::Unsupported(
+                "bulk original F32 source matrix only".into(),
+            ));
+        }
+        let layout = BulkF32Layout::dense([tensor.columns, tensor.rows, 1, 1])?;
+        let value = Self {
+            span,
+            side: BulkF32Side::Left,
+            layout,
+            origin: BulkF32Origin::SourceMatrix(tensor),
+        };
+        value.validate()?;
+        Ok(value)
+    }
+    pub fn span(&self) -> &TeacherSpan {
+        &self.span
+    }
+    pub fn side(&self) -> BulkF32Side {
+        self.side
+    }
+    pub fn layout(&self) -> BulkF32Layout {
+        self.layout
+    }
+    pub fn origin(&self) -> &BulkF32Origin {
+        &self.origin
+    }
+    pub fn name(&self) -> &str {
+        match &self.origin {
+            BulkF32Origin::ExternalObserved { name } => name,
+            BulkF32Origin::SourceMatrix(tensor) => &tensor.name,
+        }
+    }
+    pub fn validate(&self) -> Result<()> {
+        self.layout.validate()?;
+        if self.side == BulkF32Side::Right
+            && (self.layout.shape[1] != self.span.len() || !self.layout.is_dense())
+        {
+            return Err(Error::Unsupported(
+                "actual teacher N and original contiguous RHS required".into(),
+            ));
+        }
+        if let BulkF32Origin::SourceMatrix(tensor) = &self.origin {
+            if self.side != BulkF32Side::Left
+                || tensor.source != self.span.source()
+                || tensor.kind != QuantKind::F32
+                || tensor.axes != MatrixAxes::InputOutput
+                || self.layout != BulkF32Layout::dense([tensor.columns, tensor.rows, 1, 1])?
+            {
+                return Err(Error::State("bulk whole learned descriptor/axes".into()));
+            }
+        }
+        Ok(())
+    }
+}
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum BulkF32Contract {
+    HaswellTinyBlasF32V1,
+}
+#[derive(Clone, Debug, PartialEq, Eq)]
+pub struct BulkF32Matrix {
+    left: BulkF32Operand,
+    right: BulkF32Operand,
+    contract: BulkF32Contract,
+    output: BulkF32Layout,
+}
+impl BulkF32Matrix {
+    pub fn new(
+        left: BulkF32Operand,
+        right: BulkF32Operand,
+        contract: BulkF32Contract,
+    ) -> Result<Self> {
+        left.validate()?;
+        right.validate()?;
+        let a = left.layout.shape;
+        let b = right.layout.shape;
+        if left.side != BulkF32Side::Left
+            || right.side != BulkF32Side::Right
+            || left.span != right.span
+            || a[0] != b[0]
+            || !a[1].is_multiple_of(4)
+            || !b[2].is_multiple_of(a[2])
+        {
+            return Err(Error::Unsupported(
+                "closed temporal matrix issuer/fullK/head broadcast".into(),
+            ));
+        }
+        let output = BulkF32Layout::dense([a[1], b[1], b[2], 1])?;
+        if output
+            .logical_words()?
+            .checked_mul(a[0])
+            .is_none_or(|n| n > 128 * 1024 * 1024)
+        {
+            return Err(Error::Limit("bulk complete K work bound".into()));
+        }
+        Ok(Self {
+            left,
+            right,
+            contract,
+            output,
+        })
+    }
+    pub fn left(&self) -> &BulkF32Operand {
+        &self.left
+    }
+    pub fn right(&self) -> &BulkF32Operand {
+        &self.right
+    }
+    pub fn contract(&self) -> BulkF32Contract {
+        self.contract
+    }
+    pub fn output(&self) -> BulkF32Layout {
+        self.output
+    }
+    pub fn source(&self) -> SourceIdentity {
+        self.left.span.source()
+    }
+}
+
+fn check_builder(builder: &PlanBuilder, source: SourceIdentity) -> Result<()> {
+    if !builder.operator_slice
+        || builder.profile != ProgramNumericalProfile::GgufSourceV1
+        || builder.source != source
+    {
+        return Err(Error::Unsupported(
+            "bulk F32 requires explicit state-free GGUF operator slice".into(),
+        ));
+    }
+    Ok(())
+}
+pub(super) fn input_precision(
+    builder: &PlanBuilder,
+    operand: &BulkF32Operand,
+) -> Result<ActivationPrecision> {
+    check_builder(builder, operand.span.source())?;
+    operand.validate()?;
+    Ok(ActivationPrecision::F32)
+}
+pub(super) fn matrix_precision(
+    builder: &PlanBuilder,
+    matrix: &BulkF32Matrix,
+    inputs: &[PlanValue],
+) -> Result<ActivationPrecision> {
+    check_builder(builder, matrix.source())?;
+    if inputs.len() != 2 {
+        return Err(Error::Malformed("bulk input arity".into()));
+    }
+    for (input, expected) in inputs.iter().zip([matrix.left(), matrix.right()]) {
+        let producer = builder
+            .nodes
+            .iter()
+            .find(|node| node.outputs.contains(&input.id));
+        if input.precision != ActivationPrecision::F32
+            || input.elements != expected.layout.backing_words()
+            || !matches!(producer.map(|node|&node.operation),Some(PlanOperation::TypedBulkF32Input{operand})if operand==expected)
+        {
+            return Err(Error::State(
+                "bulk exact input producer/layout/span membership".into(),
+            ));
+        }
+    }
+    Ok(ActivationPrecision::F32)
+}
+pub(super) fn record_input(
+    builder: &mut PlanBuilder,
+    operand: &BulkF32Operand,
+) -> Result<PlanValue> {
+    input_precision(builder, operand)?;
+    builder.one(
+        operand.name(),
+        PlanOperation::TypedBulkF32Input {
+            operand: operand.clone(),
+        },
+        &[],
+        ValueShape::Vector(operand.layout.backing_words()),
+        None,
+    )
+}
+pub(super) fn record_matrix(
+    builder: &mut PlanBuilder,
+    matrix: &BulkF32Matrix,
+    left: &PlanValue,
+    right: &PlanValue,
+) -> Result<PlanValue> {
+    matrix_precision(builder, matrix, &[*left, *right])?;
+    builder.one(
+        "closed-bulk-f32",
+        PlanOperation::TypedBulkF32Matmul {
+            matrix: matrix.clone(),
+        },
+        &[*left, *right],
+        ValueShape::Vector(matrix.output.backing_words()),
+        None,
+    )
+}
+
+#[cfg(test)]
+mod tests;
diff --git a/crates/zerogravity-core/src/accelerator/program.rs b/crates/zerogravity-core/src/accelerator/program.rs
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -18,6 +18,12 @@
 pub use typed::{
     ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
     ProjectionContract, SourceParameter,
+};
+
+mod bulk_f32;
+pub use bulk_f32::{
+    BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Origin, BulkF32Side,
+    TeacherSpan,
 };
 
 static NEXT_PLAN_BUILDER: AtomicU64 = AtomicU64::new(1);
@@ -294,6 +300,21 @@
             "backend does not implement explicit causal convolution SiLU contract".into(),
         ))
     }
+    fn bulk_f32_input(&mut self, _operand: &BulkF32Operand) -> Result<Self::Value> {
+        Err(Error::Unsupported(
+            "backend has no closed bulk F32 input contract".into(),
+        ))
+    }
+    fn bulk_f32_matmul(
+        &mut self,
+        _matrix: &BulkF32Matrix,
+        _left: &Self::Value,
+        _right: &Self::Value,
+    ) -> Result<Self::Value> {
+        Err(Error::Unsupported(
+            "backend has no closed Haswell bulk F32 matrix contract".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -458,6 +479,12 @@
 /// Lowering may add explicit Pack/Copy instructions for qualified kernel ABIs.
 #[derive(Clone, Debug)]
 pub enum PlanOperation {
+    TypedBulkF32Input {
+        operand: BulkF32Operand,
+    },
+    TypedBulkF32Matmul {
+        matrix: BulkF32Matrix,
+    },
     TypedEmbed {
         tensor: SourceTensor,
         token: ParameterRef,
@@ -774,6 +801,12 @@
         inputs: &[PlanValue],
     ) -> Result<ActivationPrecision> {
         match operation {
+            PlanOperation::TypedBulkF32Input { operand } => {
+                bulk_f32::input_precision(self, operand)
+            }
+            PlanOperation::TypedBulkF32Matmul { matrix } => {
+                bulk_f32::matrix_precision(self, matrix, inputs)
+            }
             PlanOperation::TypedEmbed {
                 tensor,
                 contract: EmbeddingContract::Bf16OutputInput,
@@ -835,7 +868,9 @@
                     .all(|input| input.precision == ActivationPrecision::F32)
                 && !matches!(
                     operation,
-                    PlanOperation::TypedEmbed { .. }
+                    PlanOperation::TypedBulkF32Input { .. }
+                        | PlanOperation::TypedBulkF32Matmul { .. }
+                        | PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
@@ -856,6 +891,22 @@
         shapes: &[ValueShape],
         state: Option<StateEffect>,
     ) -> Result<Vec<PlanValue>> {
+        if !matches!(operation, PlanOperation::TypedBulkF32Matmul { .. })
+            && inputs.iter().any(|input| {
+                self.nodes.iter().any(|node| {
+                    node.outputs.contains(&input.id)
+                        && matches!(
+                            node.operation,
+                            PlanOperation::TypedBulkF32Input { .. }
+                                | PlanOperation::TypedBulkF32Matmul { .. }
+                        )
+                })
+            })
+        {
+            return Err(Error::Unsupported(
+                "bulk typed value cannot enter a legacy operation".into(),
+            ));
+        }
         for input in inputs {
             if input.owner != self.owner
                 || self.values.get(input.id.0 as usize) != Some(&input.shape)
@@ -1033,6 +1084,17 @@
 impl MathematicalBackend for PlanBuilder {
     type Value = PlanValue;
     type Index = ParameterRef;
+    fn bulk_f32_input(&mut self, operand: &BulkF32Operand) -> Result<PlanValue> {
+        bulk_f32::record_input(self, operand)
+    }
+    fn bulk_f32_matmul(
+        &mut self,
+        matrix: &BulkF32Matrix,
+        left: &PlanValue,
+        right: &PlanValue,
+    ) -> Result<PlanValue> {
+        bulk_f32::record_matrix(self, matrix, left, right)
+    }
     fn elements(&self, value: &PlanValue) -> usize {
         value.elements
     }
diff --git a/crates/zerogravity-model-qwen35/src/bulk_f32_slice.rs b/crates/zerogravity-model-qwen35/src/bulk_f32_slice.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/bulk_f32_slice.rs
@@ -0,0 +1,308 @@
+//! Pure explicit default-bulk F32 fragment, not a model producer/forward permit.
+use crate::accelerator::program::{
+    BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Origin, BulkF32Side,
+    PlanBuilder, ProgramNumericalProfile, RecordedOperatorSlice, TeacherSpan,
+};
+use crate::original_binding::{QwenMatrixRole, QwenNumericalProfile};
+use crate::{Error, LoadedQwenModel, Result};
+use std::sync::{
+    Arc,
+    atomic::{AtomicBool, Ordering},
+};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
+    ResourceOwner,
+};
+
+const META: u64 = 16384;
+/// Existing host ledger only; no budget registration or outer RAM claim.
+#[derive(Clone)]
+pub struct BulkF32HostAdmission {
+    pub arbiter: ResourceArbiter,
+    pub resource: PhysicalDeviceKey,
+    pub owner: ResourceOwner,
+}
+/// An explicit actual allocation view. External input is never claimed to be
+/// current model state or a mathematical producer from the present Rust graph.
+pub struct ExternalBulkF32<'a> {
+    pub name: &'a str,
+    pub layout: BulkF32Layout,
+    pub backing: &'a [f32],
+}
+
+pub(crate) struct BulkOwner {
+    pub(crate) model: Arc<LoadedQwenModel>,
+    pub(crate) matrix: BulkF32Matrix,
+    pub(crate) admission: BulkF32HostAdmission,
+    _metadata: GlobalResourceLease,
+}
+pub(crate) enum LeftInput {
+    External(Vec<f32>),
+    Source,
+}
+pub(crate) struct BulkInputs {
+    // Payloads precede staging; source/metadata owner remains last.
+    left: LeftInput,
+    right: Vec<f32>,
+    _staging: GlobalResourceLease,
+    pub(crate) owner: Arc<BulkOwner>,
+}
+impl BulkInputs {
+    pub(crate) fn values(&self, side: BulkF32Side) -> Result<&[f32]> {
+        if side == BulkF32Side::Right {
+            return Ok(&self.right);
+        }
+        match &self.left {
+            LeftInput::External(values) => Ok(values),
+            LeftInput::Source => {
+                let BulkF32Origin::SourceMatrix(tensor) = self.owner.matrix.left().origin() else {
+                    return Err(Error::State("bulk learned backing descriptor".into()));
+                };
+                tensor.validate_source(self.owner.model.as_ref())?;
+                let bytes = self.owner.model.tensor_bytes(&tensor.name)?;
+                // SAFETY: all F32 bit patterns are valid; source raw kind/length
+                // and target little endian are checked. Returned borrow retains
+                // the SAME immutable model allocation; no transposition/cache.
+                let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
+                if !cfg!(target_endian = "little")
+                    || !prefix.is_empty()
+                    || !suffix.is_empty()
+                    || values.len() != self.owner.matrix.left().layout().backing_words()
+                {
+                    return Err(Error::Unsupported(
+                        "bulk original F32 alignment/byte geometry".into(),
+                    ));
+                }
+                Ok(values)
+            }
+        }
+    }
+}
+
+pub struct BulkF32Slice {
+    pub(crate) inputs: Arc<BulkInputs>,
+}
+impl BulkF32Slice {
+    /// Exactly one immutable original GGUF model and actual consecutive IDs.
+    /// These constructors confer pure-operator ownership, never SessionAuthority.
+    #[allow(clippy::too_many_arguments)] // Explicit source, temporal and physical scope.
+    pub fn observed_pair(
+        model: Arc<LoadedQwenModel>,
+        base: usize,
+        capacity: usize,
+        ids: &[u32],
+        left: ExternalBulkF32<'_>,
+        right: ExternalBulkF32<'_>,
+        admission: BulkF32HostAdmission,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<Self> {
+        Self::bind(
+            model,
+            base,
+            capacity,
+            ids,
+            None,
+            Some(left),
+            right,
+            admission,
+            cancelled,
+        )
+    }
+    #[allow(clippy::too_many_arguments)]
+    pub fn source_projection(
+        model: Arc<LoadedQwenModel>,
+        base: usize,
+        capacity: usize,
+        ids: &[u32],
+        role: QwenMatrixRole,
+        right: ExternalBulkF32<'_>,
+        admission: BulkF32HostAdmission,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<Self> {
+        Self::bind(
+            model,
+            base,
+            capacity,
+            ids,
+            Some(role),
+            None,
+            right,
+            admission,
+            cancelled,
+        )
+    }
+    #[allow(clippy::too_many_arguments)]
+    fn bind(
+        model: Arc<LoadedQwenModel>,
+        base: usize,
+        capacity: usize,
+        ids: &[u32],
+        source_role: Option<QwenMatrixRole>,
+        external_left: Option<ExternalBulkF32<'_>>,
+        right: ExternalBulkF32<'_>,
+        admission: BulkF32HostAdmission,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<Self> {
+        not_cancelled(cancelled)?;
+        if model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
+            || admission.resource.backend != "host"
+        {
+            return Err(Error::Unsupported(
+                "bulk pure GGUF F32 existing host scope".into(),
+            ));
+        }
+        // Bounded original metadata is admitted BEFORE span/descriptor allocation.
+        let metadata = admission.arbiter.admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: META,
+            }],
+        )?;
+        let span = TeacherSpan::new(
+            model.source_identity(),
+            base,
+            capacity,
+            model.geometry().vocab_size,
+            ids,
+        )?;
+        let left = match (source_role, &external_left) {
+            (Some(role), None) => {
+                let tensor = model.matrix_descriptor(role)?;
+                BulkF32Operand::source_matrix(span.clone(), model.as_ref(), &tensor.name)?
+            }
+            (None, Some(value)) => {
+                BulkF32Operand::external(span.clone(), BulkF32Side::Left, value.name, value.layout)?
+            }
+            _ => return Err(Error::State("bulk left source transport ambiguity".into())),
+        };
+        let rhs = BulkF32Operand::external(span, BulkF32Side::Right, right.name, right.layout)?;
+        let matrix = BulkF32Matrix::new(left, rhs, BulkF32Contract::HaswellTinyBlasF32V1)?;
+        for (operand, payload) in [
+            (matrix.left(), external_left.as_ref().map(|v| v.backing)),
+            (matrix.right(), Some(right.backing)),
+        ] {
+            if payload.is_some_and(|data| data.len() != operand.layout().backing_words()) {
+                return Err(Error::Malformed(
+                    "bulk exact external allocation backing".into(),
+                ));
+            }
+        }
+        let input_words = right
+            .backing
+            .len()
+            .checked_add(external_left.as_ref().map_or(0, |v| v.backing.len()))
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("bulk aggregate staging bytes".into()))?;
+        // Both private external payloads share ONE exact staging ticket before
+        // either allocation. Learned raw matrix remains borrowed from its owner.
+        let staging = admission.arbiter.admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::TransferStaging,
+                bytes: input_words as u64,
+            }],
+        )?;
+        let owner = Arc::new(BulkOwner {
+            model,
+            matrix,
+            admission,
+            _metadata: metadata,
+        });
+        let copy = |data: &[f32]| -> Result<Vec<f32>> {
+            not_cancelled(cancelled)?;
+            let mut values = Vec::new();
+            values
+                .try_reserve_exact(data.len())
+                .map_err(|_| Error::Limit("bulk external payload allocation".into()))?;
+            if values.capacity() != data.len() {
+                return Err(Error::Limit("bulk external actual capacity".into()));
+            }
+            values.extend_from_slice(data);
+            not_cancelled(cancelled)?;
+            Ok(values)
+        };
+        let left = match external_left {
+            Some(value) => LeftInput::External(copy(value.backing)?),
+            None => LeftInput::Source,
+        };
+        let inputs = Arc::new(BulkInputs {
+            owner,
+            left,
+            right: copy(right.backing)?,
+            _staging: staging,
+        });
+        // Raw source alignment and all logical finite words are subsequently
+        // checked by the unchanged primitive before its own output allocation.
+        inputs.values(BulkF32Side::Left)?;
+        not_cancelled(cancelled)?;
+        Ok(Self { inputs })
+    }
+    pub fn matrix(&self) -> &BulkF32Matrix {
+        &self.inputs.owner.matrix
+    }
+    pub fn record(&self) -> Result<RecordedOperatorSlice> {
+        let mut builder = PlanBuilder::new_operator_slice(
+            self.matrix().source(),
+            ProgramNumericalProfile::GgufSourceV1,
+        )?;
+        let output = crate::forward::bulk_f32_slice(&mut builder, self.matrix())?;
+        builder.finish_operator_slice(output)
+    }
+    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<BulkF32Output> {
+        self.execute_with_provider(None, cancelled)
+    }
+    pub fn execute_with_provider(
+        &self,
+        kernel: Option<&dyn crate::accelerator::MatVecKernel>,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<BulkF32Output> {
+        not_cancelled(cancelled)?;
+        if kernel.is_some() {
+            return Err(Error::Unsupported(
+                "unaware provider cannot execute Haswell matrix contract".into(),
+            ));
+        }
+        let mut math = crate::host_math::HostMath::for_bulk_f32_slice(self, cancelled);
+        let value = crate::forward::bulk_f32_slice(&mut math, self.matrix())?;
+        not_cancelled(cancelled)?;
+        Ok(BulkF32Output {
+            value,
+            shape: self.matrix().output().shape(),
+            source: self.matrix().source(),
+        })
+    }
+}
+#[derive(Clone)]
+pub struct BulkF32Output {
+    value: crate::host_math::HostValue,
+    shape: [usize; 4],
+    source: crate::accelerator::SourceIdentity,
+}
+impl BulkF32Output {
+    pub fn values(&self) -> Result<&[f32]> {
+        self.value.as_slice()
+    }
+    pub fn shape(&self) -> [usize; 4] {
+        self.shape
+    }
+    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
+        self.source
+    }
+    pub fn contract(&self) -> BulkF32Contract {
+        BulkF32Contract::HaswellTinyBlasF32V1
+    }
+    pub fn view(&self, range: std::ops::Range<usize>) -> Result<&[f32]> {
+        self.values()?
+            .get(range)
+            .ok_or_else(|| Error::Malformed("bulk borrowed output range".into()))
+    }
+}
+pub(crate) fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
+    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        return Err(Error::State("closed bulk matrix cancelled".into()));
+    }
+    Ok(())
+}
diff --git a/crates/zerogravity-model-qwen35/src/forward/bulk_f32.rs b/crates/zerogravity-model-qwen35/src/forward/bulk_f32.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/forward/bulk_f32.rs
@@ -0,0 +1,11 @@
+//! One closed matrix fragment via the existing generic mathematical engine.
+use crate::Result;
+use crate::accelerator::program::{BulkF32Matrix, MathematicalBackend};
+pub(crate) fn bulk_f32_slice<M: MathematicalBackend>(
+    math: &mut M,
+    matrix: &BulkF32Matrix,
+) -> Result<M::Value> {
+    let left = math.bulk_f32_input(matrix.left())?;
+    let right = math.bulk_f32_input(matrix.right())?;
+    math.bulk_f32_matmul(matrix, &left, &right)
+}
diff --git a/crates/zerogravity-model-qwen35/src/forward.rs b/crates/zerogravity-model-qwen35/src/forward.rs
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -650,3 +650,9 @@
     }
     Ok(values)
 }
+
+#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
+#[path = "forward/bulk_f32.rs"]
+mod bulk_f32;
+#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
+pub(crate) use bulk_f32::bulk_f32_slice;
diff --git a/crates/zerogravity-model-qwen35/src/host_math/bulk_f32.rs b/crates/zerogravity-model-qwen35/src/host_math/bulk_f32.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/bulk_f32.rs
@@ -0,0 +1,225 @@
+//! Closed input/result backing under the SAME HostMath, no layer/token engine.
+use super::*;
+use crate::accelerator::program::{BulkF32Matrix, BulkF32Operand, BulkF32Side};
+use crate::operator_slice::bulk_f32::{BulkF32Slice, BulkInputs, BulkOwner, not_cancelled};
+use std::sync::atomic::AtomicBool;
+use zerogravity_cpu::math::haswell_bulk_f32::{
+    BulkF32Admission, BulkF32Completed, BulkF32Profile, BulkF32Request, F32View,
+};
+
+pub(super) struct BulkCall<'a> {
+    slice: &'a BulkF32Slice,
+    cancelled: Option<&'a AtomicBool>,
+}
+pub(super) enum BulkBacking {
+    Input {
+        inputs: Arc<BulkInputs>,
+        side: BulkF32Side,
+    },
+    Output {
+        completed: BulkF32Completed,
+        owner: Arc<BulkOwner>,
+    },
+}
+impl BulkBacking {
+    pub(super) fn values(&self) -> Result<&[f32]> {
+        match self {
+            Self::Input { inputs, side } => inputs.values(*side),
+            Self::Output { completed, .. } => Ok(completed.values()),
+        }
+    }
+    fn owner(&self) -> &Arc<BulkOwner> {
+        match self {
+            Self::Input { inputs, .. } => &inputs.owner,
+            Self::Output { owner, .. } => owner,
+        }
+    }
+}
+fn fp_environment() -> Result<()> {
+    let mut controls = 0u32;
+    // SAFETY: read-only control instruction writes one valid stack word.
+    unsafe {
+        std::arch::asm!("stmxcsr [{at}]",at=in(reg)&mut controls,options(nostack,preserves_flags));
+    }
+    if controls & 0xffc0 != 0x1f80 {
+        return Err(Error::Unsupported(
+            "closed Haswell profile requires unchanged RNE/noFTZ/noDAZ controls".into(),
+        ));
+    }
+    Ok(())
+}
+fn request_view<'a>(
+    source: crate::accelerator::SourceIdentity,
+    operand: &'a BulkF32Operand,
+    values: &'a [f32],
+) -> F32View<'a> {
+    let layout = operand.layout();
+    F32View {
+        source,
+        name: operand.name(),
+        values,
+        offset: layout.offset(),
+        shape: layout.shape(),
+        strides: layout.strides(),
+    }
+}
+impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
+    pub(crate) fn for_bulk_f32_slice(
+        slice: &'engine BulkF32Slice,
+        cancelled: Option<&'engine AtomicBool>,
+    ) -> Self {
+        let mut math = Self::for_operator_slice(slice.inputs.owner.model.as_ref(), None);
+        math.bulk_call = Some(BulkCall { slice, cancelled });
+        math
+    }
+    fn bulk_call(&self) -> Result<&BulkCall<'engine>> {
+        if self.engine.is_some()
+            || self.state.is_some()
+            || self.region_kernel.is_some()
+            || self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
+        {
+            return Err(Error::Unsupported(
+                "bulk F32 private same-engine host slice only".into(),
+            ));
+        }
+        self.bulk_call
+            .as_ref()
+            .ok_or_else(|| Error::Unsupported("bulk F32 has no bound temporal input issuer".into()))
+    }
+    pub(super) fn import_bulk_f32(&self, operand: &BulkF32Operand) -> Result<HostValue> {
+        let call = self.bulk_call()?;
+        not_cancelled(call.cancelled)?;
+        fp_environment()?;
+        let owner = &call.slice.inputs.owner;
+        let expected = match operand.side() {
+            BulkF32Side::Left => owner.matrix.left(),
+            BulkF32Side::Right => owner.matrix.right(),
+        };
+        if operand != expected || operand.span().source() != self.source.source_identity() {
+            return Err(Error::State(
+                "bulk input exact bound owner/span/layout".into(),
+            ));
+        }
+        let backing = BulkBacking::Input {
+            inputs: call.slice.inputs.clone(),
+            side: operand.side(),
+        };
+        if backing.values()?.len() != operand.layout().backing_words() {
+            return Err(Error::State("bulk actual input storage extent".into()));
+        }
+        HostValue::from_bulk(
+            backing,
+            operand.layout().backing_words(),
+            self.source.source_identity(),
+        )
+    }
+    pub(super) fn execute_bulk_f32(
+        &self,
+        matrix: &BulkF32Matrix,
+        left: &HostValue,
+        right: &HostValue,
+    ) -> Result<HostValue> {
+        let call = self.bulk_call()?;
+        let owner = &call.slice.inputs.owner;
+        not_cancelled(call.cancelled)?;
+        fp_environment()?;
+        if matrix != &owner.matrix {
+            return Err(Error::State("bulk exact bound matrix/temporal span".into()));
+        }
+        for (value, operand) in [(left, matrix.left()), (right, matrix.right())] {
+            let backing = value.data.bulk.as_ref().ok_or_else(|| {
+                Error::State("bulk input has no admitted original backing".into())
+            })?;
+            if !Arc::ptr_eq(backing.owner(), owner)
+                || value.source != Some(matrix.source())
+                || value.profile != Some(ProgramNumericalProfile::GgufSourceV1)
+                || value.precision != ActivationPrecision::F32
+                || value.elements != operand.layout().backing_words()
+                || value.view != TensorView::vector(0, value.elements)
+                || !matches!(backing,BulkBacking::Input{side,..}if *side==operand.side())
+            {
+                return Err(Error::State(
+                    "bulk foreign/partial/retagged input value".into(),
+                ));
+            }
+        }
+        let a = left.as_slice()?;
+        let b = right.as_slice()?;
+        let extent = |values: &[f32]| -> Result<std::ops::Range<usize>> {
+            let start = values.as_ptr() as usize;
+            let end = values
+                .len()
+                .checked_mul(4)
+                .and_then(|n| start.checked_add(n))
+                .ok_or_else(|| Error::Limit("bulk actual allocation range".into()))?;
+            Ok(start..end)
+        };
+        let ar = extent(a)?;
+        let br = extent(b)?;
+        if ar.start < br.end && br.start < ar.end {
+            return Err(Error::State(
+                "bulk independent input backing aliases".into(),
+            ));
+        }
+        // Unchanged source-derived primitive reserves actual result+scratch
+        // BEFORE its Vec allocation. No old MatVec/provider fallback exists.
+        let completed = zerogravity_cpu::math::haswell_bulk_f32::execute(
+            BulkF32Request {
+                profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
+                source: matrix.source(),
+                left: request_view(matrix.source(), matrix.left(), a),
+                right: request_view(matrix.source(), matrix.right(), b),
+            },
+            BulkF32Admission {
+                arbiter: &owner.admission.arbiter,
+                resource: &owner.admission.resource,
+                owner: &owner.admission.owner,
+            },
+            call.cancelled,
+        )?;
+        fp_environment()?;
+        not_cancelled(call.cancelled)?;
+        if completed.source() != matrix.source()
+            || completed.shape() != matrix.output().shape()
+            || completed.values().len() != matrix.output().backing_words()
+        {
+            return Err(Error::State(
+                "bulk actual primitive result geometry/source".into(),
+            ));
+        }
+        let value = HostValue::from_bulk(
+            BulkBacking::Output {
+                completed,
+                owner: owner.clone(),
+            },
+            matrix.output().backing_words(),
+            matrix.source(),
+        )?;
+        not_cancelled(call.cancelled)?;
+        Ok(value)
+    }
+}
+impl HostValue {
+    fn from_bulk(
+        backing: BulkBacking,
+        elements: usize,
+        source: crate::accelerator::SourceIdentity,
+    ) -> Result<Self> {
+        if !source.is_bound() || backing.values()?.len() != elements {
+            return Err(Error::State("bulk retained backing extent/source".into()));
+        }
+        Ok(Self {
+            data: Arc::new(HostStorage {
+                values: Vec::new(),
+                #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+                workspace: None,
+                bulk: Some(backing),
+            }),
+            view: TensorView::vector(0, elements),
+            elements,
+            precision: ActivationPrecision::F32,
+            source: Some(source),
+            profile: Some(ProgramNumericalProfile::GgufSourceV1),
+        })
+    }
+}
diff --git a/crates/zerogravity-model-qwen35/src/host_math.rs b/crates/zerogravity-model-qwen35/src/host_math.rs
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -22,6 +22,8 @@
 
 struct HostStorage {
     values: Vec<f32>,
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    bulk: Option<bulk_f32::BulkBacking>,
     // Payload drops before its actual same-source host reservation.
     #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
     workspace: Option<Arc<WindowWorkspace>>,
@@ -89,6 +91,8 @@
         Self {
             data: Arc::new(HostStorage {
                 values,
+                #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+                bulk: None,
                 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
                 workspace: None,
             }),
@@ -128,11 +132,31 @@
                 "host operator requires explicit contiguous input".into(),
             ));
         }
-        self.data
-            .get(self.view.offset..self.view.offset + self.elements)
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        let values = match &self.data.bulk {
+            Some(backing) => backing.values()?,
+            None => &self.data.values,
+        };
+        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
+        let values = &self.data.values;
+        values
+            .get(
+                self.view.offset
+                    ..self
+                        .view
+                        .offset
+                        .checked_add(self.elements)
+                        .ok_or_else(|| Error::Limit("host view end overflow".into()))?,
+            )
             .ok_or_else(|| Error::State("invalid private host tensor view".into()))
     }
     pub fn into_vec(self) -> Result<Vec<f32>> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if self.data.bulk.is_some() {
+            return Err(Error::Unsupported(
+                "leased bulk result has no Vec escape".into(),
+            ));
+        }
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         if self.data.workspace.is_some() {
             return Err(Error::Unsupported(
@@ -151,6 +175,8 @@
 
 pub(super) struct HostMath<'engine, 'state, 'trace> {
     source: HostSource<'engine>,
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    bulk_call: Option<bulk_f32::BulkCall<'engine>>,
     engine: Option<&'engine QwenDenseCpu>,
     state: Option<&'state mut QwenState>,
     region_kernel: Option<&'engine dyn crate::accelerator::MatVecKernel>,
@@ -168,6 +194,8 @@
         let position = state.position;
         Self {
             source: HostSource::Loaded(engine.model.as_ref()),
+            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+            bulk_call: None,
             engine: Some(engine),
             state: Some(state),
             region_kernel: engine.kernel.as_deref(),
@@ -183,6 +211,8 @@
     ) -> Self {
         Self {
             source: HostSource::Loaded(source),
+            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+            bulk_call: None,
             engine: None,
             state: None,
             region_kernel: kernel,
@@ -200,6 +230,8 @@
     ) -> Self {
         Self {
             source: HostSource::Windows(source),
+            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+            bulk_call: None,
             engine: None,
             state: None,
             region_kernel: kernel,
@@ -302,6 +334,12 @@
             .ok_or_else(|| Error::State("operator slice has no session state".into()))
     }
     fn legacy_precision(&self, values: &[&HostValue]) -> Result<()> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if values.iter().any(|v| v.data.bulk.is_some()) {
+            return Err(Error::Unsupported(
+                "bulk F32 values require explicit matrix contract".into(),
+            ));
+        }
         if self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
             || values
                 .iter()
@@ -342,6 +380,22 @@
 impl MathematicalBackend for HostMath<'_, '_, '_> {
     type Value = HostValue;
     type Index = usize;
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    fn bulk_f32_input(
+        &mut self,
+        operand: &crate::accelerator::program::BulkF32Operand,
+    ) -> Result<HostValue> {
+        self.import_bulk_f32(operand)
+    }
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    fn bulk_f32_matmul(
+        &mut self,
+        matrix: &crate::accelerator::program::BulkF32Matrix,
+        left: &HostValue,
+        right: &HostValue,
+    ) -> Result<HostValue> {
+        self.execute_bulk_f32(matrix, left, right)
+    }
     fn elements(&self, value: &HostValue) -> usize {
         value.elements
     }
@@ -545,6 +599,12 @@
         ))
     }
     fn view(&mut self, input: &HostValue, view: TensorView) -> Result<HostValue> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if input.data.bulk.is_some() {
+            return Err(Error::Unsupported(
+                "bulk geometry cannot be reinterpreted by legacy view".into(),
+            ));
+        }
         // A view of an already strided view must first use explicit packing.
         if !input.view.is_contiguous() {
             return Err(Error::State(
@@ -568,6 +628,12 @@
         })
     }
     fn contiguous(&mut self, input: &HostValue) -> Result<HostValue> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if input.data.bulk.is_some() && input.view != TensorView::vector(0, input.elements) {
+            return Err(Error::Unsupported(
+                "bulk geometry cannot be implicitly packed".into(),
+            ));
+        }
         if input.view.is_contiguous() {
             return Ok(input.clone());
         }
@@ -1053,3 +1119,6 @@
 
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 mod gdn_conv;
+
+#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+mod bulk_f32;
diff --git a/crates/zerogravity-model-qwen35/src/operator_slice.rs b/crates/zerogravity-model-qwen35/src/operator_slice.rs
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -234,3 +234,8 @@
 
 #[path = "gdn_conv_slice.rs"]
 pub mod gdn_conv;
+
+/// Explicit external/source F32 temporal matrix, no model/state eligibility.
+#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
+#[path = "bulk_f32_slice.rs"]
+pub mod bulk_f32;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/GPU_REFUSAL.patch

Original UTF-8 bytes: 2158; SHA256 `8d24045e9855533e4793721122c74b0e2d380a0bd83bec2265f5a0b000a4ea2f`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-gpu/src/resident_program/batch.rs b/crates/zerogravity-gpu/src/resident_program/batch.rs
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -928,7 +928,9 @@
         let scratch = |i: usize| &self.inner.scratch[i];
         unsafe {
             match &node.operation {
-                PlanOperation::TypedEmbed { .. }
+                PlanOperation::TypedBulkF32Input { .. }
+                | PlanOperation::TypedBulkF32Matmul { .. }
+                | PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
                 | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
diff --git a/crates/zerogravity-gpu/src/resident_program.rs b/crates/zerogravity-gpu/src/resident_program.rs
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -345,7 +345,9 @@
             || plan.nodes().iter().any(|node| {
                 matches!(
                     node.operation,
-                    PlanOperation::TypedEmbed { .. }
+                    PlanOperation::TypedBulkF32Input { .. }
+                        | PlanOperation::TypedBulkF32Matmul { .. }
+                        | PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
@@ -1428,7 +1430,9 @@
         // bound every offset/count. Parent retains the complete original cohort.
         unsafe {
             match &node.operation {
-                PlanOperation::TypedEmbed { .. }
+                PlanOperation::TypedBulkF32Input { .. }
+                | PlanOperation::TypedBulkF32Matmul { .. }
+                | PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
                 | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/NEW_DELTA.patch

Original UTF-8 bytes: 53597; SHA256 `79c065c2669ba31003ab65594af82b8f4dea45d2e652a978db323c0960afab0f`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs b/crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs
@@ -0,0 +1,193 @@
+use super::*;
+fn span() -> TeacherSpan {
+    TeacherSpan::new(
+        SourceIdentity::for_new_allocation().unwrap(),
+        0,
+        8448,
+        248320,
+        &(0..64).collect::<Vec<_>>(),
+    )
+    .unwrap()
+}
+fn matrix(span: &TeacherSpan) -> BulkF32Matrix {
+    let left = BulkF32Operand::external(
+        span.clone(),
+        BulkF32Side::Left,
+        "actual external V",
+        BulkF32Layout::new([256, 256, 2, 1], [1, 8448, 2162688, 4325376], 0, 4317184).unwrap(),
+    )
+    .unwrap();
+    let right = BulkF32Operand::external(
+        span.clone(),
+        BulkF32Side::Right,
+        "actual external P",
+        BulkF32Layout::dense([256, 64, 8, 1]).unwrap(),
+    )
+    .unwrap();
+    BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1).unwrap()
+}
+#[test]
+fn actual_temporal_n_is_not_cohort_width_and_full_k_is_preserved() {
+    let span = span();
+    let operation = matrix(&span);
+    assert_eq!(operation.left().layout().shape(), [256, 256, 2, 1]);
+    assert_eq!(operation.right().layout().shape(), [256, 64, 8, 1]);
+    assert_eq!(operation.output().shape(), [256, 64, 8, 1]);
+    assert_eq!(operation.output().backing_words(), 131072);
+    // Real N2 is a different declared teacher span, never a B2 request label.
+    let wrong = BulkF32Operand::external(
+        span.clone(),
+        BulkF32Side::Right,
+        "P",
+        BulkF32Layout::dense([256, 2, 8, 1]).unwrap(),
+    );
+    assert!(wrong.is_err());
+    assert!(TeacherSpan::new(span.source(), 0, 8448, 248320, &[11]).is_err());
+    assert!(TeacherSpan::new(span.source(), 8447, 8448, 248320, &[11, 353]).is_err());
+}
+#[test]
+fn same_ids_source_names_do_not_merge_issuers_and_wrong_full_k_refuses() {
+    let span = span();
+    let operation = matrix(&span);
+    let foreign = TeacherSpan::new(
+        span.source(),
+        span.base(),
+        span.capacity(),
+        248320,
+        span.ids(),
+    )
+    .unwrap();
+    let right = BulkF32Operand::external(
+        foreign,
+        BulkF32Side::Right,
+        "actual external P",
+        operation.right().layout(),
+    )
+    .unwrap();
+    assert!(
+        BulkF32Matrix::new(
+            operation.left().clone(),
+            right,
+            BulkF32Contract::HaswellTinyBlasF32V1
+        )
+        .is_err()
+    );
+    let right = BulkF32Operand::external(
+        span.clone(),
+        BulkF32Side::Right,
+        "P",
+        BulkF32Layout::dense([128, 64, 8, 1]).unwrap(),
+    )
+    .unwrap();
+    assert!(
+        BulkF32Matrix::new(
+            operation.left().clone(),
+            right,
+            BulkF32Contract::HaswellTinyBlasF32V1
+        )
+        .is_err()
+    );
+}
+#[test]
+fn real_nonzero_offsets_and_gaps_need_full_checked_storage() {
+    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 47).is_ok());
+    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 46).is_err());
+    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 7, 28, 28], 0, 32).is_err());
+    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], usize::MAX, 64).is_err());
+    assert!(BulkF32Layout::dense([8192, 4096, 8, 1]).is_err());
+    let span = span();
+    assert!(
+        BulkF32Operand::external(
+            span,
+            BulkF32Side::Right,
+            "strided P",
+            BulkF32Layout::new([8, 64, 1, 1], [1, 12, 768, 768], 0, 764).unwrap()
+        )
+        .is_err()
+    );
+}
+#[test]
+fn same_builder_three_nodes_stay_pure_and_legacy_use_or_foreign_values_refuse() {
+    let span = span();
+    let operation = matrix(&span);
+    let mut b =
+        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
+            .unwrap();
+    let left = b.bulk_f32_input(operation.left()).unwrap();
+    let right = b.bulk_f32_input(operation.right()).unwrap();
+    let mut other =
+        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
+            .unwrap();
+    let foreign = other.bulk_f32_input(operation.right()).unwrap();
+    assert!(b.bulk_f32_matmul(&operation, &left, &foreign).is_err());
+    assert!(b.silu("legacy", &right, None).is_err());
+    let output = b.bulk_f32_matmul(&operation, &left, &right).unwrap();
+    let slice = b.finish_operator_slice(output).unwrap();
+    assert_eq!(slice.plan().nodes().len(), 3);
+    assert!(slice.plan().required_domains().is_empty());
+    assert!(slice.plan().proposed_domains().is_empty());
+    assert_eq!(slice.plan().frontier_delta(), 0);
+    assert!(slice.plan().logits().is_none());
+    assert!(slice.plan().is_operator_slice());
+    let slot = StateSlot::Conv(ConvDomain(0));
+    let mut full = PlanBuilder::new(span.source(), 8448, vec![slot], false).unwrap();
+    assert!(full.bulk_f32_input(operation.left()).is_err());
+}
+
+struct Original {
+    identity: SourceIdentity,
+    kind: QuantKind,
+    shape: [u64; 2],
+    bytes: Vec<u8>,
+}
+impl TensorSource for Original {
+    fn source_identity(&self) -> SourceIdentity {
+        self.identity
+    }
+    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
+        if name != "original.weight" {
+            return Err(Error::Malformed("unknown original name".into()));
+        }
+        Ok(SourceTensorRef {
+            kind: self.kind,
+            shape: &self.shape,
+            packed_bytes: &self.bytes,
+        })
+    }
+}
+#[test]
+fn whole_raw_source_matrix_keeps_axes_dtype_extent_and_exact_issuer() {
+    let original = Original {
+        identity: SourceIdentity::for_new_allocation().unwrap(),
+        kind: QuantKind::F32,
+        shape: [8, 4],
+        bytes: vec![0; 128],
+    };
+    let span = TeacherSpan::new(original.identity, 0, 8, 248320, &[11, 353]).unwrap();
+    let left = BulkF32Operand::source_matrix(span.clone(), &original, "original.weight").unwrap();
+    assert_eq!(left.layout().shape(), [8, 4, 1, 1]);
+    let BulkF32Origin::SourceMatrix(raw) = left.origin() else {
+        panic!("original raw source expected")
+    };
+    assert_eq!(raw.axes, MatrixAxes::InputOutput);
+    raw.validate_source(&original).unwrap();
+    let foreign = Original {
+        identity: SourceIdentity::for_new_allocation().unwrap(),
+        ..original
+    };
+    assert!(BulkF32Operand::source_matrix(span.clone(), &foreign, "original.weight").is_err());
+    let bf16 = Original {
+        identity: span.source(),
+        kind: QuantKind::Bf16,
+        shape: [8, 4],
+        bytes: vec![0; 64],
+    };
+    assert!(BulkF32Operand::source_matrix(span.clone(), &bf16, "original.weight").is_err());
+    let short = Original {
+        identity: span.source(),
+        kind: QuantKind::F32,
+        shape: [8, 4],
+        bytes: vec![0; 124],
+    };
+    assert!(BulkF32Operand::source_matrix(span, &short, "original.weight").is_err());
+}
diff --git a/crates/zerogravity-core/src/accelerator/program/bulk_f32.rs b/crates/zerogravity-core/src/accelerator/program/bulk_f32.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/bulk_f32.rs
@@ -0,0 +1,410 @@
+//! Closed temporal matrix metadata; no session, model route or producer permit.
+use super::*;
+
+/// ONE consecutive teacher sequence. Clones share one immutable issuer; equal
+/// text/IDs from another constructor do not silently become the same scope.
+#[derive(Clone, Debug)]
+pub struct TeacherSpan(Arc<TeacherSpanData>);
+#[derive(Debug)]
+struct TeacherSpanData {
+    source: SourceIdentity,
+    base: usize,
+    capacity: usize,
+    ids: Arc<[u32]>,
+}
+impl PartialEq for TeacherSpan {
+    fn eq(&self, other: &Self) -> bool {
+        Arc::ptr_eq(&self.0, &other.0)
+    }
+}
+impl Eq for TeacherSpan {}
+impl TeacherSpan {
+    /// Metadata only; these IDs are not a current SessionAuthority/StateRoot.
+    pub fn new(
+        source: SourceIdentity,
+        base: usize,
+        capacity: usize,
+        vocabulary: u64,
+        ids: &[u32],
+    ) -> Result<Self> {
+        if !source.is_bound()
+            || !(2..=64).contains(&ids.len())
+            || capacity == 0
+            || capacity > 8448
+            || base.checked_add(ids.len()).is_none_or(|end| end > capacity)
+            || vocabulary == 0
+            || vocabulary > u64::from(u32::MAX) + 1
+            || ids.iter().any(|&id| u64::from(id) >= vocabulary)
+        {
+            return Err(Error::Malformed(
+                "closed teacher span source/geometry/IDs".into(),
+            ));
+        }
+        Ok(Self(Arc::new(TeacherSpanData {
+            source,
+            base,
+            capacity,
+            ids: Arc::from(ids),
+        })))
+    }
+    pub fn source(&self) -> SourceIdentity {
+        self.0.source
+    }
+    pub fn base(&self) -> usize {
+        self.0.base
+    }
+    pub fn capacity(&self) -> usize {
+        self.0.capacity
+    }
+    pub fn ids(&self) -> &[u32] {
+        &self.0.ids
+    }
+    pub fn len(&self) -> usize {
+        self.0.ids.len()
+    }
+    pub fn is_empty(&self) -> bool {
+        false
+    }
+}
+
+/// Actual F32 storage axes `[full_K,M_or_N,head,sequence]`, word strides.
+/// Offset/backing describe THIS allocation; original capture provenance stays
+/// separate if transport reconstructed gaps. Full K is never split/reduced.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub struct BulkF32Layout {
+    shape: [usize; 4],
+    strides: [usize; 4],
+    offset: usize,
+    backing: usize,
+}
+impl BulkF32Layout {
+    pub fn new(
+        shape: [usize; 4],
+        strides: [usize; 4],
+        offset: usize,
+        backing: usize,
+    ) -> Result<Self> {
+        let value = Self {
+            shape,
+            strides,
+            offset,
+            backing,
+        };
+        value.validate()?;
+        Ok(value)
+    }
+    pub fn dense(shape: [usize; 4]) -> Result<Self> {
+        let mut strides = [1usize; 4];
+        for i in 1..4 {
+            strides[i] = strides[i - 1]
+                .checked_mul(shape[i - 1])
+                .ok_or_else(|| Error::Limit("bulk dense stride".into()))?;
+        }
+        let backing = strides[3]
+            .checked_mul(shape[3])
+            .ok_or_else(|| Error::Limit("bulk dense words".into()))?;
+        Self::new(shape, strides, 0, backing)
+    }
+    pub fn shape(self) -> [usize; 4] {
+        self.shape
+    }
+    pub fn strides(self) -> [usize; 4] {
+        self.strides
+    }
+    pub fn offset(self) -> usize {
+        self.offset
+    }
+    pub fn backing_words(self) -> usize {
+        self.backing
+    }
+    pub fn logical_words(self) -> Result<usize> {
+        self.shape.iter().try_fold(1usize, |n, &x| {
+            n.checked_mul(x)
+                .ok_or_else(|| Error::Limit("bulk logical words".into()))
+        })
+    }
+    pub fn validate(self) -> Result<()> {
+        if self.shape.contains(&0)
+            || self.shape[0] > 8192
+            || self.shape[1] > 4096
+            || self.shape[2] > 64
+            || self.shape[3] != 1
+            || self.strides[0] != 1
+            || self.backing == 0
+            || self.backing > 8 * 1024 * 1024
+        {
+            return Err(Error::Unsupported("closed bulk F32 axes/backing".into()));
+        }
+        let mut previous = self.shape[0];
+        for (&stride, &size) in self.strides.iter().zip(&self.shape).skip(1) {
+            if stride < previous {
+                return Err(Error::Malformed("bulk overlapping rows/axes".into()));
+            }
+            previous = stride
+                .checked_mul(size)
+                .ok_or_else(|| Error::Limit("bulk stride product".into()))?;
+        }
+        let end = self
+            .shape
+            .iter()
+            .zip(self.strides)
+            .try_fold(self.offset, |n, (&size, stride)| {
+                (size - 1)
+                    .checked_mul(stride)
+                    .and_then(|x| n.checked_add(x))
+                    .ok_or_else(|| Error::Limit("bulk full span".into()))
+            })?
+            .checked_add(1)
+            .ok_or_else(|| Error::Limit("bulk span end".into()))?;
+        if end > self.backing || self.logical_words()? > 1024 * 1024 {
+            return Err(Error::Limit("bulk operand span/logical bound".into()));
+        }
+        Ok(())
+    }
+    pub fn is_dense(self) -> bool {
+        Self::dense(self.shape).is_ok_and(|dense| dense == self)
+    }
+}
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum BulkF32Side {
+    Left,
+    Right,
+}
+#[derive(Clone, Debug, PartialEq, Eq)]
+pub enum BulkF32Origin {
+    /// Caller-supplied pure operator input, explicitly NOT a model/state producer.
+    ExternalObserved { name: String },
+    /// Exact whole original raw learned matrix, no dequantization/retag/cache.
+    SourceMatrix(SourceTensor),
+}
+#[derive(Clone, Debug, PartialEq, Eq)]
+pub struct BulkF32Operand {
+    span: TeacherSpan,
+    side: BulkF32Side,
+    layout: BulkF32Layout,
+    origin: BulkF32Origin,
+}
+impl BulkF32Operand {
+    pub fn external(
+        span: TeacherSpan,
+        side: BulkF32Side,
+        name: &str,
+        layout: BulkF32Layout,
+    ) -> Result<Self> {
+        if name.is_empty() || name.len() > 256 || name.contains('\0') {
+            return Err(Error::Malformed("bulk external input label bound".into()));
+        }
+        let value = Self {
+            span,
+            side,
+            layout,
+            origin: BulkF32Origin::ExternalObserved { name: name.into() },
+        };
+        value.validate()?;
+        Ok(value)
+    }
+    pub fn source_matrix(span: TeacherSpan, source: &dyn TensorSource, name: &str) -> Result<Self> {
+        if name.is_empty() || name.len() > 256 || name.contains('\0') {
+            return Err(Error::Malformed("bulk learned input label bound".into()));
+        }
+        let tensor = SourceTensor::from_source(source, name, MatrixAxes::InputOutput)?;
+        if tensor.kind != QuantKind::F32 || tensor.source != span.source() {
+            return Err(Error::Unsupported(
+                "bulk original F32 source matrix only".into(),
+            ));
+        }
+        let layout = BulkF32Layout::dense([tensor.columns, tensor.rows, 1, 1])?;
+        let value = Self {
+            span,
+            side: BulkF32Side::Left,
+            layout,
+            origin: BulkF32Origin::SourceMatrix(tensor),
+        };
+        value.validate()?;
+        Ok(value)
+    }
+    pub fn span(&self) -> &TeacherSpan {
+        &self.span
+    }
+    pub fn side(&self) -> BulkF32Side {
+        self.side
+    }
+    pub fn layout(&self) -> BulkF32Layout {
+        self.layout
+    }
+    pub fn origin(&self) -> &BulkF32Origin {
+        &self.origin
+    }
+    pub fn name(&self) -> &str {
+        match &self.origin {
+            BulkF32Origin::ExternalObserved { name } => name,
+            BulkF32Origin::SourceMatrix(tensor) => &tensor.name,
+        }
+    }
+    pub fn validate(&self) -> Result<()> {
+        self.layout.validate()?;
+        if self.side == BulkF32Side::Right
+            && (self.layout.shape[1] != self.span.len() || !self.layout.is_dense())
+        {
+            return Err(Error::Unsupported(
+                "actual teacher N and original contiguous RHS required".into(),
+            ));
+        }
+        if let BulkF32Origin::SourceMatrix(tensor) = &self.origin {
+            if self.side != BulkF32Side::Left
+                || tensor.source != self.span.source()
+                || tensor.kind != QuantKind::F32
+                || tensor.axes != MatrixAxes::InputOutput
+                || self.layout != BulkF32Layout::dense([tensor.columns, tensor.rows, 1, 1])?
+            {
+                return Err(Error::State("bulk whole learned descriptor/axes".into()));
+            }
+        }
+        Ok(())
+    }
+}
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum BulkF32Contract {
+    HaswellTinyBlasF32V1,
+}
+#[derive(Clone, Debug, PartialEq, Eq)]
+pub struct BulkF32Matrix {
+    left: BulkF32Operand,
+    right: BulkF32Operand,
+    contract: BulkF32Contract,
+    output: BulkF32Layout,
+}
+impl BulkF32Matrix {
+    pub fn new(
+        left: BulkF32Operand,
+        right: BulkF32Operand,
+        contract: BulkF32Contract,
+    ) -> Result<Self> {
+        left.validate()?;
+        right.validate()?;
+        let a = left.layout.shape;
+        let b = right.layout.shape;
+        if left.side != BulkF32Side::Left
+            || right.side != BulkF32Side::Right
+            || left.span != right.span
+            || a[0] != b[0]
+            || !a[1].is_multiple_of(4)
+            || !b[2].is_multiple_of(a[2])
+        {
+            return Err(Error::Unsupported(
+                "closed temporal matrix issuer/fullK/head broadcast".into(),
+            ));
+        }
+        let output = BulkF32Layout::dense([a[1], b[1], b[2], 1])?;
+        if output
+            .logical_words()?
+            .checked_mul(a[0])
+            .is_none_or(|n| n > 128 * 1024 * 1024)
+        {
+            return Err(Error::Limit("bulk complete K work bound".into()));
+        }
+        Ok(Self {
+            left,
+            right,
+            contract,
+            output,
+        })
+    }
+    pub fn left(&self) -> &BulkF32Operand {
+        &self.left
+    }
+    pub fn right(&self) -> &BulkF32Operand {
+        &self.right
+    }
+    pub fn contract(&self) -> BulkF32Contract {
+        self.contract
+    }
+    pub fn output(&self) -> BulkF32Layout {
+        self.output
+    }
+    pub fn source(&self) -> SourceIdentity {
+        self.left.span.source()
+    }
+}
+
+fn check_builder(builder: &PlanBuilder, source: SourceIdentity) -> Result<()> {
+    if !builder.operator_slice
+        || builder.profile != ProgramNumericalProfile::GgufSourceV1
+        || builder.source != source
+    {
+        return Err(Error::Unsupported(
+            "bulk F32 requires explicit state-free GGUF operator slice".into(),
+        ));
+    }
+    Ok(())
+}
+pub(super) fn input_precision(
+    builder: &PlanBuilder,
+    operand: &BulkF32Operand,
+) -> Result<ActivationPrecision> {
+    check_builder(builder, operand.span.source())?;
+    operand.validate()?;
+    Ok(ActivationPrecision::F32)
+}
+pub(super) fn matrix_precision(
+    builder: &PlanBuilder,
+    matrix: &BulkF32Matrix,
+    inputs: &[PlanValue],
+) -> Result<ActivationPrecision> {
+    check_builder(builder, matrix.source())?;
+    if inputs.len() != 2 {
+        return Err(Error::Malformed("bulk input arity".into()));
+    }
+    for (input, expected) in inputs.iter().zip([matrix.left(), matrix.right()]) {
+        let producer = builder
+            .nodes
+            .iter()
+            .find(|node| node.outputs.contains(&input.id));
+        if input.precision != ActivationPrecision::F32
+            || input.elements != expected.layout.backing_words()
+            || !matches!(producer.map(|node|&node.operation),Some(PlanOperation::TypedBulkF32Input{operand})if operand==expected)
+        {
+            return Err(Error::State(
+                "bulk exact input producer/layout/span membership".into(),
+            ));
+        }
+    }
+    Ok(ActivationPrecision::F32)
+}
+pub(super) fn record_input(
+    builder: &mut PlanBuilder,
+    operand: &BulkF32Operand,
+) -> Result<PlanValue> {
+    input_precision(builder, operand)?;
+    builder.one(
+        operand.name(),
+        PlanOperation::TypedBulkF32Input {
+            operand: operand.clone(),
+        },
+        &[],
+        ValueShape::Vector(operand.layout.backing_words()),
+        None,
+    )
+}
+pub(super) fn record_matrix(
+    builder: &mut PlanBuilder,
+    matrix: &BulkF32Matrix,
+    left: &PlanValue,
+    right: &PlanValue,
+) -> Result<PlanValue> {
+    matrix_precision(builder, matrix, &[*left, *right])?;
+    builder.one(
+        "closed-bulk-f32",
+        PlanOperation::TypedBulkF32Matmul {
+            matrix: matrix.clone(),
+        },
+        &[*left, *right],
+        ValueShape::Vector(matrix.output.backing_words()),
+        None,
+    )
+}
+
+#[cfg(test)]
+mod tests;
diff --git a/crates/zerogravity-core/src/accelerator/program.rs b/crates/zerogravity-core/src/accelerator/program.rs
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -18,6 +18,12 @@
 pub use typed::{
     ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
     ProjectionContract, SourceParameter,
+};
+
+mod bulk_f32;
+pub use bulk_f32::{
+    BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Origin, BulkF32Side,
+    TeacherSpan,
 };
 
 static NEXT_PLAN_BUILDER: AtomicU64 = AtomicU64::new(1);
@@ -294,6 +300,21 @@
             "backend does not implement explicit causal convolution SiLU contract".into(),
         ))
     }
+    fn bulk_f32_input(&mut self, _operand: &BulkF32Operand) -> Result<Self::Value> {
+        Err(Error::Unsupported(
+            "backend has no closed bulk F32 input contract".into(),
+        ))
+    }
+    fn bulk_f32_matmul(
+        &mut self,
+        _matrix: &BulkF32Matrix,
+        _left: &Self::Value,
+        _right: &Self::Value,
+    ) -> Result<Self::Value> {
+        Err(Error::Unsupported(
+            "backend has no closed Haswell bulk F32 matrix contract".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -458,6 +479,12 @@
 /// Lowering may add explicit Pack/Copy instructions for qualified kernel ABIs.
 #[derive(Clone, Debug)]
 pub enum PlanOperation {
+    TypedBulkF32Input {
+        operand: BulkF32Operand,
+    },
+    TypedBulkF32Matmul {
+        matrix: BulkF32Matrix,
+    },
     TypedEmbed {
         tensor: SourceTensor,
         token: ParameterRef,
@@ -774,6 +801,12 @@
         inputs: &[PlanValue],
     ) -> Result<ActivationPrecision> {
         match operation {
+            PlanOperation::TypedBulkF32Input { operand } => {
+                bulk_f32::input_precision(self, operand)
+            }
+            PlanOperation::TypedBulkF32Matmul { matrix } => {
+                bulk_f32::matrix_precision(self, matrix, inputs)
+            }
             PlanOperation::TypedEmbed {
                 tensor,
                 contract: EmbeddingContract::Bf16OutputInput,
@@ -835,7 +868,9 @@
                     .all(|input| input.precision == ActivationPrecision::F32)
                 && !matches!(
                     operation,
-                    PlanOperation::TypedEmbed { .. }
+                    PlanOperation::TypedBulkF32Input { .. }
+                        | PlanOperation::TypedBulkF32Matmul { .. }
+                        | PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
@@ -856,6 +891,22 @@
         shapes: &[ValueShape],
         state: Option<StateEffect>,
     ) -> Result<Vec<PlanValue>> {
+        if !matches!(operation, PlanOperation::TypedBulkF32Matmul { .. })
+            && inputs.iter().any(|input| {
+                self.nodes.iter().any(|node| {
+                    node.outputs.contains(&input.id)
+                        && matches!(
+                            node.operation,
+                            PlanOperation::TypedBulkF32Input { .. }
+                                | PlanOperation::TypedBulkF32Matmul { .. }
+                        )
+                })
+            })
+        {
+            return Err(Error::Unsupported(
+                "bulk typed value cannot enter a legacy operation".into(),
+            ));
+        }
         for input in inputs {
             if input.owner != self.owner
                 || self.values.get(input.id.0 as usize) != Some(&input.shape)
@@ -1033,6 +1084,17 @@
 impl MathematicalBackend for PlanBuilder {
     type Value = PlanValue;
     type Index = ParameterRef;
+    fn bulk_f32_input(&mut self, operand: &BulkF32Operand) -> Result<PlanValue> {
+        bulk_f32::record_input(self, operand)
+    }
+    fn bulk_f32_matmul(
+        &mut self,
+        matrix: &BulkF32Matrix,
+        left: &PlanValue,
+        right: &PlanValue,
+    ) -> Result<PlanValue> {
+        bulk_f32::record_matrix(self, matrix, left, right)
+    }
     fn elements(&self, value: &PlanValue) -> usize {
         value.elements
     }
diff --git a/crates/zerogravity-model-qwen35/src/bulk_f32_slice.rs b/crates/zerogravity-model-qwen35/src/bulk_f32_slice.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/bulk_f32_slice.rs
@@ -0,0 +1,308 @@
+//! Pure explicit default-bulk F32 fragment, not a model producer/forward permit.
+use crate::accelerator::program::{
+    BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Origin, BulkF32Side,
+    PlanBuilder, ProgramNumericalProfile, RecordedOperatorSlice, TeacherSpan,
+};
+use crate::original_binding::{QwenMatrixRole, QwenNumericalProfile};
+use crate::{Error, LoadedQwenModel, Result};
+use std::sync::{
+    Arc,
+    atomic::{AtomicBool, Ordering},
+};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
+    ResourceOwner,
+};
+
+const META: u64 = 16384;
+/// Existing host ledger only; no budget registration or outer RAM claim.
+#[derive(Clone)]
+pub struct BulkF32HostAdmission {
+    pub arbiter: ResourceArbiter,
+    pub resource: PhysicalDeviceKey,
+    pub owner: ResourceOwner,
+}
+/// An explicit actual allocation view. External input is never claimed to be
+/// current model state or a mathematical producer from the present Rust graph.
+pub struct ExternalBulkF32<'a> {
+    pub name: &'a str,
+    pub layout: BulkF32Layout,
+    pub backing: &'a [f32],
+}
+
+pub(crate) struct BulkOwner {
+    pub(crate) model: Arc<LoadedQwenModel>,
+    pub(crate) matrix: BulkF32Matrix,
+    pub(crate) admission: BulkF32HostAdmission,
+    _metadata: GlobalResourceLease,
+}
+pub(crate) enum LeftInput {
+    External(Vec<f32>),
+    Source,
+}
+pub(crate) struct BulkInputs {
+    // Payloads precede staging; source/metadata owner remains last.
+    left: LeftInput,
+    right: Vec<f32>,
+    _staging: GlobalResourceLease,
+    pub(crate) owner: Arc<BulkOwner>,
+}
+impl BulkInputs {
+    pub(crate) fn values(&self, side: BulkF32Side) -> Result<&[f32]> {
+        if side == BulkF32Side::Right {
+            return Ok(&self.right);
+        }
+        match &self.left {
+            LeftInput::External(values) => Ok(values),
+            LeftInput::Source => {
+                let BulkF32Origin::SourceMatrix(tensor) = self.owner.matrix.left().origin() else {
+                    return Err(Error::State("bulk learned backing descriptor".into()));
+                };
+                tensor.validate_source(self.owner.model.as_ref())?;
+                let bytes = self.owner.model.tensor_bytes(&tensor.name)?;
+                // SAFETY: all F32 bit patterns are valid; source raw kind/length
+                // and target little endian are checked. Returned borrow retains
+                // the SAME immutable model allocation; no transposition/cache.
+                let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
+                if !cfg!(target_endian = "little")
+                    || !prefix.is_empty()
+                    || !suffix.is_empty()
+                    || values.len() != self.owner.matrix.left().layout().backing_words()
+                {
+                    return Err(Error::Unsupported(
+                        "bulk original F32 alignment/byte geometry".into(),
+                    ));
+                }
+                Ok(values)
+            }
+        }
+    }
+}
+
+pub struct BulkF32Slice {
+    pub(crate) inputs: Arc<BulkInputs>,
+}
+impl BulkF32Slice {
+    /// Exactly one immutable original GGUF model and actual consecutive IDs.
+    /// These constructors confer pure-operator ownership, never SessionAuthority.
+    #[allow(clippy::too_many_arguments)] // Explicit source, temporal and physical scope.
+    pub fn observed_pair(
+        model: Arc<LoadedQwenModel>,
+        base: usize,
+        capacity: usize,
+        ids: &[u32],
+        left: ExternalBulkF32<'_>,
+        right: ExternalBulkF32<'_>,
+        admission: BulkF32HostAdmission,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<Self> {
+        Self::bind(
+            model,
+            base,
+            capacity,
+            ids,
+            None,
+            Some(left),
+            right,
+            admission,
+            cancelled,
+        )
+    }
+    #[allow(clippy::too_many_arguments)]
+    pub fn source_projection(
+        model: Arc<LoadedQwenModel>,
+        base: usize,
+        capacity: usize,
+        ids: &[u32],
+        role: QwenMatrixRole,
+        right: ExternalBulkF32<'_>,
+        admission: BulkF32HostAdmission,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<Self> {
+        Self::bind(
+            model,
+            base,
+            capacity,
+            ids,
+            Some(role),
+            None,
+            right,
+            admission,
+            cancelled,
+        )
+    }
+    #[allow(clippy::too_many_arguments)]
+    fn bind(
+        model: Arc<LoadedQwenModel>,
+        base: usize,
+        capacity: usize,
+        ids: &[u32],
+        source_role: Option<QwenMatrixRole>,
+        external_left: Option<ExternalBulkF32<'_>>,
+        right: ExternalBulkF32<'_>,
+        admission: BulkF32HostAdmission,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<Self> {
+        not_cancelled(cancelled)?;
+        if model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
+            || admission.resource.backend != "host"
+        {
+            return Err(Error::Unsupported(
+                "bulk pure GGUF F32 existing host scope".into(),
+            ));
+        }
+        // Bounded original metadata is admitted BEFORE span/descriptor allocation.
+        let metadata = admission.arbiter.admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Other,
+                bytes: META,
+            }],
+        )?;
+        let span = TeacherSpan::new(
+            model.source_identity(),
+            base,
+            capacity,
+            model.geometry().vocab_size,
+            ids,
+        )?;
+        let left = match (source_role, &external_left) {
+            (Some(role), None) => {
+                let tensor = model.matrix_descriptor(role)?;
+                BulkF32Operand::source_matrix(span.clone(), model.as_ref(), &tensor.name)?
+            }
+            (None, Some(value)) => {
+                BulkF32Operand::external(span.clone(), BulkF32Side::Left, value.name, value.layout)?
+            }
+            _ => return Err(Error::State("bulk left source transport ambiguity".into())),
+        };
+        let rhs = BulkF32Operand::external(span, BulkF32Side::Right, right.name, right.layout)?;
+        let matrix = BulkF32Matrix::new(left, rhs, BulkF32Contract::HaswellTinyBlasF32V1)?;
+        for (operand, payload) in [
+            (matrix.left(), external_left.as_ref().map(|v| v.backing)),
+            (matrix.right(), Some(right.backing)),
+        ] {
+            if payload.is_some_and(|data| data.len() != operand.layout().backing_words()) {
+                return Err(Error::Malformed(
+                    "bulk exact external allocation backing".into(),
+                ));
+            }
+        }
+        let input_words = right
+            .backing
+            .len()
+            .checked_add(external_left.as_ref().map_or(0, |v| v.backing.len()))
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("bulk aggregate staging bytes".into()))?;
+        // Both private external payloads share ONE exact staging ticket before
+        // either allocation. Learned raw matrix remains borrowed from its owner.
+        let staging = admission.arbiter.admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::TransferStaging,
+                bytes: input_words as u64,
+            }],
+        )?;
+        let owner = Arc::new(BulkOwner {
+            model,
+            matrix,
+            admission,
+            _metadata: metadata,
+        });
+        let copy = |data: &[f32]| -> Result<Vec<f32>> {
+            not_cancelled(cancelled)?;
+            let mut values = Vec::new();
+            values
+                .try_reserve_exact(data.len())
+                .map_err(|_| Error::Limit("bulk external payload allocation".into()))?;
+            if values.capacity() != data.len() {
+                return Err(Error::Limit("bulk external actual capacity".into()));
+            }
+            values.extend_from_slice(data);
+            not_cancelled(cancelled)?;
+            Ok(values)
+        };
+        let left = match external_left {
+            Some(value) => LeftInput::External(copy(value.backing)?),
+            None => LeftInput::Source,
+        };
+        let inputs = Arc::new(BulkInputs {
+            owner,
+            left,
+            right: copy(right.backing)?,
+            _staging: staging,
+        });
+        // Raw source alignment and all logical finite words are subsequently
+        // checked by the unchanged primitive before its own output allocation.
+        inputs.values(BulkF32Side::Left)?;
+        not_cancelled(cancelled)?;
+        Ok(Self { inputs })
+    }
+    pub fn matrix(&self) -> &BulkF32Matrix {
+        &self.inputs.owner.matrix
+    }
+    pub fn record(&self) -> Result<RecordedOperatorSlice> {
+        let mut builder = PlanBuilder::new_operator_slice(
+            self.matrix().source(),
+            ProgramNumericalProfile::GgufSourceV1,
+        )?;
+        let output = crate::forward::bulk_f32_slice(&mut builder, self.matrix())?;
+        builder.finish_operator_slice(output)
+    }
+    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<BulkF32Output> {
+        self.execute_with_provider(None, cancelled)
+    }
+    pub fn execute_with_provider(
+        &self,
+        kernel: Option<&dyn crate::accelerator::MatVecKernel>,
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<BulkF32Output> {
+        not_cancelled(cancelled)?;
+        if kernel.is_some() {
+            return Err(Error::Unsupported(
+                "unaware provider cannot execute Haswell matrix contract".into(),
+            ));
+        }
+        let mut math = crate::host_math::HostMath::for_bulk_f32_slice(self, cancelled);
+        let value = crate::forward::bulk_f32_slice(&mut math, self.matrix())?;
+        not_cancelled(cancelled)?;
+        Ok(BulkF32Output {
+            value,
+            shape: self.matrix().output().shape(),
+            source: self.matrix().source(),
+        })
+    }
+}
+#[derive(Clone)]
+pub struct BulkF32Output {
+    value: crate::host_math::HostValue,
+    shape: [usize; 4],
+    source: crate::accelerator::SourceIdentity,
+}
+impl BulkF32Output {
+    pub fn values(&self) -> Result<&[f32]> {
+        self.value.as_slice()
+    }
+    pub fn shape(&self) -> [usize; 4] {
+        self.shape
+    }
+    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
+        self.source
+    }
+    pub fn contract(&self) -> BulkF32Contract {
+        BulkF32Contract::HaswellTinyBlasF32V1
+    }
+    pub fn view(&self, range: std::ops::Range<usize>) -> Result<&[f32]> {
+        self.values()?
+            .get(range)
+            .ok_or_else(|| Error::Malformed("bulk borrowed output range".into()))
+    }
+}
+pub(crate) fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
+    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        return Err(Error::State("closed bulk matrix cancelled".into()));
+    }
+    Ok(())
+}
diff --git a/crates/zerogravity-model-qwen35/src/forward/bulk_f32.rs b/crates/zerogravity-model-qwen35/src/forward/bulk_f32.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/forward/bulk_f32.rs
@@ -0,0 +1,11 @@
+//! One closed matrix fragment via the existing generic mathematical engine.
+use crate::Result;
+use crate::accelerator::program::{BulkF32Matrix, MathematicalBackend};
+pub(crate) fn bulk_f32_slice<M: MathematicalBackend>(
+    math: &mut M,
+    matrix: &BulkF32Matrix,
+) -> Result<M::Value> {
+    let left = math.bulk_f32_input(matrix.left())?;
+    let right = math.bulk_f32_input(matrix.right())?;
+    math.bulk_f32_matmul(matrix, &left, &right)
+}
diff --git a/crates/zerogravity-model-qwen35/src/forward.rs b/crates/zerogravity-model-qwen35/src/forward.rs
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -650,3 +650,9 @@
     }
     Ok(values)
 }
+
+#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
+#[path = "forward/bulk_f32.rs"]
+mod bulk_f32;
+#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
+pub(crate) use bulk_f32::bulk_f32_slice;
diff --git a/crates/zerogravity-model-qwen35/src/host_math/bulk_f32.rs b/crates/zerogravity-model-qwen35/src/host_math/bulk_f32.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/bulk_f32.rs
@@ -0,0 +1,225 @@
+//! Closed input/result backing under the SAME HostMath, no layer/token engine.
+use super::*;
+use crate::accelerator::program::{BulkF32Matrix, BulkF32Operand, BulkF32Side};
+use crate::operator_slice::bulk_f32::{BulkF32Slice, BulkInputs, BulkOwner, not_cancelled};
+use std::sync::atomic::AtomicBool;
+use zerogravity_cpu::math::haswell_bulk_f32::{
+    BulkF32Admission, BulkF32Completed, BulkF32Profile, BulkF32Request, F32View,
+};
+
+pub(super) struct BulkCall<'a> {
+    slice: &'a BulkF32Slice,
+    cancelled: Option<&'a AtomicBool>,
+}
+pub(super) enum BulkBacking {
+    Input {
+        inputs: Arc<BulkInputs>,
+        side: BulkF32Side,
+    },
+    Output {
+        completed: BulkF32Completed,
+        owner: Arc<BulkOwner>,
+    },
+}
+impl BulkBacking {
+    pub(super) fn values(&self) -> Result<&[f32]> {
+        match self {
+            Self::Input { inputs, side } => inputs.values(*side),
+            Self::Output { completed, .. } => Ok(completed.values()),
+        }
+    }
+    fn owner(&self) -> &Arc<BulkOwner> {
+        match self {
+            Self::Input { inputs, .. } => &inputs.owner,
+            Self::Output { owner, .. } => owner,
+        }
+    }
+}
+fn fp_environment() -> Result<()> {
+    let mut controls = 0u32;
+    // SAFETY: read-only control instruction writes one valid stack word.
+    unsafe {
+        std::arch::asm!("stmxcsr [{at}]",at=in(reg)&mut controls,options(nostack,preserves_flags));
+    }
+    if controls & 0xffc0 != 0x1f80 {
+        return Err(Error::Unsupported(
+            "closed Haswell profile requires unchanged RNE/noFTZ/noDAZ controls".into(),
+        ));
+    }
+    Ok(())
+}
+fn request_view<'a>(
+    source: crate::accelerator::SourceIdentity,
+    operand: &'a BulkF32Operand,
+    values: &'a [f32],
+) -> F32View<'a> {
+    let layout = operand.layout();
+    F32View {
+        source,
+        name: operand.name(),
+        values,
+        offset: layout.offset(),
+        shape: layout.shape(),
+        strides: layout.strides(),
+    }
+}
+impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
+    pub(crate) fn for_bulk_f32_slice(
+        slice: &'engine BulkF32Slice,
+        cancelled: Option<&'engine AtomicBool>,
+    ) -> Self {
+        let mut math = Self::for_operator_slice(slice.inputs.owner.model.as_ref(), None);
+        math.bulk_call = Some(BulkCall { slice, cancelled });
+        math
+    }
+    fn bulk_call(&self) -> Result<&BulkCall<'engine>> {
+        if self.engine.is_some()
+            || self.state.is_some()
+            || self.region_kernel.is_some()
+            || self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
+        {
+            return Err(Error::Unsupported(
+                "bulk F32 private same-engine host slice only".into(),
+            ));
+        }
+        self.bulk_call
+            .as_ref()
+            .ok_or_else(|| Error::Unsupported("bulk F32 has no bound temporal input issuer".into()))
+    }
+    pub(super) fn import_bulk_f32(&self, operand: &BulkF32Operand) -> Result<HostValue> {
+        let call = self.bulk_call()?;
+        not_cancelled(call.cancelled)?;
+        fp_environment()?;
+        let owner = &call.slice.inputs.owner;
+        let expected = match operand.side() {
+            BulkF32Side::Left => owner.matrix.left(),
+            BulkF32Side::Right => owner.matrix.right(),
+        };
+        if operand != expected || operand.span().source() != self.source.source_identity() {
+            return Err(Error::State(
+                "bulk input exact bound owner/span/layout".into(),
+            ));
+        }
+        let backing = BulkBacking::Input {
+            inputs: call.slice.inputs.clone(),
+            side: operand.side(),
+        };
+        if backing.values()?.len() != operand.layout().backing_words() {
+            return Err(Error::State("bulk actual input storage extent".into()));
+        }
+        HostValue::from_bulk(
+            backing,
+            operand.layout().backing_words(),
+            self.source.source_identity(),
+        )
+    }
+    pub(super) fn execute_bulk_f32(
+        &self,
+        matrix: &BulkF32Matrix,
+        left: &HostValue,
+        right: &HostValue,
+    ) -> Result<HostValue> {
+        let call = self.bulk_call()?;
+        let owner = &call.slice.inputs.owner;
+        not_cancelled(call.cancelled)?;
+        fp_environment()?;
+        if matrix != &owner.matrix {
+            return Err(Error::State("bulk exact bound matrix/temporal span".into()));
+        }
+        for (value, operand) in [(left, matrix.left()), (right, matrix.right())] {
+            let backing = value.data.bulk.as_ref().ok_or_else(|| {
+                Error::State("bulk input has no admitted original backing".into())
+            })?;
+            if !Arc::ptr_eq(backing.owner(), owner)
+                || value.source != Some(matrix.source())
+                || value.profile != Some(ProgramNumericalProfile::GgufSourceV1)
+                || value.precision != ActivationPrecision::F32
+                || value.elements != operand.layout().backing_words()
+                || value.view != TensorView::vector(0, value.elements)
+                || !matches!(backing,BulkBacking::Input{side,..}if *side==operand.side())
+            {
+                return Err(Error::State(
+                    "bulk foreign/partial/retagged input value".into(),
+                ));
+            }
+        }
+        let a = left.as_slice()?;
+        let b = right.as_slice()?;
+        let extent = |values: &[f32]| -> Result<std::ops::Range<usize>> {
+            let start = values.as_ptr() as usize;
+            let end = values
+                .len()
+                .checked_mul(4)
+                .and_then(|n| start.checked_add(n))
+                .ok_or_else(|| Error::Limit("bulk actual allocation range".into()))?;
+            Ok(start..end)
+        };
+        let ar = extent(a)?;
+        let br = extent(b)?;
+        if ar.start < br.end && br.start < ar.end {
+            return Err(Error::State(
+                "bulk independent input backing aliases".into(),
+            ));
+        }
+        // Unchanged source-derived primitive reserves actual result+scratch
+        // BEFORE its Vec allocation. No old MatVec/provider fallback exists.
+        let completed = zerogravity_cpu::math::haswell_bulk_f32::execute(
+            BulkF32Request {
+                profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
+                source: matrix.source(),
+                left: request_view(matrix.source(), matrix.left(), a),
+                right: request_view(matrix.source(), matrix.right(), b),
+            },
+            BulkF32Admission {
+                arbiter: &owner.admission.arbiter,
+                resource: &owner.admission.resource,
+                owner: &owner.admission.owner,
+            },
+            call.cancelled,
+        )?;
+        fp_environment()?;
+        not_cancelled(call.cancelled)?;
+        if completed.source() != matrix.source()
+            || completed.shape() != matrix.output().shape()
+            || completed.values().len() != matrix.output().backing_words()
+        {
+            return Err(Error::State(
+                "bulk actual primitive result geometry/source".into(),
+            ));
+        }
+        let value = HostValue::from_bulk(
+            BulkBacking::Output {
+                completed,
+                owner: owner.clone(),
+            },
+            matrix.output().backing_words(),
+            matrix.source(),
+        )?;
+        not_cancelled(call.cancelled)?;
+        Ok(value)
+    }
+}
+impl HostValue {
+    fn from_bulk(
+        backing: BulkBacking,
+        elements: usize,
+        source: crate::accelerator::SourceIdentity,
+    ) -> Result<Self> {
+        if !source.is_bound() || backing.values()?.len() != elements {
+            return Err(Error::State("bulk retained backing extent/source".into()));
+        }
+        Ok(Self {
+            data: Arc::new(HostStorage {
+                values: Vec::new(),
+                #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+                workspace: None,
+                bulk: Some(backing),
+            }),
+            view: TensorView::vector(0, elements),
+            elements,
+            precision: ActivationPrecision::F32,
+            source: Some(source),
+            profile: Some(ProgramNumericalProfile::GgufSourceV1),
+        })
+    }
+}
diff --git a/crates/zerogravity-model-qwen35/src/host_math.rs b/crates/zerogravity-model-qwen35/src/host_math.rs
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -22,6 +22,8 @@
 
 struct HostStorage {
     values: Vec<f32>,
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    bulk: Option<bulk_f32::BulkBacking>,
     // Payload drops before its actual same-source host reservation.
     #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
     workspace: Option<Arc<WindowWorkspace>>,
@@ -89,6 +91,8 @@
         Self {
             data: Arc::new(HostStorage {
                 values,
+                #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+                bulk: None,
                 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
                 workspace: None,
             }),
@@ -128,11 +132,31 @@
                 "host operator requires explicit contiguous input".into(),
             ));
         }
-        self.data
-            .get(self.view.offset..self.view.offset + self.elements)
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        let values = match &self.data.bulk {
+            Some(backing) => backing.values()?,
+            None => &self.data.values,
+        };
+        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
+        let values = &self.data.values;
+        values
+            .get(
+                self.view.offset
+                    ..self
+                        .view
+                        .offset
+                        .checked_add(self.elements)
+                        .ok_or_else(|| Error::Limit("host view end overflow".into()))?,
+            )
             .ok_or_else(|| Error::State("invalid private host tensor view".into()))
     }
     pub fn into_vec(self) -> Result<Vec<f32>> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if self.data.bulk.is_some() {
+            return Err(Error::Unsupported(
+                "leased bulk result has no Vec escape".into(),
+            ));
+        }
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         if self.data.workspace.is_some() {
             return Err(Error::Unsupported(
@@ -151,6 +175,8 @@
 
 pub(super) struct HostMath<'engine, 'state, 'trace> {
     source: HostSource<'engine>,
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    bulk_call: Option<bulk_f32::BulkCall<'engine>>,
     engine: Option<&'engine QwenDenseCpu>,
     state: Option<&'state mut QwenState>,
     region_kernel: Option<&'engine dyn crate::accelerator::MatVecKernel>,
@@ -168,6 +194,8 @@
         let position = state.position;
         Self {
             source: HostSource::Loaded(engine.model.as_ref()),
+            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+            bulk_call: None,
             engine: Some(engine),
             state: Some(state),
             region_kernel: engine.kernel.as_deref(),
@@ -183,6 +211,8 @@
     ) -> Self {
         Self {
             source: HostSource::Loaded(source),
+            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+            bulk_call: None,
             engine: None,
             state: None,
             region_kernel: kernel,
@@ -200,6 +230,8 @@
     ) -> Self {
         Self {
             source: HostSource::Windows(source),
+            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+            bulk_call: None,
             engine: None,
             state: None,
             region_kernel: kernel,
@@ -302,6 +334,12 @@
             .ok_or_else(|| Error::State("operator slice has no session state".into()))
     }
     fn legacy_precision(&self, values: &[&HostValue]) -> Result<()> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if values.iter().any(|v| v.data.bulk.is_some()) {
+            return Err(Error::Unsupported(
+                "bulk F32 values require explicit matrix contract".into(),
+            ));
+        }
         if self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
             || values
                 .iter()
@@ -342,6 +380,22 @@
 impl MathematicalBackend for HostMath<'_, '_, '_> {
     type Value = HostValue;
     type Index = usize;
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    fn bulk_f32_input(
+        &mut self,
+        operand: &crate::accelerator::program::BulkF32Operand,
+    ) -> Result<HostValue> {
+        self.import_bulk_f32(operand)
+    }
+    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+    fn bulk_f32_matmul(
+        &mut self,
+        matrix: &crate::accelerator::program::BulkF32Matrix,
+        left: &HostValue,
+        right: &HostValue,
+    ) -> Result<HostValue> {
+        self.execute_bulk_f32(matrix, left, right)
+    }
     fn elements(&self, value: &HostValue) -> usize {
         value.elements
     }
@@ -545,6 +599,12 @@
         ))
     }
     fn view(&mut self, input: &HostValue, view: TensorView) -> Result<HostValue> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if input.data.bulk.is_some() {
+            return Err(Error::Unsupported(
+                "bulk geometry cannot be reinterpreted by legacy view".into(),
+            ));
+        }
         // A view of an already strided view must first use explicit packing.
         if !input.view.is_contiguous() {
             return Err(Error::State(
@@ -568,6 +628,12 @@
         })
     }
     fn contiguous(&mut self, input: &HostValue) -> Result<HostValue> {
+        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+        if input.data.bulk.is_some() && input.view != TensorView::vector(0, input.elements) {
+            return Err(Error::Unsupported(
+                "bulk geometry cannot be implicitly packed".into(),
+            ));
+        }
         if input.view.is_contiguous() {
             return Ok(input.clone());
         }
@@ -1053,3 +1119,6 @@
 
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 mod gdn_conv;
+
+#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
+mod bulk_f32;
diff --git a/crates/zerogravity-model-qwen35/src/operator_slice.rs b/crates/zerogravity-model-qwen35/src/operator_slice.rs
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -234,3 +234,8 @@
 
 #[path = "gdn_conv_slice.rs"]
 pub mod gdn_conv;
+
+/// Explicit external/source F32 temporal matrix, no model/state eligibility.
+#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
+#[path = "bulk_f32_slice.rs"]
+pub mod bulk_f32;

````

## Source: .agents/runs/numerics-and-context-cpu-mapped-placement-control-20261020/V6_POLICY_DIFF.patch

Original UTF-8 bytes: 3415; SHA256 `ecf647843c511eba985a663ce622b0a17e6018185799b2b64df07c21153ebdd9`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- V6/native_ref_control_v6.cpp
+++ placement/native_cpu_placement.cpp
@@ -21,6 +21,16 @@
 #include <mutex>
 #include <string>
 #include <vector>
+#include <array>
+#include <set>
+#include <sstream>
+namespace cpu_placement {
+void require_compute_owner(ggml_backend_t);
+void before_backend_free(ggml_backend_t);
+void initialize(const char *);
+void finish();
+}
+
 
 namespace ref_control {
 constexpr const char *BASE="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml-base.so.0.26.0";
@@ -235,6 +245,7 @@
         ggml_backend_graph_plan_t plan,const char *api) {
     check_affinity();checked(scope==nullptr && (graph || plan),"one nonnested public CPU compute scope");
     std::lock_guard<std::mutex> guard(mutex);cpu_compute_only(backend,api);
+    cpu_placement::require_compute_owner(backend);
     checked(inflight==0,"one globally serialized original public compute");auto &record=known(backend);
     size_t plan_id=0;
     if(plan) {auto found=plans.find(plan);checked(found!=plans.end() && found->second.backend==backend,
@@ -260,6 +271,8 @@
 }
 }
 
+#include "placement_control.hpp"
+
 extern "C" ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device,const char *params) {
     const int saved=errno;
     static auto original=ref_control::symbol<decltype(&ggml_backend_dev_init)>(ref_control::BASE,
@@ -408,6 +421,7 @@
             && (record.kind==ref_control::Kind::Cpu?(record.abort==1 && record.set==2):(record.set==0 && record.abort==0)),
             "only exact configured CPU or unmodified BLAS retirement outside math");
         for(const auto &item:ref_control::plans)ref_control::checked(item.second.backend!=backend,"backend retains live opaque plan");
+        cpu_placement::before_backend_free(backend);
         record.freed=true;kind=record.kind;
     }
     errno=saved;original(backend);const int after=errno;
@@ -444,9 +458,11 @@
         require(ref_control::declared_cpus.size()==8,"exact eight declared CPUs");
         ref_control::check_affinity();
         ref_control::trace_path=std::string(argv[8])+"/use-ref-public-trace.jsonl";
+        cpu_placement::initialize(argv[8]);
         ref_control::armed=true;
         const int result=zg_unchanged_native_raw_main(argc,argv);
         require(result==0,"unchanged native capture failed");
+        cpu_placement::finish();
         std::lock_guard<std::mutex> guard(ref_control::mutex);
         require(ref_control::backends.size()==4 && ref_control::plans.empty()
             && ref_control::scope==nullptr && ref_control::sync_scope==nullptr && ref_control::inflight==0,
@@ -488,7 +504,7 @@
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
         report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
-            <<"\"control_source_revision\":\"public_use_ref_exact_blas_cpu_compute_legal_null_free_v6\","
+            <<"\"control_source_revision\":\"public_use_ref_distinct_cpu_mapped_placement_v1\","
             <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\","
             <<"\"CPU_backend_count\":2,\"BLAS_backend_count\":2,\"public_setter_calls\":4,"
             <<"\"existing_abort_setter_calls\":2,\"BLAS_compute_delegations\":0,"

````

## Source: .agents/runs/numerics-and-context-cpu-mapped-placement-control-20261020/frozen-prepared/V6_POLICY_DIFF.patch

Original UTF-8 bytes: 3415; SHA256 `ecf647843c511eba985a663ce622b0a17e6018185799b2b64df07c21153ebdd9`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- V6/native_ref_control_v6.cpp
+++ placement/native_cpu_placement.cpp
@@ -21,6 +21,16 @@
 #include <mutex>
 #include <string>
 #include <vector>
+#include <array>
+#include <set>
+#include <sstream>
+namespace cpu_placement {
+void require_compute_owner(ggml_backend_t);
+void before_backend_free(ggml_backend_t);
+void initialize(const char *);
+void finish();
+}
+
 
 namespace ref_control {
 constexpr const char *BASE="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml-base.so.0.26.0";
@@ -235,6 +245,7 @@
         ggml_backend_graph_plan_t plan,const char *api) {
     check_affinity();checked(scope==nullptr && (graph || plan),"one nonnested public CPU compute scope");
     std::lock_guard<std::mutex> guard(mutex);cpu_compute_only(backend,api);
+    cpu_placement::require_compute_owner(backend);
     checked(inflight==0,"one globally serialized original public compute");auto &record=known(backend);
     size_t plan_id=0;
     if(plan) {auto found=plans.find(plan);checked(found!=plans.end() && found->second.backend==backend,
@@ -260,6 +271,8 @@
 }
 }
 
+#include "placement_control.hpp"
+
 extern "C" ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device,const char *params) {
     const int saved=errno;
     static auto original=ref_control::symbol<decltype(&ggml_backend_dev_init)>(ref_control::BASE,
@@ -408,6 +421,7 @@
             && (record.kind==ref_control::Kind::Cpu?(record.abort==1 && record.set==2):(record.set==0 && record.abort==0)),
             "only exact configured CPU or unmodified BLAS retirement outside math");
         for(const auto &item:ref_control::plans)ref_control::checked(item.second.backend!=backend,"backend retains live opaque plan");
+        cpu_placement::before_backend_free(backend);
         record.freed=true;kind=record.kind;
     }
     errno=saved;original(backend);const int after=errno;
@@ -444,9 +458,11 @@
         require(ref_control::declared_cpus.size()==8,"exact eight declared CPUs");
         ref_control::check_affinity();
         ref_control::trace_path=std::string(argv[8])+"/use-ref-public-trace.jsonl";
+        cpu_placement::initialize(argv[8]);
         ref_control::armed=true;
         const int result=zg_unchanged_native_raw_main(argc,argv);
         require(result==0,"unchanged native capture failed");
+        cpu_placement::finish();
         std::lock_guard<std::mutex> guard(ref_control::mutex);
         require(ref_control::backends.size()==4 && ref_control::plans.empty()
             && ref_control::scope==nullptr && ref_control::sync_scope==nullptr && ref_control::inflight==0,
@@ -488,7 +504,7 @@
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
         report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
-            <<"\"control_source_revision\":\"public_use_ref_exact_blas_cpu_compute_legal_null_free_v6\","
+            <<"\"control_source_revision\":\"public_use_ref_distinct_cpu_mapped_placement_v1\","
             <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\","
             <<"\"CPU_backend_count\":2,\"BLAS_backend_count\":2,\"public_setter_calls\":4,"
             <<"\"existing_abort_setter_calls\":2,\"BLAS_compute_delegations\":0,"

````

## Source: .agents/runs/numerics-and-context-long-context-audit-20261020/FOLLOWUP_COMPARATOR.patch

Original UTF-8 bytes: 2011; SHA256 `6bbd719a95f8fd76eed10e5b25ee5ff87233a4617a9a2958b795faeae4f9cca8`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- frozen-prepared/compare_raw.py
+++ compare_raw.py
@@ -234,6 +234,7 @@
 
 
 def compare(args):
+    integer(args.cut, 1, 8192, "finite selected-token cut1..8192")
     require(args.model_sha256 in MODELS, "exact approved original model SHA required")
     require(args.raw_sha256 == hashlib.sha256(bounded(args.raw_ids, 8192 * 4)).hexdigest(),
             "caller-pinned raw IDs changed")
@@ -285,6 +286,8 @@
 def self_test():
     require(len(expected_geometry(1024, 3584)) == 730, "small geometry")
     require(len(expected_geometry(2048, 6144)) == 730, "large geometry")
+    for cut in [1, 2, 511, 512, 1023, 8191, 8192]:
+        require(integer(cut, 1, 8192, "selected-token cut") == cut, "bounded diagnostic cut")
     values = struct.pack("<III", 0x00000000, 0x3f800000, 0x3f800000)
     changed = struct.pack("<III", 0x80000000, 0x3f800001, 0x3f800000)
     result = compare_words(values, changed)
@@ -292,6 +295,10 @@
             and result["max_ulp"] == 1, "signedzero and ULP must remain exact mismatches")
     for action in [lambda: unique_object([("x", 1), ("x", 2)]),
                    lambda: integer(True, 1, 8, "bool"),
+                   lambda: integer(0, 1, 8192, "cut zero"),
+                   lambda: integer(8193, 1, 8192, "cut overflow"),
+                   lambda: integer(True, 1, 8192, "cut bool"),
+                   lambda: integer(2.0, 1, 8192, "cut float"),
                    lambda: compare_words(struct.pack("<I", 0x7f800000), struct.pack("<f", 0.0))]:
         try:
             action()
@@ -312,7 +319,7 @@
     parser.add_argument("--model-sha256")
     parser.add_argument("--raw-ids", type=Path)
     parser.add_argument("--raw-sha256")
-    parser.add_argument("--cut", type=int, choices=[512, 1024, 2048, 4096, 8192])
+    parser.add_argument("--cut", type=int, help="bounded selected-token cut1..8192; tiny cuts are instrumentation/bisection only")
     parser.add_argument("--output", type=Path)
     args = parser.parse_args()
     if args.self_test:

````

