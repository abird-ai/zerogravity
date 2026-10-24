# Draft Patches — volume 18

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/CLEANUP_V2.patch

Original UTF-8 bytes: 8904; SHA256 `28598954899beec157e29ddad844728045bc943bfc3bf5f5066dda83eb26da7e`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- unchanged-run_replay.py
+++ additive-run_replay_v2.py
@@ -14,52 +14,91 @@
 LIB = '/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib'
 
 
-def group_exists(pgid):
+def unreaped(child):
+    require(child.returncode is None, "leader already reaped")
+    value = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if value is None:
+        return None
+    require(value.si_pid == child.pid, "owned child identity")
+    return value.si_status if value.si_code == os.CLD_EXITED else -value.si_status
+
+
+def cleanup_owned(child, record, errors, *, observe=unreaped, pause=time.sleep):
+    """Independent bounded stages; never signal after the first reap attempt."""
+    def bound():
+        observe(child)
+        require(os.getpgid(child.pid) == child.pid and os.getsid(child.pid) == child.pid,
+                "owned session/group binding")
+        return child.pid
+
     try:
-        os.killpg(pgid, 0)
-        return True
+        os.killpg(bound(), signal.SIGTERM)
+        record["group_term_before_reap"] = True
     except ProcessLookupError:
-        return False
-
-
-def signal_group(pgid, sig):
+        record["group_term_absent"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
     try:
-        os.killpg(pgid, sig)
-        return True
-    except ProcessLookupError:
-        return False
+        grace = time.monotonic() + 0.5
+        while time.monotonic() < grace:
+            observe(child)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # TERM/grace/one KILL error cannot skip the separately bound KILL stage.
+    # Both attempts occur before ANY wait, while the owned leader is unreaped.
+    for attempt in range(2):
+        try:
+            os.killpg(bound(), signal.SIGKILL)
+            record["group_kill_before_reap"] = True
+            break
+        except ProcessLookupError:
+            record["group_kill_absent_before_reap"] = True
+            break
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    # ALWAYS attempt reap even after signal/ownership failures. Popen.wait only
+    # targets this child. An interrupted first wait cannot skip the final wait.
+    # No signal operation occurs below this point, including after timeout.
+    try:
+        record["reaped_exit_code"] = child.wait(timeout=5)
+        record["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned initial reap: " + repr(error))
+    finally:
+        if child.returncode is None:
+            try:
+                record["reaped_exit_code"] = child.wait(timeout=5)
+                record["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final reap: " + repr(error))
+        else:
+            record["reaped_exit_code"] = child.returncode
+            record["leader_reaped"] = True
+    if record.get("leader_reaped"):
+        try:
+            deadline = time.monotonic() + 5
+            while True:
+                try:
+                    os.killpg(child.pid, 0)  # Read-only absence check, NEVER kill after reap.
+                except ProcessLookupError:
+                    record["group_absent"] = True
+                    break
+                require(time.monotonic() < deadline, "owned process group remains")
+                pause(0.05)
+        except BaseException as error:
+            errors.append("owned group absence: " + repr(error))
+
 
 
 def leader_exit_without_reaping(process, seconds):
-    # Keep the owned leader PID reserved until group cleanup begins.
     deadline = time.monotonic() + seconds
     while time.monotonic() < deadline:
-        observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+        observed = unreaped(process)
         if observed is not None:
-            return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+            return observed
         time.sleep(0.05)
     raise TimeoutError('owned leader exit deadline')
-
-
-def cleanup_owned_group(process, receipt):
-    pgid = process.pid  # start_new_session=True; never an inherited/shared group.
-    cleanup = receipt['cleanup'] = {'pgid': pgid, 'term_sent': False, 'kill_sent': False,
-                                    'leader_reaped': False, 'group_absent': False}
-    cleanup['term_sent'] = signal_group(pgid, signal.SIGTERM)
-    try:
-        process.wait(timeout=3)
-    except subprocess.TimeoutExpired:
-        cleanup['kill_sent'] = signal_group(pgid, signal.SIGKILL)
-        process.wait(timeout=5)
-    cleanup['leader_reaped'] = True
-    # A reaped leader does not establish that its owned descendants retired.
-    if group_exists(pgid):
-        cleanup['kill_sent'] = signal_group(pgid, signal.SIGKILL) or cleanup['kill_sent']
-    deadline = time.monotonic() + 5
-    while group_exists(pgid) and time.monotonic() < deadline:
-        time.sleep(0.05)
-    cleanup['group_absent'] = not group_exists(pgid)
-    require(cleanup['group_absent'], 'owned process group remains after cleanup')
 
 
 def finish_pipe(process, log, log_bytes, receipt):
@@ -92,6 +131,14 @@
     p.add_argument('--preload', type=Path)
     p.add_argument('--preload-sha256')
     a = p.parse_args()
+    followup = jread(HERE/'cleanup-v2-manifest.json')
+    require(followup['original_freeze_sha256'] == 'ee78cf570d4d9f3f48378a90c7a916436eeb3791906ca8c3a1c03e188b3cce60'
+            and digest(HERE/'freeze-manifest.json')['sha256'] == followup['original_freeze_sha256'],
+            'unchanged original alpha preparation manifest')
+    for name, expected in jread(HERE/'freeze-manifest.json')['files'].items():
+        require(digest(HERE/name) == expected,'unchanged original alpha prepared source')
+    for name, expected in followup['files'].items():
+        require(digest(HERE/name) == expected,'exact cleanup followup source pin')
     pins = authenticate()
     capture_guards()
     for path in [a.new_output, a.new_log, a.new_receipt]:
@@ -123,6 +170,8 @@
                'actual_gpu_or_model_execution': False, 'internal_dispatch_observed': False,
                'preload': str(a.preload.resolve()) if a.preload else None, 'preload_digest': before_preload,
                'source_profile': 'pinned llama.cpp0.6.0 haswell public alpha default/control diagnostic',
+               'runner_sha256':digest(Path(__file__)),
+               'cleanup_v2_manifest_sha256':digest(HERE/'cleanup-v2-manifest.json'),
                'exit_code': None, 'status': 'UNQUALIFIED'}
     with a.new_receipt.open('x') as receipt_file, a.new_log.open('xb') as log:
         process = None
@@ -158,14 +207,18 @@
             receipt['error'] = f'{type(error).__name__}: {error}'
             primary_error = error
         finally:
+            cleanup_errors = []
             try:
                 if process is not None:
-                    cleanup_owned_group(process, receipt)
+                    receipt['cleanup'] = {'pgid':process.pid}
+                    cleanup_owned(process, receipt['cleanup'], cleanup_errors)
+            except BaseException as error:
+                cleanup_errors.append('unexpected owned cleanup: '+repr(error))
+            try:
+                if process is not None:
                     finish_pipe(process, log, log_bytes, receipt)
             except BaseException as error:
-                cleanup_error = error
-                receipt['cleanup_error'] = f'{type(error).__name__}: {error}'
-                receipt['status'] = 'UNQUALIFIED'
+                cleanup_errors.append('final owned pipe drain: '+repr(error))
             finally:
                 closers = []
                 if selector is not None:
@@ -178,10 +231,14 @@
                         receipt.setdefault('cleanup', {})[label] = True
                     except BaseException as error:
                         cleanup_error = cleanup_error or error
-                        receipt.setdefault('cleanup_errors', []).append(f'{label}: {type(error).__name__}: {error}')
+                        cleanup_errors.append(f'{label}: {type(error).__name__}: {error}')
                         receipt['status'] = 'UNQUALIFIED'
                 if process is not None:
                     receipt['exit_code'] = process.returncode
+            if cleanup_errors:
+                receipt['cleanup_errors'] = cleanup_errors
+                receipt['status'] = 'UNQUALIFIED'
+                cleanup_error = cleanup_error or RuntimeError('; '.join(cleanup_errors))
             receipt['elapsed_seconds'] = time.monotonic() - started
             json.dump(receipt, receipt_file, indent=2, allow_nan=False)
             receipt_file.write('\n')

````

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 113153; SHA256 `25fb733d7171fbf5fe07ad09525f63b92923e491804d36db5db775c7ca9a27e0`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ original2B-alpha-public-replay/main.cpp
@@ -0,0 +1,247 @@
+// SPDX-License-Identifier: Apache-2.0
+// Diagnostic public ggml operator only; upstream headers retain MIT notices.
+// No model load, arithmetic replacement, production linkage or eligibility.
+#include "ggml.h"
+#include "ggml-alloc.h"
+#include "ggml-backend.h"
+#include "ggml-cpu.h"
+#include "observer.h"
+#include <cstdlib>
+#include <array>
+#include <chrono>
+#include <cstring>
+#include <dlfcn.h>
+#include <filesystem>
+#include <fstream>
+#include <iostream>
+#include <limits>
+#include <memory>
+#include <stdexcept>
+#include <string>
+#include <sys/resource.h>
+#include <vector>
+
+namespace fs = std::filesystem;
+constexpr size_t K = 2048, M = 16;
+constexpr size_t W_BYTES = K * M * sizeof(float);
+constexpr size_t ARTIFACT_CAP = 512 * 1024 * 1024;
+static size_t written = 0;
+static bool tracing = false;
+static decltype(&zg_alpha_trace_begin) trace_begin = nullptr;
+static decltype(&zg_alpha_trace_end) trace_end = nullptr;
+static decltype(&zg_alpha_trace_save) trace_save = nullptr;
+static void load_observer() {
+    const char *enabled = std::getenv("ZG_ALPHA_TRACE");
+    if (!enabled) return;
+    if (std::strcmp(enabled, "1") != 0) throw std::runtime_error("exact observer enable value");
+    void *begin = dlsym(RTLD_DEFAULT, "zg_alpha_trace_begin");
+    void *end = dlsym(RTLD_DEFAULT, "zg_alpha_trace_end");
+    void *save = dlsym(RTLD_DEFAULT, "zg_alpha_trace_save");
+    if (!begin || !end || !save) throw std::runtime_error("explicit preloaded observer unavailable");
+    static_assert(sizeof(begin) == sizeof(trace_begin));
+    static_assert(sizeof(end) == sizeof(trace_end));
+    static_assert(sizeof(save) == sizeof(trace_save));
+    std::memcpy(&trace_begin, &begin, sizeof(trace_begin));
+    std::memcpy(&trace_end, &end, sizeof(trace_end));
+    std::memcpy(&trace_save, &save, sizeof(trace_save));
+    tracing = true;
+}
+static std::chrono::steady_clock::time_point deadline;
+
+static void require(bool yes, const std::string &message) {
+    if (!yes) throw std::runtime_error(message);
+}
+static bool expired(void *) { return std::chrono::steady_clock::now() >= deadline; }
+static std::vector<unsigned char> read_exact(const fs::path &path, size_t bytes) {
+    require(!fs::is_symlink(path) && fs::is_regular_file(path) && fs::file_size(path) == bytes,
+            "exact regular input: " + path.string());
+    std::vector<unsigned char> result(bytes);
+    std::ifstream in(path, std::ios::binary);
+    require(bool(in), "input open");
+    in.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(bytes));
+    require(bool(in) && in.peek() == std::char_traits<char>::eof(), "exact immutable-length input");
+    for (size_t at = 0; at < bytes; at += 4) {
+        uint32_t word = 0; std::memcpy(&word, result.data() + at, 4);
+        require((word & 0x7f800000u) != 0x7f800000u, "finite actual operand/output");
+    }
+    return result;
+}
+static void write_bytes(const fs::path &path, const void *data, size_t bytes) {
+    require(written <= ARTIFACT_CAP && bytes <= ARTIFACT_CAP - written, "aggregate output cap");
+    require(!fs::exists(path), "exclusive output file");
+    std::ofstream out(path, std::ios::binary);
+    require(bool(out), "output open");
+    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
+    out.flush(); require(bool(out), "output write"); written += bytes;
+}
+static void maps(const fs::path &out, const fs::path &cpu) {
+    std::ifstream in("/proc/self/maps"); require(bool(in), "maps open");
+    std::string text, line; bool found = false;
+    while (std::getline(in, line)) {
+        require(text.size() + line.size() + 1 <= 1024 * 1024, "bounded maps");
+        if (line.find(cpu.string()) != std::string::npos) found = true;
+        if (line.find("libggml-cpu-") != std::string::npos)
+            require(line.find(cpu.string()) != std::string::npos, "foreign CPU backend");
+        text += line + '\n';
+    }
+    require(in.eof() && found, "actual pinned CPU backend mapped");
+    write_bytes(out, text.data(), text.size());
+}
+static std::string name(size_t call) {
+    require(call < 64, "actual AR1 index");
+    return "ar1-" + std::to_string(call);
+}
+static int positive(const char *argument, int maximum) {
+    const std::string text(argument); size_t used = 0;
+    const long value = std::stol(text, &used);
+    require(used == text.size() && value >= 1 && value <= maximum, "bounded exact positive argument");
+    return static_cast<int>(value);
+}
+struct Result { size_t unequal = 0; size_t first = 0; uint32_t actual = 0, expected = 0; };
+static Result compare(const std::vector<unsigned char> &actual, const std::vector<unsigned char> &expected) {
+    require(actual.size() == expected.size(), "complete output geometry");
+    Result result;
+    for (size_t at = 0; at < actual.size(); at += 4) {
+        uint32_t a, b; std::memcpy(&a, actual.data() + at, 4); std::memcpy(&b, expected.data() + at, 4);
+        if (a != b) {
+            if (result.unequal == 0) { result.first = at / 4; result.actual = a; result.expected = b; }
+            ++result.unequal;
+        }
+    }
+    return result;
+}
+static std::vector<unsigned char> replay(ggml_backend_t backend, size_t rows, uint32_t phase, uint32_t which,
+                                       const std::vector<unsigned char> &wraw,
+                                       const std::vector<unsigned char> &xraw) {
+    require(rows == 1 || rows == 64, "unchanged actual call width");
+    require(wraw.size() == W_BYTES && xraw.size() == K * rows * 4, "actual alpha input geometry");
+    require(!expired(nullptr), "finite replay deadline");
+    ggml_init_params init{}; init.mem_size = 2 * 1024 * 1024; init.no_alloc = true;
+    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(init), ggml_free);
+    require(bool(ctx), "bounded graph metadata allocation");
+    auto *w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
+    auto *x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, rows);
+    auto *y = ggml_mul_mat(ctx.get(), w, x);
+    const std::array<int64_t, 4> wne{2048,16,1,1}, xne{2048,static_cast<int64_t>(rows),1,1};
+    const std::array<int64_t, 4> yne{16,static_cast<int64_t>(rows),1,1};
+    const std::array<size_t, 4> wnb{4,8192,131072,131072}, xnb{4,8192,K*rows*4,K*rows*4};
+    const std::array<size_t, 4> ynb{4,64,M*rows*4,M*rows*4};
+    for(size_t i=0;i<4;++i) {
+        require(w->ne[i]==wne[i] && w->nb[i]==wnb[i], "exact captured original F32 weight layout");
+        require(x->ne[i]==xne[i] && x->nb[i]==xnb[i], "exact captured original alpha input layout");
+        require(y->ne[i]==yne[i] && y->nb[i]==ynb[i], "exact captured original alpha output layout");
+    }
+    require(w->type==GGML_TYPE_F32 && x->type==GGML_TYPE_F32 && y->type==GGML_TYPE_F32
+            && y->op==GGML_OP_MUL_MAT && ggml_is_contiguous(w) && ggml_is_contiguous(x)
+            && ggml_is_contiguous(y), "original public F32 alpha projection");
+    auto *graph=ggml_new_graph_custom(ctx.get(),16,false); require(graph, "graph metadata");
+    ggml_build_forward_expand(graph,y);
+    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
+        ggml_backend_alloc_ctx_tensors(ctx.get(),backend),ggml_backend_buffer_free);
+    require(bool(buffer) && ggml_backend_buffer_get_size(buffer.get()) <= 2*1024*1024,
+            "bounded genuine CPU backing allocation");
+    ggml_backend_tensor_set(w,wraw.data(),0,wraw.size());
+    ggml_backend_tensor_set(x,xraw.data(),0,xraw.size());
+    if(tracing)require(trace_begin(phase,which,static_cast<uint32_t>(rows),w->data,x->data,y->data),"observer begin");
+    const auto status=ggml_backend_graph_compute(backend,graph);
+    if(tracing)require(trace_end(),"observer end after synchronous completion");
+    require(status==GGML_STATUS_SUCCESS,"public alpha operator completion");
+    std::vector<unsigned char> result(M*rows*4),after_w(wraw.size()),after_x(xraw.size());
+    ggml_backend_tensor_get(y,result.data(),0,result.size());
+    ggml_backend_tensor_get(w,after_w.data(),0,after_w.size());
+    ggml_backend_tensor_get(x,after_x.data(),0,after_x.size());
+    require(after_w==wraw && after_x==xraw,"operator changed actual original input bytes");
+    for(size_t at=0;at<result.size();at+=4) {
+        uint32_t word;std::memcpy(&word,result.data()+at,4);
+        require((word&0x7f800000u)!=0x7f800000u,"finite public alpha result");
+    }
+    return result;
+}
+
+int main(int argc, char **argv) {
+    try {
+        load_observer();
+        require(argc == 6, "usage: replay FIXTURES NEW_OUTPUT CPU_DSO THREADS SECONDS");
+        const fs::path input = fs::canonical(argv[1]), output = argv[2], cpu = fs::canonical(argv[3]);
+        require(cpu == "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so",
+                "explicit pinned original CPU backend");
+        const int threads = positive(argv[4], 8), seconds = positive(argv[5], 300);
+        deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
+        rlimit address{8ull * 1024 * 1024 * 1024, 8ull * 1024 * 1024 * 1024};
+        rlimit cpu_limit{static_cast<rlim_t>(seconds * threads + 1), static_cast<rlim_t>(seconds * threads + 1)};
+        rlimit file_limit{ARTIFACT_CAP, ARTIFACT_CAP};
+        require(setrlimit(RLIMIT_AS, &address) == 0 && setrlimit(RLIMIT_CPU, &cpu_limit) == 0
+                && setrlimit(RLIMIT_FSIZE, &file_limit) == 0, "finite process limits");
+        const uint32_t endian = 1;
+        require(*reinterpret_cast<const unsigned char *>(&endian) == 1 && sizeof(float) == 4, "LE F32 host");
+        require(fs::create_directory(output), "new exclusive output directory");
+        auto *reg = ggml_backend_load(cpu.c_str());
+        require(reg && std::string(ggml_backend_reg_name(reg)) == "CPU" && ggml_backend_reg_dev_count(reg) == 1,
+                "original public CPU backend registration");
+        auto get = [reg](const char *key) { auto *p = ggml_backend_reg_get_proc_address(reg, key); require(p, key); return p; };
+        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(get("ggml_backend_set_n_threads"));
+        auto set_abort = reinterpret_cast<ggml_backend_set_abort_callback_t>(get("ggml_backend_set_abort_callback"));
+        using SetRef = decltype(&ggml_backend_cpu_set_use_ref);
+        auto set_ref = reinterpret_cast<SetRef>(get("ggml_backend_cpu_set_use_ref"));
+        Dl_info info{};
+        require(dladdr(reinterpret_cast<void *>(set_ref), &info) != 0 && info.dli_fname
+                && fs::canonical(info.dli_fname) == cpu, "public control belongs to exact CPU DSO");
+        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
+            ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr), ggml_backend_free);
+        require(bool(backend), "original CPU backend owner");
+        set_threads(backend.get(), threads); set_abort(backend.get(), expired, nullptr);
+        maps(output / "before.maps", cpu);
+        std::ofstream report(output / "cases.jsonl"); require(bool(report), "case report open");
+        std::vector<std::vector<unsigned char>> ar_actual(64);
+        size_t defaults = 0, controls = 0;
+        for (int control = 0; control < 2; ++control) {
+            // Ref control is unreachable until all65 complete default outputs matched.
+            if (control) require(defaults == 65, "default all-output gate before ref control");
+            set_ref(backend.get(), control != 0);
+            for (size_t which = 0; which < 65; ++which) {
+                require(!expired(nullptr), "bounded replay deadline");
+                const std::string key = which == 0 ? "bulk64" : name(which - 1);
+                const size_t rows = which == 0 ? 64 : 1, bytes = M * rows * 4;
+                const auto w = read_exact(input / "weight.f32le", W_BYTES);
+                const auto x = read_exact(input / (key + ".x.f32le"), K * rows * 4);
+                const auto actual = replay(backend.get(), rows, static_cast<uint32_t>(control), static_cast<uint32_t>(which), w, x);
+                // Expected output is first opened after genuine public computation.
+                const auto expected = read_exact(input / (key + ".expected.f32le"), bytes);
+                const Result result = compare(actual, expected);
+                const std::string prefix = control ? "ref-control-" : "default-";
+                write_bytes(output / (prefix + key + ".actual.f32le"), actual.data(), actual.size());
+                Result ar_comparison{};
+                if (!control && which) ar_actual[which - 1] = actual;
+                if (control && which == 0) {
+                    for (size_t at = 0; at < 64; ++at) {
+                        std::vector<unsigned char> slice(M*4);
+                        std::memcpy(slice.data(),actual.data()+at*M*4,M*4);
+                        ar_comparison.unequal += compare(slice, ar_actual[at]).unequal;
+                    }
+                }
+                report << "{\"mode\":\"" << (control ? "PUBLIC_USE_REF_CONTROL" : "DEFAULT_PUBLIC_ALPHA")
+                       << "\",\"case\":\"" << key << "\",\"words\":" << bytes / 4
+                       << ",\"actual_capture_unequal_words\":" << result.unequal
+                       << ",\"first_index\":" << result.first << ",\"first_actual_bits\":" << result.actual
+                       << ",\"first_capture_bits\":" << result.expected
+                       << ",\"ref_bulk_vs_all_default_ar1_unequal_words\":" << ar_comparison.unequal << "}\n";
+                report.flush(); require(bool(report), "case report write");
+                if (!control) { require(result.unequal == 0, "default original public operator differs from capture"); ++defaults; }
+                else ++controls;
+                require(w == read_exact(input / "weight.f32le", W_BYTES)
+                        && x == read_exact(input / (key + ".x.f32le"), K*rows*4), "file inputs changed during operator");
+            }
+        }
+        if (tracing) require(trace_save((output / "sgemm-trace.json").c_str()), "exclusive bounded observer trace");
+        maps(output / "after.maps", cpu);
+        require(defaults == 65 && controls == 65, "complete actual default/control populations");
+        std::ofstream done(output / "complete.json"); require(bool(done), "completion open");
+        done << "{\"status\":\"PUBLIC_ALPHA_REPLAY_DEFAULT_EXACT_CONTROL_DIAGNOSTIC\",\"default_cases\":65,\"control_cases\":65,\"threads\":"
+             << threads << ",\"default_words\":2048,\"cause_qualified\":false,\"production_math_changed\":false,\"internal_dispatch_observed\":false}\n";
+        done.flush(); require(bool(done), "completion write");
+        std::cout << "PUBLIC_ALPHA_REPLAY_DEFAULT_EXACT_CONTROL_DIAGNOSTIC defaults=65 words=2048\n";
+        return 0;
+    } catch (const std::exception &error) {
+        std::cerr << "PUBLIC_ALPHA_REPLAY_UNQUALIFIED " << error.what() << '\n'; return 3;
+    }
+}
--- /dev/null
+++ original2B-alpha-public-replay/observer.cpp
@@ -0,0 +1,143 @@
+// SPDX-License-Identifier: Apache-2.0
+// Diagnostic only. Exact exported source declaration; params remains opaque.
+#include "observer.h"
+#include "sgemm.h"
+#include <atomic>
+#include <cstdint>
+#include <cstring>
+#include <dlfcn.h>
+#include <fcntl.h>
+#include <sstream>
+#include <string>
+#include <unistd.h>
+
+namespace {
+constexpr const char *CPU = "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so";
+constexpr uint32_t MAX_CALLS = 2048, MAX_CASES = 130;
+using Original = decltype(&llamafile_sgemm);
+std::atomic<Original> original{nullptr};
+static_assert(std::atomic<Original>::is_always_lock_free);
+void *original_owner = nullptr; // Bounded NOLOAD reference retained to process exit.
+std::atomic<bool> active{false}, failed{false};
+std::atomic<uint32_t> call_count{0};
+static_assert(std::atomic<uint32_t>::is_always_lock_free);
+struct Context { uint32_t phase, which, width, first; uintptr_t a, b, c; } context{};
+struct Case { uint32_t phase, which, width, first, last; } cases[MAX_CASES]{};
+uint32_t case_count = 0;
+struct Call {
+    uint32_t phase, which;
+    int64_t m, n, k, lda, ldb, ldc;
+    uintptr_t a, b, c;
+    int atype, btype, ctype;
+    bool accepted;
+};
+Call calls[MAX_CALLS]{};
+
+bool resolve_original() {
+    if (original.load(std::memory_order_acquire)) return true;
+    original_owner = dlopen(CPU, RTLD_NOW | RTLD_NOLOAD);
+    if (!original_owner) return false;
+    void *symbol = dlsym(original_owner, "llamafile_sgemm");
+    Dl_info info{};
+    if (!symbol || !dladdr(symbol, &info) || !info.dli_fname || !info.dli_fbase
+        || std::strcmp(info.dli_fname, CPU) != 0
+        || reinterpret_cast<uintptr_t>(symbol) - reinterpret_cast<uintptr_t>(info.dli_fbase) != 0x10c8f0u)
+        return false;
+    Original function = nullptr;
+    static_assert(sizeof(symbol) == sizeof(function));
+    std::memcpy(&function, &symbol, sizeof(function));
+    original.store(function, std::memory_order_release);
+    return true;
+}
+
+bool write_all(int fd, const char *bytes, size_t length) {
+    while (length) {
+        const ssize_t n = write(fd, bytes, length);
+        if (n <= 0) return false;
+        bytes += n; length -= static_cast<size_t>(n);
+    }
+    return true;
+}
+} // namespace
+
+extern "C" bool zg_alpha_trace_begin(uint32_t phase, uint32_t which, uint32_t width,
+                                  const void *a, const void *b, const void *c) {
+    if (active.load(std::memory_order_acquire) || failed.load(std::memory_order_acquire)
+        || phase > 1 || which >= 65 || width != (which ? 1u : 64u)
+        || case_count != phase * 65 + which || !a || !b || !c || !resolve_original()) {
+        failed.store(true, std::memory_order_release); return false;
+    }
+    context = {phase, which, width, call_count.load(std::memory_order_acquire),
+               reinterpret_cast<uintptr_t>(a), reinterpret_cast<uintptr_t>(b), reinterpret_cast<uintptr_t>(c)};
+    active.store(true, std::memory_order_release);
+    return true;
+}
+
+extern "C" bool llamafile_sgemm(const struct ggml_compute_params *params,
+        int64_t m, int64_t n, int64_t k, const void *a, int64_t lda,
+        const void *b, int64_t ldb, void *c, int64_t ldc, int atype, int btype, int ctype) {
+    // Resolution occurs on the owner thread before the real graph is invoked.
+    // Never return false as an observer failure: that would change dispatch.
+    const Original function = original.load(std::memory_order_acquire);
+    if (!function) _exit(120);
+    const bool observed = active.load(std::memory_order_acquire);
+    const Context bound = observed ? context : Context{};
+    const bool accepted = function(params, m, n, k, a, lda, b, ldb, c, ldc, atype, btype, ctype);
+    const uint32_t index = call_count.fetch_add(1, std::memory_order_acq_rel);
+    if (!observed || index >= MAX_CALLS
+        || reinterpret_cast<uintptr_t>(a) < bound.a
+        || reinterpret_cast<uintptr_t>(b) < bound.b
+        || reinterpret_cast<uintptr_t>(c) < bound.c) {
+        failed.store(true, std::memory_order_release);
+    } else {
+        calls[index] = {bound.phase, bound.which, m, n, k, lda, ldb, ldc,
+                        reinterpret_cast<uintptr_t>(a) - bound.a,
+                        reinterpret_cast<uintptr_t>(b) - bound.b,
+                        reinterpret_cast<uintptr_t>(c) - bound.c, atype, btype, ctype, accepted};
+    }
+    return accepted;
+}
+
+extern "C" bool zg_alpha_trace_end(void) {
+    if (!active.exchange(false, std::memory_order_acq_rel) || case_count >= MAX_CASES) {
+        failed.store(true, std::memory_order_release); return false;
+    }
+    // The original synchronous graph-compute joins all worker calls first.
+    cases[case_count++] = {context.phase, context.which, context.width, context.first,
+                          call_count.load(std::memory_order_acquire)};
+    return !failed.load(std::memory_order_acquire);
+}
+
+extern "C" bool zg_alpha_trace_save(const char *path) {
+    if (!path || active.load(std::memory_order_acquire) || failed.load(std::memory_order_acquire)
+        || case_count != MAX_CASES || call_count.load(std::memory_order_acquire) > MAX_CALLS)
+        return false;
+    std::ostringstream out;
+    out << "{\"scope\":\"exact exported llamafile_sgemm entry passthrough; opaque params\","
+        << "\"internal_microkernel_observed\":false,\"math_replaced\":false,\"cpu\":\"" << CPU
+        << "\",\"symbol_offset\":1100016,\"case_count\":" << case_count
+        << ",\"call_count\":" << call_count.load(std::memory_order_acquire) << ",\"cases\":[";
+    for (uint32_t i = 0; i < case_count; ++i) {
+        const auto &r = cases[i];
+        if (i) out << ',';
+        out << "{\"phase\":" << r.phase << ",\"which\":" << r.which << ",\"width\":" << r.width
+            << ",\"first\":" << r.first << ",\"last\":" << r.last << '}';
+    }
+    out << "],\"calls\":[";
+    for (uint32_t i = 0; i < call_count.load(std::memory_order_acquire); ++i) {
+        const auto &r = calls[i];
+        if (i) out << ',';
+        out << "{\"phase\":" << r.phase << ",\"which\":" << r.which << ",\"m\":" << r.m
+            << ",\"n\":" << r.n << ",\"k\":" << r.k << ",\"lda\":" << r.lda << ",\"ldb\":" << r.ldb
+            << ",\"ldc\":" << r.ldc << ",\"a_offset\":" << r.a << ",\"b_offset\":" << r.b
+            << ",\"c_offset\":" << r.c << ",\"atype\":" << r.atype << ",\"btype\":" << r.btype
+            << ",\"ctype\":" << r.ctype << ",\"accepted\":" << (r.accepted ? "true" : "false") << '}';
+    }
+    out << "]}\n";
+    const std::string bytes = out.str();
+    if (bytes.size() > 1024 * 1024) return false;
+    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
+    if (fd < 0) return false;
+    const bool ok = write_all(fd, bytes.data(), bytes.size());
+    return close(fd) == 0 && ok;
+}
--- /dev/null
+++ original2B-alpha-public-replay/observer.h
@@ -0,0 +1,14 @@
+// SPDX-License-Identifier: Apache-2.0
+#pragma once
+#include <stdbool.h>
+#include <stdint.h>
+#ifdef __cplusplus
+extern "C" {
+#endif
+bool zg_alpha_trace_begin(uint32_t phase, uint32_t which, uint32_t width,
+                       const void *a, const void *b, const void *c);
+bool zg_alpha_trace_end(void);
+bool zg_alpha_trace_save(const char *new_path);
+#ifdef __cplusplus
+}
+#endif
--- /dev/null
+++ original2B-alpha-public-replay/prepare.py
@@ -0,0 +1,205 @@
+#!/usr/bin/env python3
+"""ROOT ONLY: byte-copy authentic alpha operands/results, no operator/model calls."""
+import argparse
+from functools import cache
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import struct
+
+HERE = Path(__file__).resolve().parent
+CAPTURE = HERE.parent / 'numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport'
+MODEL = HERE.parents[2] / 'target/models/qwen35-2b/model.gguf'
+CPU = '/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so'
+W_OFFSET, W_BYTES, X_BYTES = 459622304, 131072, 8192
+FIXTURE_CAP = 4 * 1024 * 1024
+
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+
+def digest(path):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
+        before = os.fstat(stream.fileno())
+        require(stat.S_ISREG(before.st_mode), 'regular immutable source')
+        total, h = 0, hashlib.sha256()
+        while chunk := stream.read(1024 * 1024):
+            total += len(chunk)
+            h.update(chunk)
+        after = os.fstat(stream.fileno())
+        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
+                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
+                and total == before.st_size, 'source changed during hash')
+    return {'bytes': total, 'sha256': h.hexdigest()}
+
+
+def bounded(path, cap):
+    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= cap, 'bounded regular source')
+    raw = path.read_bytes()
+    require(len(raw) <= cap, 'bounded immutable-length read')
+    return raw
+
+
+def jread(path, cap=1024 * 1024):
+    return json.loads(bounded(path, cap), object_pairs_hook=unique)
+
+
+def authenticate():
+    pins = jread(HERE / 'source-pins.json')
+    require(pins['capture'] == str(CAPTURE) and pins['cpu_dso'] == CPU, 'closed original capture/backend')
+    for path, expected in pins['inputs'].items():
+        require(digest(path) == expected, 'source/content pin changed: ' + path)
+    return pins
+
+
+@cache
+def metadata(mode):
+    require(mode in ['bulk64', 'ar1'], 'closed actual mode')
+    return [json.loads(line, object_pairs_hook=unique) for line in
+            bounded(CAPTURE / mode / 'records.jsonl', 16 * 1024 * 1024).splitlines()]
+
+
+def actual(mode, output, call):
+    stage = 'projection_' + ('output' if output else 'input') + '::blk.0.ssm_alpha.weight'
+    rows = [row for row in metadata(mode) if row['call'] == call and row['layer'] == 0
+            and row['stage'] == stage and row['occurrence'] == 0]
+    require(len(rows) == 1, 'unique complete original alpha field')
+    row = rows[0]
+    width, inner = (64 if mode == 'bulk64' else 1), (16 if output else 2048)
+    shape, strides = [inner, width, 1, 1], [4, inner*4, inner*width*4, inner*width*4]
+    require(row['mode'] == mode and type(row['call']) is int and row['call'] == call
+            and row['call_tokens'] == width and row['window_start'] == 0 and row['window_tokens'] == 64
+            and row['token_axis'] == 1 and row['scope'] == 'actual_tensor'
+            and row['masked_negative_infinity_allowed'] is False
+            and row['source_slice_start_elements'] == 0, 'actual unchanged full64 versus AR1 call')
+    require(row['dense_ne'] == row['native_ne'] == shape and row['native_nb'] == strides
+            and row['native_op'] == ('MUL_MAT' if output else 'MUL')
+            and row['elements'] == inner * width and '/' not in row['file'], 'full original F32 alpha geometry')
+    raw = bounded(CAPTURE / mode / row['file'], 524288)
+    require(len(raw) == inner * width * 4 and all(word & 0x7f800000 != 0x7f800000
+            for (word,) in struct.iter_unpack('<I', raw)), 'complete finite actual alpha words')
+    return row, raw
+
+
+def actual_weight():
+    rows = [json.loads(line, object_pairs_hook=unique) for line in
+            bounded(CAPTURE / 'learned-operands/records.jsonl', 128*1024).splitlines()]
+    selected = [row for row in rows if row['name'] == 'blk.0.ssm_alpha.weight']
+    require(len(rows) == 28 and len(selected) == 1, 'original learned role closure')
+    row = selected[0]
+    require(row['type'] == 0 and row['type_name'] == 'f32' and row['ne'] == [2048,16,1,1]
+            and row['nb'] == [4,8192,131072,131072] and row['bytes'] == W_BYTES
+            and row['encoding'] == 'GGUF_RAW' and row['buffer_name'] == 'CPU_Mapped'
+            and row['chunks'] == [{'file':'learned-4-0.bin','offset':0,'bytes':W_BYTES}],
+            'actual ordinary unmodified full-width F32 alpha weight; no repack')
+    raw = bounded(CAPTURE / 'learned-operands/learned-4-0.bin', W_BYTES)
+    require(len(raw) == W_BYTES, 'all actual learned weight bytes')
+    with os.fdopen(os.open(MODEL, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
+        source.seek(W_OFFSET)
+        require(source.read(W_BYTES) == raw, 'all physical weight bits equal approved GGUF source interval')
+    require(all(word & 0x7f800000 != 0x7f800000 for (word,) in struct.iter_unpack('<I', raw)), 'finite original F32 weight')
+    return row, raw
+
+
+def capture_guards():
+    done = jread(CAPTURE / 'complete.json')
+    require(done['status'] == 'TRANSPARENT_WINDOW_CAPTURE_NOT_CAUSE_QUALIFICATION'
+            and all(done[key] is True for key in ['common_prior_exact','bulk_instrumentation_complete_state_exact',
+                'ar1_instrumentation_complete_state_exact','both_next_AR1_full248320_and_whole_state_exact',
+                'all28_owners_seen_bulk_and_all64_AR1_calls','learned_unchanged_after_compute'])
+            and done['window_start'] == 0 and done['window_tokens'] == 64 and done['threads'] == 8
+            and done['learned_verified_boundaries'] == 2, 'full actual own-context transparency/readset guards')
+    bulk_row, bulk = actual('bulk64', False, 0)
+    producer = [row for row in metadata('bulk64') if row['stage'] == 'learned_op_output::blk.0.attn_norm.weight'
+                and row['layer'] == 0 and row['occurrence'] == 0]
+    output, _ = actual('bulk64', True, 0)
+    require(len(producer) == 1 and producer[0]['native_tensor_id'] == bulk_row['native_tensor_id']
+            == output['native_source_ids'][1] and producer[0]['native_op'] == 'MUL'
+            and bounded(CAPTURE/'bulk64'/producer[0]['file'],524288) == bulk, 'actual exact input producer binding')
+    for at in range(64):
+        row, raw = actual('ar1', False, at)
+        out, _ = actual('ar1', True, at)
+        require(raw == bulk[at*X_BYTES:(at+1)*X_BYTES] and row['native_tensor_id'] == out['native_source_ids'][1],
+                'all131072 actual input words and AR1 projection producer bindings close')
+
+
+def verify_fixtures(folder, pins):
+    manifest = jread(folder / 'manifest.json')
+    require(manifest['status'] == 'AUTHENTIC_BYTE_ONLY_ALPHA_FIXTURES' and len(manifest['cases']) == 65
+            and manifest['source_pins'] == digest(HERE/'source-pins.json')
+            and manifest['weight']['captured_row'] == actual_weight()[0], 'source-bound full alpha fixture manifest')
+    _, weight = actual_weight()
+    require(bounded(folder/'weight.f32le', W_BYTES) == weight
+            and digest(folder/'weight.f32le') == manifest['weight']['digest'], 'exact shared original weight fixture')
+    total = len(weight)
+    for which, descriptor in enumerate(manifest['cases']):
+        mode, at = ('bulk64', 0) if which == 0 else ('ar1', which-1)
+        key = 'bulk64' if which == 0 else f'ar1-{at}'
+        require(descriptor['case'] == key and descriptor['mode'] == mode and descriptor['call'] == at
+                and set(descriptor['fields']) == {'x','expected'}, 'same original65call population/order')
+        for suffix, output in [('x',False),('expected',True)]:
+            row, raw = actual(mode,output,at)
+            part = descriptor['fields'][suffix]
+            require(part['file'] == f'{key}.{suffix}.f32le' and part['captured_row'] == row
+                    and digest(folder/part['file']) == part['digest']
+                    and bounded(folder/part['file'],524288) == raw, 'authentic full actual fixture bytes')
+            total += len(raw)
+    require(total == manifest['payload_bytes'] == 1187840 and total <= FIXTURE_CAP, 'complete bounded fixture payload')
+    require(pins['cpu_dso'] == CPU, 'same public original backend fixture scope')
+    return digest(folder/'manifest.json')
+
+
+def prepare(output):
+    require(output.resolve().is_relative_to(HERE), 'owned exclusive fixture directory')
+    pins = authenticate()
+    capture_guards()
+    output.mkdir(mode=0o700)
+    row, weight = actual_weight()
+    with (output/'weight.f32le').open('xb') as stream:
+        stream.write(weight)
+    (output/'weight.f32le').chmod(0o600)
+    entries, total = [], len(weight)
+    for which in range(65):
+        mode, at = ('bulk64',0) if which == 0 else ('ar1',which-1)
+        key = 'bulk64' if which == 0 else f'ar1-{at}'
+        fields = {}
+        for suffix, output_field in [('x',False),('expected',True)]:
+            captured, raw = actual(mode,output_field,at)
+            target = output/f'{key}.{suffix}.f32le'
+            total += len(raw)
+            require(total <= FIXTURE_CAP, 'bounded raw fixture payload')
+            with target.open('xb') as stream:
+                stream.write(raw)
+            target.chmod(0o600)
+            fields[suffix] = {'file':target.name,'captured_row':captured,'digest':digest(target)}
+        entries.append({'case':key,'mode':mode,'call':at,'fields':fields})
+    manifest = {'status':'AUTHENTIC_BYTE_ONLY_ALPHA_FIXTURES','source_pins':digest(HERE/'source-pins.json'),
+                'weight':{'file':'weight.f32le','captured_row':row,'digest':digest(output/'weight.f32le')},
+                'cases':entries,'payload_bytes':total,'arithmetic_executed':False,
+                'expected_values':'unchanged complete actual original public alpha output bits',
+                'source_range':{'model':str(MODEL),'start':W_OFFSET,'bytes':W_BYTES,'encoding':'F32_GGUF_RAW'}}
+    raw = (json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
+    require(len(raw) <= 1024*1024, 'bounded fixture metadata')
+    with (output/'manifest.json').open('xb') as stream:
+        stream.write(raw)
+    authenticate()
+    verify_fixtures(output,pins)
+    print('AUTHENTIC_BYTE_ONLY_ALPHA_FIXTURES',len(entries),total)
+
+
+if __name__ == '__main__':
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('new_fixture_directory',type=Path)
+    prepare(parser.parse_args().new_fixture_directory)
--- /dev/null
+++ original2B-alpha-public-replay/run_replay.py
@@ -0,0 +1,196 @@
+#!/usr/bin/env python3
+"""ROOT ONLY: finite owned public-operator process, no model/provider server."""
+import argparse
+import json
+import os
+from pathlib import Path
+import selectors
+import signal
+import subprocess
+import time
+from prepare import HERE, CPU, authenticate, verify_fixtures, digest, jread, require, capture_guards
+
+LOADER = '/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib/ld-linux-x86-64.so.2'
+LIB = '/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib'
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
+
+
+def main():
+    p = argparse.ArgumentParser(description=__doc__)
+    p.add_argument('--binary', type=Path, required=True)
+    p.add_argument('--binary-sha256', required=True)
+    p.add_argument('--fixtures', type=Path, required=True)
+    p.add_argument('--new-output', type=Path, required=True)
+    p.add_argument('--new-log', type=Path, required=True)
+    p.add_argument('--new-receipt', type=Path, required=True)
+    p.add_argument('--preload', type=Path)
+    p.add_argument('--preload-sha256')
+    a = p.parse_args()
+    pins = authenticate()
+    capture_guards()
+    for path in [a.new_output, a.new_log, a.new_receipt]:
+        require(path.resolve().is_relative_to(HERE), 'owned ignored-run output path')
+    require(a.fixtures.resolve() == HERE / 'root-fixtures', 'exact accepted original fixtures')
+    before_fixtures = verify_fixtures(a.fixtures, pins)
+    before_binary = digest(a.binary)
+    require((a.preload is None) == (a.preload_sha256 is None), 'preload path and SHA paired')
+    before_preload = None
+    if a.preload is not None:
+        require(a.preload.resolve().is_relative_to(HERE), 'owned observer DSO only')
+        before_preload = digest(a.preload)
+        require(before_preload['bytes'] <= 8 * 1024 * 1024
+                and before_preload['sha256'] == a.preload_sha256, 'bounded pinned observer DSO')
+    require(before_binary['bytes'] <= 16*1024*1024 and before_binary['sha256'] == a.binary_sha256
+            and not a.new_output.exists(), 'bounded pinned newly built binary/new output')
+    require(a.new_log.parent.resolve() != a.new_output.resolve()
+            and a.new_receipt.parent.resolve() != a.new_output.resolve(), 'runner evidence outside native output')
+    argv = [LOADER, '--library-path', LIB, str(a.binary.resolve()), str(a.fixtures.resolve()),
+            str(a.new_output.resolve()), CPU, '8', '300']
+    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'CUDA_VISIBLE_DEVICES': '-1',
+           'NVIDIA_VISIBLE_DEVICES': 'void', 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
+    if a.preload is not None:
+        env['LD_PRELOAD'] = str(a.preload.resolve())
+        env['ZG_ALPHA_TRACE'] = '1'
+    started = time.monotonic()
+    receipt = {'argv': argv, 'binary': before_binary, 'fixture_manifest': before_fixtures,
+               'wall_cap_seconds': 300, 'address_space_cap_bytes': 8 * 1024**3,
+               'actual_gpu_or_model_execution': False, 'internal_dispatch_observed': False,
+               'preload': str(a.preload.resolve()) if a.preload else None, 'preload_digest': before_preload,
+               'source_profile': 'pinned llama.cpp0.6.0 haswell public alpha default/control diagnostic',
+               'exit_code': None, 'status': 'UNQUALIFIED'}
+    with a.new_receipt.open('x') as receipt_file, a.new_log.open('xb') as log:
+        process = None
+        selector = None
+        log_bytes = 0
+        primary_error = None
+        cleanup_error = None
+        try:
+            process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                                       stderr=subprocess.STDOUT, start_new_session=True)
+            receipt['owned_pid'] = process.pid
+            selector = selectors.DefaultSelector()
+            selector.register(process.stdout, selectors.EVENT_READ)
+            while selector.get_map():
+                require(time.monotonic() - started < 300, 'finite wall deadline')
+                for key, _ in selector.select(0.2):
+                    raw = os.read(key.fileobj.fileno(), 65536)
+                    if not raw:
+                        selector.unregister(key.fileobj)
+                        continue
+                    log_bytes += len(raw)
+                    require(log_bytes <= 8 * 1024 * 1024, 'bounded child log')
+                    log.write(raw)
+            receipt['exit_code'] = leader_exit_without_reaping(process, 5)
+            require(receipt['exit_code'] == 0, 'public alpha process refused')
+            require(digest(a.binary) == before_binary, 'binary changed during run')
+            require(before_preload is None or digest(a.preload) == before_preload, 'observer changed during run')
+            authenticate()
+            require(verify_fixtures(a.fixtures, pins) == before_fixtures, 'fixtures changed during run')
+            capture_guards()
+            receipt['status'] = 'PUBLIC_ALPHA_OBSERVER_TRANSPARENCY_VALIDATOR_REQUIRED'
+        except BaseException as error:
+            receipt['error'] = f'{type(error).__name__}: {error}'
+            primary_error = error
+        finally:
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
+                    try:
+                        close()
+                        receipt.setdefault('cleanup', {})[label] = True
+                    except BaseException as error:
+                        cleanup_error = cleanup_error or error
+                        receipt.setdefault('cleanup_errors', []).append(f'{label}: {type(error).__name__}: {error}')
+                        receipt['status'] = 'UNQUALIFIED'
+                if process is not None:
+                    receipt['exit_code'] = process.returncode
+            receipt['elapsed_seconds'] = time.monotonic() - started
+            json.dump(receipt, receipt_file, indent=2, allow_nan=False)
+            receipt_file.write('\n')
+            receipt_file.flush()
+        if primary_error is not None:
+            raise primary_error
+        if cleanup_error is not None:
+            raise cleanup_error
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ original2B-alpha-public-replay/validate.py
@@ -0,0 +1,164 @@
+#!/usr/bin/env python3
+"""ROOT ONLY: all default/output transparency before exported alpha route evidence."""
+import argparse
+import hashlib
+import json
+from pathlib import Path
+import struct
+from prepare import HERE, CPU, authenticate, actual, verify_fixtures, bounded, digest, jread, require, capture_guards
+
+
+def integer(value):
+    return type(value) is int
+
+
+def words_diff(left, right):
+    require(len(left) == len(right) and len(left) % 4 == 0, 'complete raw word comparison')
+    return sum(a != b for a, b in zip(struct.iter_unpack('<I', left), struct.iter_unpack('<I', right)))
+
+
+def run_receipt(path, output, observed):
+    record = jread(path)
+    require(record['status'] == 'PUBLIC_ALPHA_OBSERVER_TRANSPARENCY_VALIDATOR_REQUIRED'
+            and integer(record['exit_code']) and record['exit_code'] == 0
+            and record['actual_gpu_or_model_execution'] is False and record['internal_dispatch_observed'] is False,
+            'complete unchanged public alpha replay')
+    require(record['argv'][5] == str(output.resolve()) and record['argv'][-3:] == [CPU,'8','300'], 'exact owned original invocation')
+    require(all(record['cleanup'].get(key) is True for key in
+                ['leader_reaped','group_absent','pipe_eof','selector_closed','pipe_closed']), 'complete finite owned PGID/pipe cleanup')
+    require(record['wall_cap_seconds'] == 300 and record['address_space_cap_bytes'] == 8*1024**3,
+            'same bounded resource policy')
+    require(digest(record['argv'][3]) == record['binary'], 'same pinned public replay executable')
+    if observed:
+        require(type(record['preload']) is str and Path(record['preload']).resolve().is_relative_to(HERE)
+                and digest(record['preload']) == record['preload_digest'], 'actual own pinned observer')
+    else:
+        require(record['preload'] is None and record['preload_digest'] is None, 'genuine unobserved baseline')
+    return record
+
+
+def outputs(folder, observer):
+    require(folder.resolve().is_relative_to(HERE), 'owned new output directory')
+    done = jread(folder/'complete.json')
+    require(done['status'] == 'PUBLIC_ALPHA_REPLAY_DEFAULT_EXACT_CONTROL_DIAGNOSTIC'
+            and done['default_cases'] == done['control_cases'] == 65 and done['default_words'] == 2048
+            and integer(done['threads']) and done['threads'] == 8 and done['cause_qualified'] is False
+            and done['production_math_changed'] is False and done['internal_dispatch_observed'] is False,
+            'complete public65 default/65 control alpha output gate')
+    cases = [json.loads(line) for line in bounded(folder/'cases.jsonl',128*1024).splitlines()]
+    require(len(cases) == 130, 'complete source case order/count')
+    content, descriptors, defaults, total = {}, {}, {}, 0
+    for index,row in enumerate(cases):
+        which = index % 65
+        key = 'bulk64' if which == 0 else f'ar1-{which-1}'
+        control = index >= 65
+        file = ('ref-control-' if control else 'default-') + key + '.actual.f32le'
+        count = 1024 if which == 0 else 16
+        require(row['case'] == key and row['mode'] == ('PUBLIC_USE_REF_CONTROL' if control else 'DEFAULT_PUBLIC_ALPHA')
+                and integer(row['words']) and row['words'] == count, 'all complete raw alpha words/cases')
+        raw = bounded(folder/file,4096)
+        require(len(raw) == count*4 and all(word & 0x7f800000 != 0x7f800000
+                for (word,) in struct.iter_unpack('<I',raw)), 'exact full finite original operator output')
+        _, expected = actual('bulk64' if which == 0 else 'ar1',True,0 if which == 0 else which-1)
+        different = words_diff(raw,expected)
+        require(type(row['actual_capture_unequal_words']) is int and row['actual_capture_unequal_words'] == different,
+                'independently checked full capture comparison')
+        if not control:
+            require(different == 0, 'all65 default public projections exact to original capture')
+            defaults[which] = raw
+        content[file] = raw
+        descriptors[file] = {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
+        total += len(raw)
+    require(total == 16384 and len(defaults) == 65, 'all4096 alpha output words per run')
+    ref_bulk = content['ref-control-bulk64.actual.f32le']
+    ref_vs_ar = sum(words_diff(ref_bulk[at*64:(at+1)*64],defaults[at+1]) for at in range(64))
+    ref_ar_diff = sum(words_diff(content[f'ref-control-ar1-{at}.actual.f32le'],defaults[at+1]) for at in range(64))
+    require(cases[65]['ref_bulk_vs_all_default_ar1_unequal_words'] == ref_vs_ar,
+            'independently checked complete refbulk versus all originalAR1 comparison')
+    for phase in ['before','after']:
+        maps = bounded(folder/f'{phase}.maps',1024*1024).decode()
+        require(CPU in maps and all(CPU in line for line in maps.splitlines() if 'libggml-cpu-' in line),
+                'actual pinned Haswell backend only')
+        if observer:
+            require(observer in maps, 'actual observer mapped around computation')
+    return descriptors, content, {'ref_bulk_vs_all_default_AR1_unequal_words':ref_vs_ar,
+        'all_ref_AR1_vs_default_AR1_unequal_words':ref_ar_diff,'all65_defaults_exact':True}
+
+
+def routes(path):
+    trace = jread(path)
+    require(trace['cpu'] == CPU and trace['symbol_offset'] == 0x10c8f0
+            and trace['internal_microkernel_observed'] is False and trace['math_replaced'] is False,
+            'exact pinned exported entry, opaque params/no internal-kernel inference')
+    require(integer(trace['case_count']) and trace['case_count'] == 130 and len(trace['cases']) == 130
+            and integer(trace['call_count']) and trace['call_count'] == 520 and len(trace['calls']) == 520,
+            'complete bounded original public entry trace')
+    at = 0
+    for index,case in enumerate(trace['cases']):
+        phase,which = divmod(index,65)
+        width,count = (64 if which == 0 else 1),(0 if phase else 8)
+        require(all(integer(case[key]) for key in ['phase','which','width','first','last'])
+                and case == {'phase':phase,'which':which,'width':width,'first':at,'last':at+count},
+                'exact original public alpha route population/order')
+        for row in trace['calls'][at:at+count]:
+            require(all(integer(row[key]) for key in ['phase','which','m','n','k','lda','ldb','ldc',
+                    'a_offset','b_offset','c_offset','atype','btype','ctype']), 'exact integer entry descriptor')
+            require(row['phase'] == phase and row['which'] == which
+                    and (row['m'],row['n'],row['k']) == (16,width,2048)
+                    and (row['lda'],row['ldb'],row['ldc']) == (2048,2048,16)
+                    and (row['atype'],row['btype'],row['ctype']) == (0,0,0)
+                    and (row['a_offset'],row['b_offset'],row['c_offset']) == (0,0,0)
+                    and type(row['accepted']) is bool and row['accepted'] is (width == 64),
+                    'full original F32 alpha pointers/geometry and untouched public return')
+        at += count
+    require(at == 520, 'no omitted/trailing public call')
+    return {'default_bulk64':'8 exported calls, all true',
+            'default_AR1':'64x8 exported calls, all false',
+            'public_use_ref':'65 cases, zero exported calls','exported_calls':at}
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--plain-output',type=Path,required=True)
+    parser.add_argument('--plain-receipt',type=Path,required=True)
+    parser.add_argument('--observed-output',type=Path,required=True)
+    parser.add_argument('--observed-receipt',type=Path,required=True)
+    parser.add_argument('--new-report',type=Path,required=True)
+    args = parser.parse_args()
+    pins = authenticate()
+    capture_guards()
+    fixtures = verify_fixtures(HERE/'root-fixtures',pins)
+    plain = run_receipt(args.plain_receipt,args.plain_output,False)
+    observed = run_receipt(args.observed_receipt,args.observed_output,True)
+    require(plain['binary'] == observed['binary'] and plain['fixture_manifest'] == observed['fixture_manifest'] == fixtures
+            and plain['argv'] == observed['argv'][:5]+[str(args.plain_output.resolve())]+observed['argv'][6:],
+            'same exact original operator inputs/binary/conditions')
+    require(not (args.plain_output/'sgemm-trace.json').exists(), 'plain unobserved execution')
+    a,raw_a,summary = outputs(args.plain_output,None)
+    b,raw_b,observed_summary = outputs(args.observed_output,observed['preload'])
+    require(a == b and raw_a == raw_b and summary == observed_summary,
+            'all130 full plain/observed default AND control outputs bit-identical before route acceptance')
+    route = routes(args.observed_output/'sgemm-trace.json')
+    total = 0
+    for path in HERE.rglob('*'):
+        require(not path.is_symlink() and (path.is_file() or path.is_dir()),'regular own diagnostic artifact tree')
+        if path.is_file():
+            total += path.stat().st_size
+    require(total <= 512*1024*1024-128*1024,'bounded complete new-run artifacts with report reserve')
+    authenticate()
+    report = {'status':'TRANSPARENT_PUBLIC_ALPHA_EXPORTED_ENTRY_ROUTE_OBSERVED',
+        'all65_defaults_exact_to_actual_captures':True,'all130_plain_observed_outputs_exact':True,
+        'words_each_run':4096,'routes':route,'controls':summary,'outputs':a,
+        'input_closure':'same actual131072 F32 input words and full32768-word raw originalF32 alpha weights',
+        'source_pins':digest(HERE/'source-pins.json'),'binary':plain['binary'],'observer':observed['preload_digest'],
+        'pre_report_artifact_bytes':total,'internal_microkernel_observed':False,'full_runtime_closure_qualified':False,
+        'long_context_cause_qualified':False,'production_math_changed':False,'model_or_profile_eligibility':False}
+    raw = (json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
+    require(len(raw) <= 128*1024 and args.new_report.resolve().is_relative_to(HERE),'bounded new owned report')
+    with args.new_report.open('xb') as stream:
+        stream.write(raw)
+    print(report['status'],str(args.new_report))
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ original2B-alpha-public-replay/source-pins.json
@@ -0,0 +1,695 @@
+{
+  "scope": "SOURCE_ONLY_PUBLIC_ORIGINAL2B_ALPHA_REPLAY_OBSERVER_PREPARATION",
+  "capture": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport",
+  "original_source": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0",
+  "cpu_dso": "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so",
+  "public_symbol": "llamafile_sgemm",
+  "public_symbol_offset": 1100016,
+  "static_elf_evidence": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/source-pins.json",
+  "model_hash_provenance": "Previously independently approved exact GGUF; author only read bounded131072-byte alpha source interval; ROOT authenticate hashes whole model before/after",
+  "inputs": {
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/complete.json": {
+      "bytes": 1181,
+      "sha256": "0b73666812a6dd9a99558562c0460ab1810e05399f645fed96aa70d37725e1f3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/bulk64/records.jsonl": {
+      "bytes": 55822,
+      "sha256": "eacee1e7cf45fdac3ea40e48152432347f1f8f3915c633ecc2ce4c8f59e54437"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/records.jsonl": {
+      "bytes": 3378646,
+      "sha256": "114bbbf9464e288b225ece7e63f73ba07b9ffeae42343e45cc0b5d1b4fac2ed3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/learned-operands/records.jsonl": {
+      "bytes": 17570,
+      "sha256": "26d8a9d96194a38d6125ab339abc7564229cd98e65dc6ef7bee1751fa7bb7590"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/learned-operands/learned-4-0.bin": {
+      "bytes": 131072,
+      "sha256": "0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport-comparison.json": {
+      "bytes": 4809755,
+      "sha256": "0fd6266cb8437d7e1f1113944428a58fd31951bd44d2409d80ea9c3f0a638bae"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/learned-source-headers.json": {
+      "bytes": 6898,
+      "sha256": "5a0e2c8ee280d08c06252b25ddcc38fab300ccbaf56186104b3564ca864b9615"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-repack-transport-followup/native_window.cpp": {
+      "bytes": 33599,
+      "sha256": "43e3da3ceb6bd482e4e8419bbdc257687b7021720c0afcb35f8701ef784db614"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-repack-transport-followup/compare_window.py": {
+      "bytes": 24185,
+      "sha256": "1bc9fe24c2500017b339678884ad4a4113fc6271f5a315391b2f847868ba9b58"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/main.cpp": {
+      "bytes": 15528,
+      "sha256": "f14c8831a178bb89ee6e854fd84539d239d8605526acc14584b08b4e24434b94"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/observer.cpp": {
+      "bytes": 6793,
+      "sha256": "04ceca23c9c816fda083c27a748599b0c489baa31129559dba5689a8ffaec786"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/observer.h": {
+      "bytes": 373,
+      "sha256": "249cadd15ed550f6113edf48f457276850598871a84c5c01d518c6902e57a880"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/run_replay.py": {
+      "bytes": 9225,
+      "sha256": "c226112446b7d90f60363f9a08291680e467be49e171f146e663d31891a44b7c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/root-validation.json": {
+      "bytes": 1043,
+      "sha256": "aba581431ecbb32bc9d0d8f7cc85d1edded11da218fc81933e8ee79dc1def3d9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/source-pins.json": {
+      "bytes": 3993,
+      "sha256": "51ffc0976cf38182e3fe0f355c2f820723566e6e1526e1726c2303408e08e82a"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/bulk64/field-15.f32le": {
+      "bytes": 524288,
+      "sha256": "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/bulk64/field-16.f32le": {
+      "bytes": 4096,
+      "sha256": "448f5626f2f06de0e5fe0f375a92589bd87b81833fcbb9daca530c20f8424b3e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-15.f32le": {
+      "bytes": 8192,
+      "sha256": "b1d1ac012f78247da9d59c268de262cb88d569fcc5f28df8e580ab3a1e837208"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-16.f32le": {
+      "bytes": 64,
+      "sha256": "c5966078b46b065ea9b7505b36e7fdfd4f769350f47ac4b52a8e9c3ddcc5f90c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-126.f32le": {
+      "bytes": 8192,
+      "sha256": "d8c1e375e97820914cd5fc4709879f042e7406a0a303f1fa2d6e2e13669d50a2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-127.f32le": {
+      "bytes": 64,
+      "sha256": "ad7c50ccc5731d098bffbe7b8c49b4740226037f074ce36a127ab401057f2b13"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-235.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-236.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-344.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-345.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-453.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-454.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-562.f32le": {
+      "bytes": 8192,
+      "sha256": "d6394c00fd7f2c1566e93699e79c0aa400568310aea2abaec4e31a6f478e7a8f"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-563.f32le": {
+      "bytes": 64,
+      "sha256": "532e7275e293b0dbc749eecd9808ce05f6ea5e1e6bf114adfe31bf2f6efcfc8f"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-671.f32le": {
+      "bytes": 8192,
+      "sha256": "4debe33280e810932c53160b23d82a923a004f84f59b543c5517eb9e3c310461"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-672.f32le": {
+      "bytes": 64,
+      "sha256": "e9d014c55d2dbb556818e78fd486606b75550c0bb12b018c2270f1186bd66797"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-780.f32le": {
+      "bytes": 8192,
+      "sha256": "e5f627101f0b7ea40d8d7d262a28916add245737fb551403e0e7240c214f0e28"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-781.f32le": {
+      "bytes": 64,
+      "sha256": "9fdf1df6a995d85b81c6a8182bc9a5d5af401603e7113f078b15d8cbdafb6a91"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-889.f32le": {
+      "bytes": 8192,
+      "sha256": "dd96115805b5424ba80832d1d0f98a53a1fa8762cdcfc1152b1d965d900b5e2e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-890.f32le": {
+      "bytes": 64,
+      "sha256": "52a936318dbe26ab7df850f47716b16b8da5f7423700f81929aca0c0ad82d6ce"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-998.f32le": {
+      "bytes": 8192,
+      "sha256": "04b06b0330f2d25de147a54c0685a030ce18d35f5d2ebba0262f1f2022bbf000"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-999.f32le": {
+      "bytes": 64,
+      "sha256": "f579ac32b7007efa37a7aa5da70f20896f865a7807afbd343216c91296fc70d1"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1107.f32le": {
+      "bytes": 8192,
+      "sha256": "179941df556fb79370c6190ee4d528c530935f9c4397a1cfca82c15d2de49ca5"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1108.f32le": {
+      "bytes": 64,
+      "sha256": "bb4e5a53a864a9aca9c369d7e84982f03d6cb4aacd3e83a51b7f39d6ce4d1a2c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1216.f32le": {
+      "bytes": 8192,
+      "sha256": "3d1d394ac0dd107357192ac9d000202fe01acfce20e6380518cf84fc103f8394"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1217.f32le": {
+      "bytes": 64,
+      "sha256": "c588cd2252500b74b9e93d981f4f5b4d928d29c8786db6e590882181febde8bc"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1325.f32le": {
+      "bytes": 8192,
+      "sha256": "86f6f7a22e288e49ad29ffc2e7635a169d281beb4333768fe88bca49c6663a6f"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1326.f32le": {
+      "bytes": 64,
+      "sha256": "18c0fbe8a620e8ac25b7015191a6135198ae128a9e654154c0be50a6d6702c1b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1434.f32le": {
+      "bytes": 8192,
+      "sha256": "c0adc6be7d6c5a74a0c398ea033b45bff455eaa4023e79f2700cf4b80f7c8226"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1435.f32le": {
+      "bytes": 64,
+      "sha256": "06c4fa2f9601385da9c2067c40473c705456b2d09abf4a686fd233d98169f1b4"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1543.f32le": {
+      "bytes": 8192,
+      "sha256": "5beb5350ecfda3a3ee1f4373ae4963fa1fe8446640da40417864b6e7cd45182b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1544.f32le": {
+      "bytes": 64,
+      "sha256": "f335de41562407dae49acfef40b894b8f589e46cb7b97496c79b3fe27ae92bd4"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1652.f32le": {
+      "bytes": 8192,
+      "sha256": "790c85507bc57a6a4953ace8d0ee163dd3c247b6b42575e6177ebf5551b76792"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1653.f32le": {
+      "bytes": 64,
+      "sha256": "263e18418766b9f3eb33b46bbf335fdb5bb161f839af61c38c7cb660f13b943b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1761.f32le": {
+      "bytes": 8192,
+      "sha256": "952eac836d2bbfc0889d5c0a4b7ed7f1d6fe82b00bd2bd1c7f32c3f6f577331e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1762.f32le": {
+      "bytes": 64,
+      "sha256": "44969c46575a1f474c8b427c6f5e17f24a0a854736a5c179bc66e76c68f405b7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1870.f32le": {
+      "bytes": 8192,
+      "sha256": "db1bc46461d226e632d5f90a896063fc962d5677b9ae841fabcc72a03574afac"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1871.f32le": {
+      "bytes": 64,
+      "sha256": "0c0eefc34cb33fc8ee34da8517e17796a2c98f91dbede5d1d06a0e940e4ed982"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1979.f32le": {
+      "bytes": 8192,
+      "sha256": "c81baccc0b256e13db4edd567f72095378e85b187924839fd4b5bce68fe870cd"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1980.f32le": {
+      "bytes": 64,
+      "sha256": "96a67aa232ec9c1843b81bb0fdad48d60b5020f76040a7b1cbeb1aa02720ec4e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2088.f32le": {
+      "bytes": 8192,
+      "sha256": "93b61dc38c2d6cefc5d4a9c95651b2f98f45561512ee427bc5bba6952c87eacd"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2089.f32le": {
+      "bytes": 64,
+      "sha256": "3b12967b82c0c96808c1bf16ff2ebd82175dd9b0b74d5baaf4fccd7591d6eb31"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2197.f32le": {
+      "bytes": 8192,
+      "sha256": "06b6a0540a93872575ebe326330b19a0d881d0a94025fab9037f244843b8d30c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2198.f32le": {
+      "bytes": 64,
+      "sha256": "3709eadf0cc78f3a38f6f68ae2048afc96f21b0ad576d22a2902e55690d13c61"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2306.f32le": {
+      "bytes": 8192,
+      "sha256": "88a090dac7c622a215631a1920ad654760fa46e23f6149d0e89b675167fa7963"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2307.f32le": {
+      "bytes": 64,
+      "sha256": "76dc55427c84b5798eea8d9a4b8e892f4f0a3029a884277d461db0d993c0071a"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2415.f32le": {
+      "bytes": 8192,
+      "sha256": "655cc2c29feb1a794bc7aaa2ac93103104d717911e99f45eba93741de3e9fe02"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2416.f32le": {
+      "bytes": 64,
+      "sha256": "6541bf946984c14ae2fcec20e9a74c495dddf0146789b490889aca921f3ef84c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2524.f32le": {
+      "bytes": 8192,
+      "sha256": "88a090dac7c622a215631a1920ad654760fa46e23f6149d0e89b675167fa7963"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2525.f32le": {
+      "bytes": 64,
+      "sha256": "76dc55427c84b5798eea8d9a4b8e892f4f0a3029a884277d461db0d993c0071a"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2633.f32le": {
+      "bytes": 8192,
+      "sha256": "df57eba0c74a8f626e55ff0174bfdd5796c230a7ada1dc043ab89aeec30d962d"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2634.f32le": {
+      "bytes": 64,
+      "sha256": "401f04f19c9a802e2bb31e64d240550da1818d82159d2d7d7a69b0dd17217b78"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2742.f32le": {
+      "bytes": 8192,
+      "sha256": "5a748c996f44af066bfb0201952b91c81cc2a82a549f87b728e4461d6923e3fe"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2743.f32le": {
+      "bytes": 64,
+      "sha256": "4660e1b573834c90873514f00db01c119caa2bb7e8f6f60e4d4833820d1cad19"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2851.f32le": {
+      "bytes": 8192,
+      "sha256": "5bb0f960655d306707c9396463d78a209f4cf4eb18282cfa1782117eb44785c7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2852.f32le": {
+      "bytes": 64,
+      "sha256": "bac1a44db7be34bca0f66c7bdb29a57382563fa345d497e3300eaa188efefb6e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2960.f32le": {
+      "bytes": 8192,
+      "sha256": "07fc9735c3c73f304becc2501cacb24baf981795b3dd7732970384e2a4cc301e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2961.f32le": {
+      "bytes": 64,
+      "sha256": "3108d244ef07bb9d3b6c2587a1c093bccb349635dfb94f7628b7734d786a5708"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3069.f32le": {
+      "bytes": 8192,
+      "sha256": "d8c1e375e97820914cd5fc4709879f042e7406a0a303f1fa2d6e2e13669d50a2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3070.f32le": {
+      "bytes": 64,
+      "sha256": "ad7c50ccc5731d098bffbe7b8c49b4740226037f074ce36a127ab401057f2b13"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3178.f32le": {
+      "bytes": 8192,
+      "sha256": "a60820c0bdaab2d062f82213fd23f2db10ad03b8748634fec024d6a3a4a5a5d5"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3179.f32le": {
+      "bytes": 64,
+      "sha256": "f05a4871f61a0826a7e9dd3cf472d23b165fd18fc42c380c9d43a289d178fe04"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3287.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3288.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3396.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3397.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3505.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3506.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3614.f32le": {
+      "bytes": 8192,
+      "sha256": "952eac836d2bbfc0889d5c0a4b7ed7f1d6fe82b00bd2bd1c7f32c3f6f577331e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3615.f32le": {
+      "bytes": 64,
+      "sha256": "44969c46575a1f474c8b427c6f5e17f24a0a854736a5c179bc66e76c68f405b7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3723.f32le": {
+      "bytes": 8192,
+      "sha256": "4debe33280e810932c53160b23d82a923a004f84f59b543c5517eb9e3c310461"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3724.f32le": {
+      "bytes": 64,
+      "sha256": "e9d014c55d2dbb556818e78fd486606b75550c0bb12b018c2270f1186bd66797"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3832.f32le": {
+      "bytes": 8192,
+      "sha256": "30279f1aff15ed5348dd29912b375e0bcc1763066a99e863233c08b959e1af13"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3833.f32le": {
+      "bytes": 64,
+      "sha256": "4bb0900b75f3a143e4b491045ceb67d3507bd2bb0bab69822689b94b910717b1"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3941.f32le": {
+      "bytes": 8192,
+      "sha256": "6c93b707dbf7ac24a57a67fbb1976703083263d05cd4ff5ec3f8b0ecf2a3177b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3942.f32le": {
+      "bytes": 64,
+      "sha256": "f747628720fb5b6da6303c210988d76ecbc08e7d17e2e7397e8a39e0a8c63b73"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4050.f32le": {
+      "bytes": 8192,
+      "sha256": "4a7fc5a5f871ef5e796e0a197c33b6395dcc9f92543b56c2e4a4e581bd1b7a53"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4051.f32le": {
+      "bytes": 64,
+      "sha256": "0e2d3dcd80ebd53400cd5a1caf93371d8e551cb233b8e48a77d65f410b150e95"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4159.f32le": {
+      "bytes": 8192,
+      "sha256": "7bed4534f054fc0dc094483be0435a0a536c6361b11a26b07850558aa22e639b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4160.f32le": {
+      "bytes": 64,
+      "sha256": "cee0240edadad1e603de7a7d1363f871869be124bb925fb40c1a984efa5e7514"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4268.f32le": {
+      "bytes": 8192,
+      "sha256": "fdf64cc096f707625d60f8f46b05e0e353d408f01043c03ee3d7df06fc933dab"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4269.f32le": {
+      "bytes": 64,
+      "sha256": "322a6b71a51c5cd4cf98e3115c8935b75ac2b3a0d6a24c41b48259dd0521be53"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4377.f32le": {
+      "bytes": 8192,
+      "sha256": "640e7d893c2be8e3f2787d1dc2e64b9fcec3aafb530a2dd3e06e3d7ac18c1b30"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4378.f32le": {
+      "bytes": 64,
+      "sha256": "4356d54d83e0825fa281d7666335f207ea3464b6e80141fa520a4b08ae3794f5"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4486.f32le": {
+      "bytes": 8192,
+      "sha256": "70c58c99a9fa9e6b1c72ed04276dcc6a063c1bbbb059f5268b3c4aaafb7abe03"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4487.f32le": {
+      "bytes": 64,
+      "sha256": "7adf12e0f53418ad2734087b688dbeecc59bc55757bd31d0b9967c5ad2dcac7e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4595.f32le": {
+      "bytes": 8192,
+      "sha256": "d45800a66a8471f2f51e3e3d9596242623da6e2a40152368d143eb6f5488dc05"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4596.f32le": {
+      "bytes": 64,
+      "sha256": "c7594b20ac3b3560e62d0d216fd91f8535172eb51b952c79c6c479dee0309dce"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4704.f32le": {
+      "bytes": 8192,
+      "sha256": "bc7c46a182f858a9bfd0748d6f6b0d3b17bcb2be46178575b3301edfac0442be"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4705.f32le": {
+      "bytes": 64,
+      "sha256": "c1b20ad3e91ffab35d442276d6110bf8c8d86aba8a80ece89620a7bb701bdca8"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4813.f32le": {
+      "bytes": 8192,
+      "sha256": "8044a7109847783b4c7030a18a7d290d932fc425853e96ea86302a90d8aaceb1"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4814.f32le": {
+      "bytes": 64,
+      "sha256": "aaee68636f666392f947a954cd9c22574fd9c305a352cf6bdf5002fa56956ccb"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4922.f32le": {
+      "bytes": 8192,
+      "sha256": "ed177a9737fbaf32463e6f5d2341b4d17c57d96b20a463b66ac667272f0919ad"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4923.f32le": {
+      "bytes": 64,
+      "sha256": "e39cf6c108dc99f31ec8a79ffa88a080b00e9ae3dfbe4c3a044d4a2af66982a2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5031.f32le": {
+      "bytes": 8192,
+      "sha256": "a772bbd8b3f2146ddf89913c465981b6a28fed96b3f42c5dd82bb55e4ae72e0b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5032.f32le": {
+      "bytes": 64,
+      "sha256": "ee9be70e64148e4b53d7579856a098d5a918dc5c2d7cfaa7e8e2d01cae6fb5c8"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5140.f32le": {
+      "bytes": 8192,
+      "sha256": "952eac836d2bbfc0889d5c0a4b7ed7f1d6fe82b00bd2bd1c7f32c3f6f577331e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5141.f32le": {
+      "bytes": 64,
+      "sha256": "44969c46575a1f474c8b427c6f5e17f24a0a854736a5c179bc66e76c68f405b7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5249.f32le": {
+      "bytes": 8192,
+      "sha256": "962cfcb9d4ada261808fc4019c96cd26e9acd8cf7a27fe68c4c14ff939ef473e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5250.f32le": {
+      "bytes": 64,
+      "sha256": "68c68406964ac3c551c8d350e58134eeef24b4b804b7ff50d2f34875716dfbb0"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5358.f32le": {
+      "bytes": 8192,
+      "sha256": "d3d5ebfdb9cc40e8df3fdb299615c6e09f41a7242c67a4cc20672417e977db15"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5359.f32le": {
+      "bytes": 64,
+      "sha256": "f920005af91a8e835dff529afc1f2da178b96a6ffa50c2aa3827770e945374c4"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5467.f32le": {
+      "bytes": 8192,
+      "sha256": "584ac46bc4fe4f9d13bf9bd02900b02642ef83e29977b481e1334e40b56f1b53"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5468.f32le": {
+      "bytes": 64,
+      "sha256": "53bca5f623355d27ba8f9a53c759835e6f087b49d5f407b1471d41e43765cf59"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5576.f32le": {
+      "bytes": 8192,
+      "sha256": "9acf49772686defd2d43e538c72938b05268eee49dc99da998ae548fed396e13"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5577.f32le": {
+      "bytes": 64,
+      "sha256": "5701629a49fdbb768ab65cbc350b7165bf5971ab23aff23b39416f447fc49946"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5685.f32le": {
+      "bytes": 8192,
+      "sha256": "a596e5691cb5bf07ad62af3f541be551e434490aea103e12d9e7778293886219"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5686.f32le": {
+      "bytes": 64,
+      "sha256": "6a87bba2339718192ee1bad1a8e73c24ef9c2f6de36da177958b2709fc7354a6"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5794.f32le": {
+      "bytes": 8192,
+      "sha256": "dd802d89f1f3c4c5cb75687f51fed3829e950b6ba1879103bb68a59576d8ba36"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5795.f32le": {
+      "bytes": 64,
+      "sha256": "98cceaea01bf5bbb0a4aaf33bc109456c9a0b680e75daad852ef982d1f78e453"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5903.f32le": {
+      "bytes": 8192,
+      "sha256": "b398f850a9e63d11595cb2e366466787f66cd983b5661fc8c32387a988fdc1bb"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5904.f32le": {
+      "bytes": 64,
+      "sha256": "4b5759daab893b5c0b995fb3ec265ef6b213c1bfceead6bce7be20fe00352ea2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6012.f32le": {
+      "bytes": 8192,
+      "sha256": "5beb5350ecfda3a3ee1f4373ae4963fa1fe8446640da40417864b6e7cd45182b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6013.f32le": {
+      "bytes": 64,
+      "sha256": "f335de41562407dae49acfef40b894b8f589e46cb7b97496c79b3fe27ae92bd4"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6121.f32le": {
+      "bytes": 8192,
+      "sha256": "3853943bbfeffbd4bdb2859473b7cc36cc0657fc202a8d58ba45b74f7104c4b2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6122.f32le": {
+      "bytes": 64,
+      "sha256": "fa80cd553b89c503d1ff62ce6ddee8ef2c0fcab8effd92b66f026885ac73d5bc"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6230.f32le": {
+      "bytes": 8192,
+      "sha256": "952eac836d2bbfc0889d5c0a4b7ed7f1d6fe82b00bd2bd1c7f32c3f6f577331e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6231.f32le": {
+      "bytes": 64,
+      "sha256": "44969c46575a1f474c8b427c6f5e17f24a0a854736a5c179bc66e76c68f405b7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6339.f32le": {
+      "bytes": 8192,
+      "sha256": "ccfe8c09fcb21c89534ab8985990b6ac83ded3add67c6551f1af8abf6b782338"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6340.f32le": {
+      "bytes": 64,
+      "sha256": "7337853eae863ab65390a6125b22a4076f038ada00c2db7b921df1bc34c5bb05"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6448.f32le": {
+      "bytes": 8192,
+      "sha256": "f3f61d0612993e4cd1ef4590041d3fe587df9c7156934ec9d1516cc7b6a277ca"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6449.f32le": {
+      "bytes": 64,
+      "sha256": "92835b8506605cfd4c8099586f954f5f17e892dd70f93e6c8e806f165e49939c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6557.f32le": {
+      "bytes": 8192,
+      "sha256": "b12c7354e8514958460d4a2d42bbe11536dc0cd9939ce8ee5ca8101a83c0c9cb"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6558.f32le": {
+      "bytes": 64,
+      "sha256": "a9007b8c96331a0bf30a9b4c35fff51fdafb41ac937c9283b3c8fadd8d7d5b5f"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6666.f32le": {
+      "bytes": 8192,
+      "sha256": "2a3ab0c598316d8d8bc4225622045664fe25bd736a446ba678c38a990b72fdfd"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6667.f32le": {
+      "bytes": 64,
+      "sha256": "e58345262462af5ed86b9d443dc652972338615a5637e8052f60bd117506d0c7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6775.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6776.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6884.f32le": {
+      "bytes": 8192,
+      "sha256": "78e3cdb01cdafbba49da82158d2e6a48be24b24d62c54e0911c398a84f994a94"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6885.f32le": {
+      "bytes": 64,
+      "sha256": "68806d6d15514e1c902a0f80b0275a754436b2135d04a4b4a385a5ee8292653c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/bulk64/field-3.f32le": {
+      "bytes": 524288,
+      "sha256": "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include/llama.h": {
+      "bytes": 89978,
+      "sha256": "c9a5f025ad7dbf07eda6660c8ff2c32cd80c07ba440d80728707a9c9d13c1bc3"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/src/models/qwen35.cpp": {
+      "bytes": 28742,
+      "sha256": "deec1f546ed1562a8fb8811fe21ce823ef3ed66bae2dde7a3c791068e89a9efd"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/src/llama-graph.cpp": {
+      "bytes": 144062,
+      "sha256": "a9acd6050aad04e82dee31a13a013656195c83c56e2554a2f07d997e25d8cae4"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml.h": {
+      "bytes": 116144,
+      "sha256": "12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml-alloc.h": {
+      "bytes": 3689,
+      "sha256": "d114067f9099c4a0e2bbc0625c676d4894d9cf0ea27168ee072c7de0e71bb51d"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml-backend.h": {
+      "bytes": 25776,
+      "sha256": "5791edae1fb7622027621d15bc7a160198bd3b75ed8f96d06cb7b85e2b873756"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml.c": {
+      "bytes": 261112,
+      "sha256": "7e8bf2b60598b50d718f46aa3c36c9293b10f4e7dfde6844f0693e6fc976d505"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend.cpp": {
+      "bytes": 105561,
+      "sha256": "cdb79f2937aa147084a76f949adc0eb760ad1883e0ccc015bb4e0f599c89bd5f"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend-dl.cpp": {
+      "bytes": 1051,
+      "sha256": "acd76df8b83bcdef6807bc7b0520f57b379d61b4ff1027c04aa4d140d16ad07f"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend-reg.cpp": {
+      "bytes": 18555,
+      "sha256": "0f63c69ad083e0744872f57d59d049b069283303f5bbc646222907d779afe4d1"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.cpp": {
+      "bytes": 25005,
+      "sha256": "ccf382d3fbf126b5ddb5850a0a970656fb24e00587da4003419439277c91cd01"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.c": {
+      "bytes": 132440,
+      "sha256": "45b22f8c83b2cbccabfc2f81a38ef8a5810223e1ce366b7c6f77d42faa5cc6f3"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/repack.cpp": {
+      "bytes": 213604,
+      "sha256": "5fa9f24cc2631609f40fa47a64078f77dccbe9f327711af91af671cd29b8b87d"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/tiled/tiled.cpp": {
+      "bytes": 58049,
+      "sha256": "a5d92ce8f617696778e225de6658180649143fec3792c5cfef84170641f2d06e"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/llamafile/sgemm.h": {
+      "bytes": 524,
+      "sha256": "6ec12aa9244a8ffb13ae56b2e391f905b1d152ae25ff45da6a66e18daf650b3b"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/llamafile/sgemm.cpp": {
+      "bytes": 166263,
+      "sha256": "8413c9a0828ddea7289aee72a32c03bd56a23ee40179ec3600f9b0589a21e338"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/vec.cpp": {
+      "bytes": 25373,
+      "sha256": "31e43ec66a1e2920a549dd2f71e6ce474e68f2fc298c4d2942404ca327e763d1"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/simd-mappings.h": {
+      "bytes": 55790,
+      "sha256": "a936298928e4dcfc1b61fb9e99cf8a152572b413e30f38dc8e7c3211ee9cb6a0"
+    },
+    "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml.so.0.26.0": {
+      "bytes": 58424,
+      "sha256": "1df83a5f445c119d10cd4fcac260c150a32d1b2ddf01dd49c9686c8b958f5c70"
+    },
+    "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml-base.so.0.26.0": {
+      "bytes": 972584,
+      "sha256": "dc4856a2ad667979352c9f270274b84f8135575c7394bd3b00e415e9e371e903"
+    },
+    "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so": {
+      "bytes": 1459208,
+      "sha256": "ca75af8b6281c57114f411d8883120a4ee595d8cb05c998a0b5e40921417c935"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/target/models/qwen35-2b/model.gguf": {
+      "bytes": 1296764000,
+      "sha256": "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/INPUT-CLOSURE.json": {
+      "bytes": 1488,
+      "sha256": "8f73dd76505ec074f0aed61d96ea2cc22ba0404e0e2121df9e5bc13b465683cb"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml-cpu.h": {
+      "bytes": 7753,
+      "sha256": "316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e"
+    }
+  },
+  "total_source_payload_copy_cap_bytes": 4194304,
+  "intended_route": "F32 tiled-ineligible; public sgemm n64 accepted versusn1 early refusal; public use_ref skips sgemm. Actual entry/control/output proof ROOT pending.",
+  "runtime_closure_scope": "Pins complete relevant source/ggml DSO files and records actualmaps; no entire host-runtime/internalmicrokernel qualification claim"
+}
--- /dev/null
+++ original2B-alpha-public-replay/INPUT-CLOSURE.json
@@ -0,0 +1,55 @@
+{
+  "scope": "Byte-only source/capture audit; no operator/model execution; first OBSERVED gap, not automatic global cause",
+  "preceding_exact_rows": 960,
+  "preceding_stage_families": 15,
+  "actual_input_words": 131072,
+  "actual_input_shape_bulk": [
+    2048,
+    64,
+    1,
+    1
+  ],
+  "actual_input_shape_AR1": [
+    2048,
+    1,
+    1,
+    1
+  ],
+  "all64_actual_inputs_byte_equal": true,
+  "input_sha256": "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6",
+  "weight_source_interval": {
+    "start": 459622304,
+    "bytes": 131072,
+    "end_exclusive": 459753376,
+    "shape": [
+      2048,
+      16,
+      1,
+      1
+    ],
+    "kind": "F32_GGUF_RAW",
+    "sha256": "0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876"
+  },
+  "first_observed_causal_scope_difference": {
+    "layer": 0,
+    "stage": "projection_output::blk.0.ssm_alpha.weight",
+    "occurrence": 0,
+    "position": 0,
+    "causal_scope": "all actual token-result words",
+    "all_raw_unequal_words": 13,
+    "causal_unequal_words": 13,
+    "first_raw_mismatch": {
+      "index": 0,
+      "bulk_bits": "bf1ef412",
+      "ar1_bits": "bf1ef414"
+    },
+    "first_causal_mismatch": {
+      "index": 0,
+      "bulk_bits": "bf1ef412",
+      "ar1_bits": "bf1ef414"
+    }
+  },
+  "weight_readset_scope": "Actual same ordinaryCPU_Mapped owner, fullbytes independently exact to source; no opaque repack enters this operator",
+  "internal_dispatch_observed": false,
+  "cause_qualified": false
+}
--- /dev/null
+++ original2B-alpha-public-replay/README.md
@@ -0,0 +1,123 @@
+# Original2B alpha projection public replay — ROOT EXECUTES
+
+Parent: numerics-and-context. Owner: spine_review. New ignored diagnostic only.
+No MAIN/plan/math edits, build, native/operator/model/container/GPU execution by
+this author. Preparation used source, metadata and bounded raw-byte comparison.
+Original capture/helpers/frozen attempts remain untouched.
+
+## Earliest actual input closure
+
+Accepted root-first64-repack-transport capture independently retains plain versus
+observed whole post-state and next real AR1 full248320/whole-state transparency.
+Its first observed gap is layer0 projection_output::blk.0.ssm_alpha.weight at
+position0:13/16 words, first bulkbf1ef412 versus AR1bf1ef414, before bias/softplus.
+
+- All960 preceding compared rows (15 captured stage families x64) are exact:
+  embedding, actual RMS/weighted norm, QKV, conv four-tap inputs/results, Q/K/V,
+  and every alpha input. This is captured-scope order, not an unobserved global
+  first-cause claim.
+- Independently read all131072 actual F32 alpha-input words. Bulk [2048,64,1,1]
+  contiguous bytes equal every corresponding64 AR1 [2048,1,1,1] input exactly;
+  SHA731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6.
+  Actual source tensorID5 attn_norm-0/MUL matches the captured producer and each
+  actual projection src1, checked before/after. No tokenizer/state reconstruction.
+- Full32768 F32 learned alpha words, shape[2048,16,1,1],131072B, SHA
+  0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876,
+  independently equal original approved GGUF bytes [459622304,459753376).
+  Actual source buffer is ordinary CPU_Mapped/GGUF_RAW; opaque CPU_REPACK bytes
+  from other model operators do not enter this standalone F32 projection.
+- Exact original GGUF SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6,
+  full source/header/capture metadata and relevant original source/DSO files are
+  pinned. Author read only bounded alpha source range; ROOT authenticate performs
+  whole-source before/after hashing. INPUT-CLOSURE.json records this audit.
+
+## Genuine public computation and observation
+
+- Same pinned original llama.cpp0.6.0 Haswell CPU backend and public
+  ggml_mul_mat(F32 weight[2048,16],F32 input[2048,64 or1]); all ne/nb/dtypes checked.
+  CPU payloads are populated with unchanged actual raw bits. No model load,
+  dequantization, custom dot loop, kernel replacement or mathematical fallback.
+  Input tensor/buffer ownership is independent diagnostic storage, not an alias
+  or substitute model/source identity.
+- Original64-row operator plus64 original AR1 operators,8 backend threads.
+  Every default output must bit-match its entire authentic capture:1024 bulk
+  words+64x16 AR1words=2048. Any default mismatch refuses BEFORE any control.
+- Only after all65 defaults pass, repeat65 calls using the explicitly public
+  ggml_backend_cpu_set_use_ref control. This is a named diagnostic condition,
+  not original model parity or permission to change Rust/production kernels.
+  Record every control output and complete refbulk versus defaultAR1 comparison.
+- Plain no-observer run first; separate same binary/inputs run with source-only
+  observer. ALL130 full default/control outputs in each run must be identical
+  across plain/observed before route evidence is accepted (4096words/run).
+- Optional LD_PRELOAD observer uses exact public sgemm.h13-argument ABI and
+  unchanged opaque params. Original public target is exact already-loaded
+  Haswell DSO via NOLOAD+dlsym+dladdr at authenticated exported offset0x10c8f0.
+  Call original exactly once, forward all pointers/scalars and return original
+  bool unchanged. No private params layout read, arithmetic or substitution.
+- Integer-only bounded2048-call ring;8 expected source entry calls/bulk and each
+  AR1 call, no control calls. Source-derived intended population:8 bulk64 true,
+  64x8 AR1 false,65 use_ref zero =520. Exact m16/n64-or1/k2048, lda/ldb2048,
+  ldc16, F32 types, zero bound operand offsets. ROOT must actually establish it.
+- Primary source: qwen35.cpp379-381/build_lora_mm uses this exact projection;
+  tiled.cpp rejects F32 source; ggml-cpu.c1308+ gates the public sgemm branch;
+  sgemm.cpp3911 returns false for n<2 and its F32 AVX branch selects tinyBLAS;
+  use_ref skips sgemm. These are source branch facts, not observed internal
+  instruction/microkernel routes. The wrapper records exported entry only.
+- Actual maps before/after must show only pinned Haswell CPU backend. Exact
+  relevant source and ggml DSO pins do not establish entire host runtime closure.
+  Internal microkernel/long-context cause/profile/model eligibility remain false.
+  Existing Rust declared AR1 kernels are untouched; no fitting nativebulk math.
+
+## Bounds and ROOT procedure
+
+Fresh private fixtures<4MiB (actual payload1,187,840B), expected values copied only
+from authentic actual captures, never candidate math. C++ opens expected payloads
+for comparison only after genuine public computation; outer source authentication
+hashes fixture files before/after. Every input is re-read in CPU storage and file
+form after each operator. F32 finite checks preserve all raw bits/signedzero.
+
+No network/GPU/model loading. Process AS<=8GiB, CPU deadline<=300x8+1s, wall300s,
+8 backend threads, backing<=2MiB/case, graph metadata2MiB/case. Owned PGID launcher
+is copied from accepted AV wrapper: WNOWAIT preserves leader identity, unconditional
+TERM/KILL/reap/group-absence and pipeEOF/closure happen before finalreceipt I/O,
+even if leader already exited. Logs<=8MiB/run; binary<=16MiB, observer<=8MiB.
+Entire new ignored-run artifact tree<=512MiB including reserved bounded report.
+No old capture/failed attempt is overwritten. Python AST only by author; no tests.
+
+ROOT verifies freeze and source pins, then builds with its accepted GCC/glibc2.44
+closure and strict warnings. No fast-math or upstream build. Root alone runs this
+after competing model/CPU/GPU work permits; placeholders require actual hashes.
+
+```sh
+TASK_ALPHA=.agents/runs/numerics-and-context-2b-alpha-replay-20261020
+TASK_NATIVE=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
+TASK_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
+TASK_C=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
+python3 "$TASK_ALPHA/prepare.py" "$TASK_ALPHA/root-fixtures"
+c++ -std=c++17 -O2 -Wall -Wextra -Werror \
+  -I "$TASK_NATIVE/ggml/include" "$TASK_ALPHA/main.cpp" \
+  -L "$TASK_LIB" -L "$TASK_C" -Wl,-rpath,"$TASK_LIB:$TASK_C" \
+  -Wl,--dynamic-linker,"$TASK_C/ld-linux-x86-64.so.2" \
+  -lggml -lggml-base -ldl -lm -o "$TASK_ALPHA/root-replay"
+c++ -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC \
+  -I "$TASK_NATIVE/ggml/src/ggml-cpu/llamafile" "$TASK_ALPHA/observer.cpp" \
+  -L "$TASK_C" -Wl,-rpath,"$TASK_C" -ldl -o "$TASK_ALPHA/root-observer.so"
+python3 "$TASK_ALPHA/run_replay.py" --binary "$TASK_ALPHA/root-replay" \
+  --binary-sha256 ACTUAL_REPLAY_SHA --fixtures "$TASK_ALPHA/root-fixtures" \
+  --new-output "$TASK_ALPHA/root-plain" --new-log "$TASK_ALPHA/root-plain.log" \
+  --new-receipt "$TASK_ALPHA/root-plain.invocation.json"
+python3 "$TASK_ALPHA/run_replay.py" --binary "$TASK_ALPHA/root-replay" \
+  --binary-sha256 ACTUAL_REPLAY_SHA --fixtures "$TASK_ALPHA/root-fixtures" \
+  --preload "$TASK_ALPHA/root-observer.so" --preload-sha256 ACTUAL_OBSERVER_SHA \
+  --new-output "$TASK_ALPHA/root-observed" --new-log "$TASK_ALPHA/root-observed.log" \
+  --new-receipt "$TASK_ALPHA/root-observed.invocation.json"
+python3 "$TASK_ALPHA/validate.py" --plain-output "$TASK_ALPHA/root-plain" \
+  --plain-receipt "$TASK_ALPHA/root-plain.invocation.json" \
+  --observed-output "$TASK_ALPHA/root-observed" \
+  --observed-receipt "$TASK_ALPHA/root-observed.invocation.json" \
+  --new-report "$TASK_ALPHA/root-validation.json"
+```
+
+Successful output means full public-operator replay/observer transparency and
+exported-entry diagnostic evidence. It does not close full runtime/internal
+kernel causality, historical8K corpus equivalence, HF/vLLM math or model support.
--- /dev/null
+++ original2B-alpha-public-replay/TODO.md
@@ -0,0 +1,18 @@
+# Original2B alpha first-gap public replay
+
+Parent: numerics-and-context. Owner: spine_review; ignored new run only.
+ROOT owns all builds/tests/native/operator/model/GPU/container execution.
+
+- [x] Independently inspect actual first64 capture/comparison;960 preceding rows
+  exact, full131072 alpha input words exact and actual producer binding closed.
+- [x] Independently byte-match all32768 alpha F32 weights to original GGUF
+  range[459622304,459753376); distinguish ordinary raw from opaque otherweights.
+- [x] Prepare genuine public ggml full64+64AR default replay, gated publicuse_ref
+  controls, unchanged ABI/return entry observer and full plain/observed validator.
+- [x] Preserve existing helpers/attempts; pin all relevant capture/source/DSO files,
+  bounded owned lifecycle, exact finite shapes/outputs and no internalcause claim.
+- [ ] Independent source review, ROOT strict build and byte-only fixture creation.
+- [ ] ROOT actual all65 default2048 words exact BEFORE65 controls; all130
+  plain/observed4096 words exact BEFORE520 exported-entry route population accepted.
+- [ ] Separately interpret actual control results/source branch proof. Do not alter
+  declared Rust AR1 math by fitting nativebulk; no automatic full-longcontext cause.
--- /dev/null
+++ original2B-alpha-public-replay/UPSTREAM_LICENSE
@@ -0,0 +1,21 @@
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

````

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/frozen-cleanup-v2/CLEANUP_V2.patch

Original UTF-8 bytes: 8904; SHA256 `28598954899beec157e29ddad844728045bc943bfc3bf5f5066dda83eb26da7e`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- unchanged-run_replay.py
+++ additive-run_replay_v2.py
@@ -14,52 +14,91 @@
 LIB = '/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib'
 
 
-def group_exists(pgid):
+def unreaped(child):
+    require(child.returncode is None, "leader already reaped")
+    value = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if value is None:
+        return None
+    require(value.si_pid == child.pid, "owned child identity")
+    return value.si_status if value.si_code == os.CLD_EXITED else -value.si_status
+
+
+def cleanup_owned(child, record, errors, *, observe=unreaped, pause=time.sleep):
+    """Independent bounded stages; never signal after the first reap attempt."""
+    def bound():
+        observe(child)
+        require(os.getpgid(child.pid) == child.pid and os.getsid(child.pid) == child.pid,
+                "owned session/group binding")
+        return child.pid
+
     try:
-        os.killpg(pgid, 0)
-        return True
+        os.killpg(bound(), signal.SIGTERM)
+        record["group_term_before_reap"] = True
     except ProcessLookupError:
-        return False
-
-
-def signal_group(pgid, sig):
+        record["group_term_absent"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
     try:
-        os.killpg(pgid, sig)
-        return True
-    except ProcessLookupError:
-        return False
+        grace = time.monotonic() + 0.5
+        while time.monotonic() < grace:
+            observe(child)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # TERM/grace/one KILL error cannot skip the separately bound KILL stage.
+    # Both attempts occur before ANY wait, while the owned leader is unreaped.
+    for attempt in range(2):
+        try:
+            os.killpg(bound(), signal.SIGKILL)
+            record["group_kill_before_reap"] = True
+            break
+        except ProcessLookupError:
+            record["group_kill_absent_before_reap"] = True
+            break
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    # ALWAYS attempt reap even after signal/ownership failures. Popen.wait only
+    # targets this child. An interrupted first wait cannot skip the final wait.
+    # No signal operation occurs below this point, including after timeout.
+    try:
+        record["reaped_exit_code"] = child.wait(timeout=5)
+        record["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned initial reap: " + repr(error))
+    finally:
+        if child.returncode is None:
+            try:
+                record["reaped_exit_code"] = child.wait(timeout=5)
+                record["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final reap: " + repr(error))
+        else:
+            record["reaped_exit_code"] = child.returncode
+            record["leader_reaped"] = True
+    if record.get("leader_reaped"):
+        try:
+            deadline = time.monotonic() + 5
+            while True:
+                try:
+                    os.killpg(child.pid, 0)  # Read-only absence check, NEVER kill after reap.
+                except ProcessLookupError:
+                    record["group_absent"] = True
+                    break
+                require(time.monotonic() < deadline, "owned process group remains")
+                pause(0.05)
+        except BaseException as error:
+            errors.append("owned group absence: " + repr(error))
+
 
 
 def leader_exit_without_reaping(process, seconds):
-    # Keep the owned leader PID reserved until group cleanup begins.
     deadline = time.monotonic() + seconds
     while time.monotonic() < deadline:
-        observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+        observed = unreaped(process)
         if observed is not None:
-            return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+            return observed
         time.sleep(0.05)
     raise TimeoutError('owned leader exit deadline')
-
-
-def cleanup_owned_group(process, receipt):
-    pgid = process.pid  # start_new_session=True; never an inherited/shared group.
-    cleanup = receipt['cleanup'] = {'pgid': pgid, 'term_sent': False, 'kill_sent': False,
-                                    'leader_reaped': False, 'group_absent': False}
-    cleanup['term_sent'] = signal_group(pgid, signal.SIGTERM)
-    try:
-        process.wait(timeout=3)
-    except subprocess.TimeoutExpired:
-        cleanup['kill_sent'] = signal_group(pgid, signal.SIGKILL)
-        process.wait(timeout=5)
-    cleanup['leader_reaped'] = True
-    # A reaped leader does not establish that its owned descendants retired.
-    if group_exists(pgid):
-        cleanup['kill_sent'] = signal_group(pgid, signal.SIGKILL) or cleanup['kill_sent']
-    deadline = time.monotonic() + 5
-    while group_exists(pgid) and time.monotonic() < deadline:
-        time.sleep(0.05)
-    cleanup['group_absent'] = not group_exists(pgid)
-    require(cleanup['group_absent'], 'owned process group remains after cleanup')
 
 
 def finish_pipe(process, log, log_bytes, receipt):
@@ -92,6 +131,14 @@
     p.add_argument('--preload', type=Path)
     p.add_argument('--preload-sha256')
     a = p.parse_args()
+    followup = jread(HERE/'cleanup-v2-manifest.json')
+    require(followup['original_freeze_sha256'] == 'ee78cf570d4d9f3f48378a90c7a916436eeb3791906ca8c3a1c03e188b3cce60'
+            and digest(HERE/'freeze-manifest.json')['sha256'] == followup['original_freeze_sha256'],
+            'unchanged original alpha preparation manifest')
+    for name, expected in jread(HERE/'freeze-manifest.json')['files'].items():
+        require(digest(HERE/name) == expected,'unchanged original alpha prepared source')
+    for name, expected in followup['files'].items():
+        require(digest(HERE/name) == expected,'exact cleanup followup source pin')
     pins = authenticate()
     capture_guards()
     for path in [a.new_output, a.new_log, a.new_receipt]:
@@ -123,6 +170,8 @@
                'actual_gpu_or_model_execution': False, 'internal_dispatch_observed': False,
                'preload': str(a.preload.resolve()) if a.preload else None, 'preload_digest': before_preload,
                'source_profile': 'pinned llama.cpp0.6.0 haswell public alpha default/control diagnostic',
+               'runner_sha256':digest(Path(__file__)),
+               'cleanup_v2_manifest_sha256':digest(HERE/'cleanup-v2-manifest.json'),
                'exit_code': None, 'status': 'UNQUALIFIED'}
     with a.new_receipt.open('x') as receipt_file, a.new_log.open('xb') as log:
         process = None
@@ -158,14 +207,18 @@
             receipt['error'] = f'{type(error).__name__}: {error}'
             primary_error = error
         finally:
+            cleanup_errors = []
             try:
                 if process is not None:
-                    cleanup_owned_group(process, receipt)
+                    receipt['cleanup'] = {'pgid':process.pid}
+                    cleanup_owned(process, receipt['cleanup'], cleanup_errors)
+            except BaseException as error:
+                cleanup_errors.append('unexpected owned cleanup: '+repr(error))
+            try:
+                if process is not None:
                     finish_pipe(process, log, log_bytes, receipt)
             except BaseException as error:
-                cleanup_error = error
-                receipt['cleanup_error'] = f'{type(error).__name__}: {error}'
-                receipt['status'] = 'UNQUALIFIED'
+                cleanup_errors.append('final owned pipe drain: '+repr(error))
             finally:
                 closers = []
                 if selector is not None:
@@ -178,10 +231,14 @@
                         receipt.setdefault('cleanup', {})[label] = True
                     except BaseException as error:
                         cleanup_error = cleanup_error or error
-                        receipt.setdefault('cleanup_errors', []).append(f'{label}: {type(error).__name__}: {error}')
+                        cleanup_errors.append(f'{label}: {type(error).__name__}: {error}')
                         receipt['status'] = 'UNQUALIFIED'
                 if process is not None:
                     receipt['exit_code'] = process.returncode
+            if cleanup_errors:
+                receipt['cleanup_errors'] = cleanup_errors
+                receipt['status'] = 'UNQUALIFIED'
+                cleanup_error = cleanup_error or RuntimeError('; '.join(cleanup_errors))
             receipt['elapsed_seconds'] = time.monotonic() - started
             json.dump(receipt, receipt_file, indent=2, allow_nan=False)
             receipt_file.write('\n')

````

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/frozen-prepared/NEW_DELTA.patch

Original UTF-8 bytes: 113153; SHA256 `25fb733d7171fbf5fe07ad09525f63b92923e491804d36db5db775c7ca9a27e0`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ original2B-alpha-public-replay/main.cpp
@@ -0,0 +1,247 @@
+// SPDX-License-Identifier: Apache-2.0
+// Diagnostic public ggml operator only; upstream headers retain MIT notices.
+// No model load, arithmetic replacement, production linkage or eligibility.
+#include "ggml.h"
+#include "ggml-alloc.h"
+#include "ggml-backend.h"
+#include "ggml-cpu.h"
+#include "observer.h"
+#include <cstdlib>
+#include <array>
+#include <chrono>
+#include <cstring>
+#include <dlfcn.h>
+#include <filesystem>
+#include <fstream>
+#include <iostream>
+#include <limits>
+#include <memory>
+#include <stdexcept>
+#include <string>
+#include <sys/resource.h>
+#include <vector>
+
+namespace fs = std::filesystem;
+constexpr size_t K = 2048, M = 16;
+constexpr size_t W_BYTES = K * M * sizeof(float);
+constexpr size_t ARTIFACT_CAP = 512 * 1024 * 1024;
+static size_t written = 0;
+static bool tracing = false;
+static decltype(&zg_alpha_trace_begin) trace_begin = nullptr;
+static decltype(&zg_alpha_trace_end) trace_end = nullptr;
+static decltype(&zg_alpha_trace_save) trace_save = nullptr;
+static void load_observer() {
+    const char *enabled = std::getenv("ZG_ALPHA_TRACE");
+    if (!enabled) return;
+    if (std::strcmp(enabled, "1") != 0) throw std::runtime_error("exact observer enable value");
+    void *begin = dlsym(RTLD_DEFAULT, "zg_alpha_trace_begin");
+    void *end = dlsym(RTLD_DEFAULT, "zg_alpha_trace_end");
+    void *save = dlsym(RTLD_DEFAULT, "zg_alpha_trace_save");
+    if (!begin || !end || !save) throw std::runtime_error("explicit preloaded observer unavailable");
+    static_assert(sizeof(begin) == sizeof(trace_begin));
+    static_assert(sizeof(end) == sizeof(trace_end));
+    static_assert(sizeof(save) == sizeof(trace_save));
+    std::memcpy(&trace_begin, &begin, sizeof(trace_begin));
+    std::memcpy(&trace_end, &end, sizeof(trace_end));
+    std::memcpy(&trace_save, &save, sizeof(trace_save));
+    tracing = true;
+}
+static std::chrono::steady_clock::time_point deadline;
+
+static void require(bool yes, const std::string &message) {
+    if (!yes) throw std::runtime_error(message);
+}
+static bool expired(void *) { return std::chrono::steady_clock::now() >= deadline; }
+static std::vector<unsigned char> read_exact(const fs::path &path, size_t bytes) {
+    require(!fs::is_symlink(path) && fs::is_regular_file(path) && fs::file_size(path) == bytes,
+            "exact regular input: " + path.string());
+    std::vector<unsigned char> result(bytes);
+    std::ifstream in(path, std::ios::binary);
+    require(bool(in), "input open");
+    in.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(bytes));
+    require(bool(in) && in.peek() == std::char_traits<char>::eof(), "exact immutable-length input");
+    for (size_t at = 0; at < bytes; at += 4) {
+        uint32_t word = 0; std::memcpy(&word, result.data() + at, 4);
+        require((word & 0x7f800000u) != 0x7f800000u, "finite actual operand/output");
+    }
+    return result;
+}
+static void write_bytes(const fs::path &path, const void *data, size_t bytes) {
+    require(written <= ARTIFACT_CAP && bytes <= ARTIFACT_CAP - written, "aggregate output cap");
+    require(!fs::exists(path), "exclusive output file");
+    std::ofstream out(path, std::ios::binary);
+    require(bool(out), "output open");
+    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
+    out.flush(); require(bool(out), "output write"); written += bytes;
+}
+static void maps(const fs::path &out, const fs::path &cpu) {
+    std::ifstream in("/proc/self/maps"); require(bool(in), "maps open");
+    std::string text, line; bool found = false;
+    while (std::getline(in, line)) {
+        require(text.size() + line.size() + 1 <= 1024 * 1024, "bounded maps");
+        if (line.find(cpu.string()) != std::string::npos) found = true;
+        if (line.find("libggml-cpu-") != std::string::npos)
+            require(line.find(cpu.string()) != std::string::npos, "foreign CPU backend");
+        text += line + '\n';
+    }
+    require(in.eof() && found, "actual pinned CPU backend mapped");
+    write_bytes(out, text.data(), text.size());
+}
+static std::string name(size_t call) {
+    require(call < 64, "actual AR1 index");
+    return "ar1-" + std::to_string(call);
+}
+static int positive(const char *argument, int maximum) {
+    const std::string text(argument); size_t used = 0;
+    const long value = std::stol(text, &used);
+    require(used == text.size() && value >= 1 && value <= maximum, "bounded exact positive argument");
+    return static_cast<int>(value);
+}
+struct Result { size_t unequal = 0; size_t first = 0; uint32_t actual = 0, expected = 0; };
+static Result compare(const std::vector<unsigned char> &actual, const std::vector<unsigned char> &expected) {
+    require(actual.size() == expected.size(), "complete output geometry");
+    Result result;
+    for (size_t at = 0; at < actual.size(); at += 4) {
+        uint32_t a, b; std::memcpy(&a, actual.data() + at, 4); std::memcpy(&b, expected.data() + at, 4);
+        if (a != b) {
+            if (result.unequal == 0) { result.first = at / 4; result.actual = a; result.expected = b; }
+            ++result.unequal;
+        }
+    }
+    return result;
+}
+static std::vector<unsigned char> replay(ggml_backend_t backend, size_t rows, uint32_t phase, uint32_t which,
+                                       const std::vector<unsigned char> &wraw,
+                                       const std::vector<unsigned char> &xraw) {
+    require(rows == 1 || rows == 64, "unchanged actual call width");
+    require(wraw.size() == W_BYTES && xraw.size() == K * rows * 4, "actual alpha input geometry");
+    require(!expired(nullptr), "finite replay deadline");
+    ggml_init_params init{}; init.mem_size = 2 * 1024 * 1024; init.no_alloc = true;
+    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(init), ggml_free);
+    require(bool(ctx), "bounded graph metadata allocation");
+    auto *w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
+    auto *x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, rows);
+    auto *y = ggml_mul_mat(ctx.get(), w, x);
+    const std::array<int64_t, 4> wne{2048,16,1,1}, xne{2048,static_cast<int64_t>(rows),1,1};
+    const std::array<int64_t, 4> yne{16,static_cast<int64_t>(rows),1,1};
+    const std::array<size_t, 4> wnb{4,8192,131072,131072}, xnb{4,8192,K*rows*4,K*rows*4};
+    const std::array<size_t, 4> ynb{4,64,M*rows*4,M*rows*4};
+    for(size_t i=0;i<4;++i) {
+        require(w->ne[i]==wne[i] && w->nb[i]==wnb[i], "exact captured original F32 weight layout");
+        require(x->ne[i]==xne[i] && x->nb[i]==xnb[i], "exact captured original alpha input layout");
+        require(y->ne[i]==yne[i] && y->nb[i]==ynb[i], "exact captured original alpha output layout");
+    }
+    require(w->type==GGML_TYPE_F32 && x->type==GGML_TYPE_F32 && y->type==GGML_TYPE_F32
+            && y->op==GGML_OP_MUL_MAT && ggml_is_contiguous(w) && ggml_is_contiguous(x)
+            && ggml_is_contiguous(y), "original public F32 alpha projection");
+    auto *graph=ggml_new_graph_custom(ctx.get(),16,false); require(graph, "graph metadata");
+    ggml_build_forward_expand(graph,y);
+    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
+        ggml_backend_alloc_ctx_tensors(ctx.get(),backend),ggml_backend_buffer_free);
+    require(bool(buffer) && ggml_backend_buffer_get_size(buffer.get()) <= 2*1024*1024,
+            "bounded genuine CPU backing allocation");
+    ggml_backend_tensor_set(w,wraw.data(),0,wraw.size());
+    ggml_backend_tensor_set(x,xraw.data(),0,xraw.size());
+    if(tracing)require(trace_begin(phase,which,static_cast<uint32_t>(rows),w->data,x->data,y->data),"observer begin");
+    const auto status=ggml_backend_graph_compute(backend,graph);
+    if(tracing)require(trace_end(),"observer end after synchronous completion");
+    require(status==GGML_STATUS_SUCCESS,"public alpha operator completion");
+    std::vector<unsigned char> result(M*rows*4),after_w(wraw.size()),after_x(xraw.size());
+    ggml_backend_tensor_get(y,result.data(),0,result.size());
+    ggml_backend_tensor_get(w,after_w.data(),0,after_w.size());
+    ggml_backend_tensor_get(x,after_x.data(),0,after_x.size());
+    require(after_w==wraw && after_x==xraw,"operator changed actual original input bytes");
+    for(size_t at=0;at<result.size();at+=4) {
+        uint32_t word;std::memcpy(&word,result.data()+at,4);
+        require((word&0x7f800000u)!=0x7f800000u,"finite public alpha result");
+    }
+    return result;
+}
+
+int main(int argc, char **argv) {
+    try {
+        load_observer();
+        require(argc == 6, "usage: replay FIXTURES NEW_OUTPUT CPU_DSO THREADS SECONDS");
+        const fs::path input = fs::canonical(argv[1]), output = argv[2], cpu = fs::canonical(argv[3]);
+        require(cpu == "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so",
+                "explicit pinned original CPU backend");
+        const int threads = positive(argv[4], 8), seconds = positive(argv[5], 300);
+        deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
+        rlimit address{8ull * 1024 * 1024 * 1024, 8ull * 1024 * 1024 * 1024};
+        rlimit cpu_limit{static_cast<rlim_t>(seconds * threads + 1), static_cast<rlim_t>(seconds * threads + 1)};
+        rlimit file_limit{ARTIFACT_CAP, ARTIFACT_CAP};
+        require(setrlimit(RLIMIT_AS, &address) == 0 && setrlimit(RLIMIT_CPU, &cpu_limit) == 0
+                && setrlimit(RLIMIT_FSIZE, &file_limit) == 0, "finite process limits");
+        const uint32_t endian = 1;
+        require(*reinterpret_cast<const unsigned char *>(&endian) == 1 && sizeof(float) == 4, "LE F32 host");
+        require(fs::create_directory(output), "new exclusive output directory");
+        auto *reg = ggml_backend_load(cpu.c_str());
+        require(reg && std::string(ggml_backend_reg_name(reg)) == "CPU" && ggml_backend_reg_dev_count(reg) == 1,
+                "original public CPU backend registration");
+        auto get = [reg](const char *key) { auto *p = ggml_backend_reg_get_proc_address(reg, key); require(p, key); return p; };
+        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(get("ggml_backend_set_n_threads"));
+        auto set_abort = reinterpret_cast<ggml_backend_set_abort_callback_t>(get("ggml_backend_set_abort_callback"));
+        using SetRef = decltype(&ggml_backend_cpu_set_use_ref);
+        auto set_ref = reinterpret_cast<SetRef>(get("ggml_backend_cpu_set_use_ref"));
+        Dl_info info{};
+        require(dladdr(reinterpret_cast<void *>(set_ref), &info) != 0 && info.dli_fname
+                && fs::canonical(info.dli_fname) == cpu, "public control belongs to exact CPU DSO");
+        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
+            ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr), ggml_backend_free);
+        require(bool(backend), "original CPU backend owner");
+        set_threads(backend.get(), threads); set_abort(backend.get(), expired, nullptr);
+        maps(output / "before.maps", cpu);
+        std::ofstream report(output / "cases.jsonl"); require(bool(report), "case report open");
+        std::vector<std::vector<unsigned char>> ar_actual(64);
+        size_t defaults = 0, controls = 0;
+        for (int control = 0; control < 2; ++control) {
+            // Ref control is unreachable until all65 complete default outputs matched.
+            if (control) require(defaults == 65, "default all-output gate before ref control");
+            set_ref(backend.get(), control != 0);
+            for (size_t which = 0; which < 65; ++which) {
+                require(!expired(nullptr), "bounded replay deadline");
+                const std::string key = which == 0 ? "bulk64" : name(which - 1);
+                const size_t rows = which == 0 ? 64 : 1, bytes = M * rows * 4;
+                const auto w = read_exact(input / "weight.f32le", W_BYTES);
+                const auto x = read_exact(input / (key + ".x.f32le"), K * rows * 4);
+                const auto actual = replay(backend.get(), rows, static_cast<uint32_t>(control), static_cast<uint32_t>(which), w, x);
+                // Expected output is first opened after genuine public computation.
+                const auto expected = read_exact(input / (key + ".expected.f32le"), bytes);
+                const Result result = compare(actual, expected);
+                const std::string prefix = control ? "ref-control-" : "default-";
+                write_bytes(output / (prefix + key + ".actual.f32le"), actual.data(), actual.size());
+                Result ar_comparison{};
+                if (!control && which) ar_actual[which - 1] = actual;
+                if (control && which == 0) {
+                    for (size_t at = 0; at < 64; ++at) {
+                        std::vector<unsigned char> slice(M*4);
+                        std::memcpy(slice.data(),actual.data()+at*M*4,M*4);
+                        ar_comparison.unequal += compare(slice, ar_actual[at]).unequal;
+                    }
+                }
+                report << "{\"mode\":\"" << (control ? "PUBLIC_USE_REF_CONTROL" : "DEFAULT_PUBLIC_ALPHA")
+                       << "\",\"case\":\"" << key << "\",\"words\":" << bytes / 4
+                       << ",\"actual_capture_unequal_words\":" << result.unequal
+                       << ",\"first_index\":" << result.first << ",\"first_actual_bits\":" << result.actual
+                       << ",\"first_capture_bits\":" << result.expected
+                       << ",\"ref_bulk_vs_all_default_ar1_unequal_words\":" << ar_comparison.unequal << "}\n";
+                report.flush(); require(bool(report), "case report write");
+                if (!control) { require(result.unequal == 0, "default original public operator differs from capture"); ++defaults; }
+                else ++controls;
+                require(w == read_exact(input / "weight.f32le", W_BYTES)
+                        && x == read_exact(input / (key + ".x.f32le"), K*rows*4), "file inputs changed during operator");
+            }
+        }
+        if (tracing) require(trace_save((output / "sgemm-trace.json").c_str()), "exclusive bounded observer trace");
+        maps(output / "after.maps", cpu);
+        require(defaults == 65 && controls == 65, "complete actual default/control populations");
+        std::ofstream done(output / "complete.json"); require(bool(done), "completion open");
+        done << "{\"status\":\"PUBLIC_ALPHA_REPLAY_DEFAULT_EXACT_CONTROL_DIAGNOSTIC\",\"default_cases\":65,\"control_cases\":65,\"threads\":"
+             << threads << ",\"default_words\":2048,\"cause_qualified\":false,\"production_math_changed\":false,\"internal_dispatch_observed\":false}\n";
+        done.flush(); require(bool(done), "completion write");
+        std::cout << "PUBLIC_ALPHA_REPLAY_DEFAULT_EXACT_CONTROL_DIAGNOSTIC defaults=65 words=2048\n";
+        return 0;
+    } catch (const std::exception &error) {
+        std::cerr << "PUBLIC_ALPHA_REPLAY_UNQUALIFIED " << error.what() << '\n'; return 3;
+    }
+}
--- /dev/null
+++ original2B-alpha-public-replay/observer.cpp
@@ -0,0 +1,143 @@
+// SPDX-License-Identifier: Apache-2.0
+// Diagnostic only. Exact exported source declaration; params remains opaque.
+#include "observer.h"
+#include "sgemm.h"
+#include <atomic>
+#include <cstdint>
+#include <cstring>
+#include <dlfcn.h>
+#include <fcntl.h>
+#include <sstream>
+#include <string>
+#include <unistd.h>
+
+namespace {
+constexpr const char *CPU = "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so";
+constexpr uint32_t MAX_CALLS = 2048, MAX_CASES = 130;
+using Original = decltype(&llamafile_sgemm);
+std::atomic<Original> original{nullptr};
+static_assert(std::atomic<Original>::is_always_lock_free);
+void *original_owner = nullptr; // Bounded NOLOAD reference retained to process exit.
+std::atomic<bool> active{false}, failed{false};
+std::atomic<uint32_t> call_count{0};
+static_assert(std::atomic<uint32_t>::is_always_lock_free);
+struct Context { uint32_t phase, which, width, first; uintptr_t a, b, c; } context{};
+struct Case { uint32_t phase, which, width, first, last; } cases[MAX_CASES]{};
+uint32_t case_count = 0;
+struct Call {
+    uint32_t phase, which;
+    int64_t m, n, k, lda, ldb, ldc;
+    uintptr_t a, b, c;
+    int atype, btype, ctype;
+    bool accepted;
+};
+Call calls[MAX_CALLS]{};
+
+bool resolve_original() {
+    if (original.load(std::memory_order_acquire)) return true;
+    original_owner = dlopen(CPU, RTLD_NOW | RTLD_NOLOAD);
+    if (!original_owner) return false;
+    void *symbol = dlsym(original_owner, "llamafile_sgemm");
+    Dl_info info{};
+    if (!symbol || !dladdr(symbol, &info) || !info.dli_fname || !info.dli_fbase
+        || std::strcmp(info.dli_fname, CPU) != 0
+        || reinterpret_cast<uintptr_t>(symbol) - reinterpret_cast<uintptr_t>(info.dli_fbase) != 0x10c8f0u)
+        return false;
+    Original function = nullptr;
+    static_assert(sizeof(symbol) == sizeof(function));
+    std::memcpy(&function, &symbol, sizeof(function));
+    original.store(function, std::memory_order_release);
+    return true;
+}
+
+bool write_all(int fd, const char *bytes, size_t length) {
+    while (length) {
+        const ssize_t n = write(fd, bytes, length);
+        if (n <= 0) return false;
+        bytes += n; length -= static_cast<size_t>(n);
+    }
+    return true;
+}
+} // namespace
+
+extern "C" bool zg_alpha_trace_begin(uint32_t phase, uint32_t which, uint32_t width,
+                                  const void *a, const void *b, const void *c) {
+    if (active.load(std::memory_order_acquire) || failed.load(std::memory_order_acquire)
+        || phase > 1 || which >= 65 || width != (which ? 1u : 64u)
+        || case_count != phase * 65 + which || !a || !b || !c || !resolve_original()) {
+        failed.store(true, std::memory_order_release); return false;
+    }
+    context = {phase, which, width, call_count.load(std::memory_order_acquire),
+               reinterpret_cast<uintptr_t>(a), reinterpret_cast<uintptr_t>(b), reinterpret_cast<uintptr_t>(c)};
+    active.store(true, std::memory_order_release);
+    return true;
+}
+
+extern "C" bool llamafile_sgemm(const struct ggml_compute_params *params,
+        int64_t m, int64_t n, int64_t k, const void *a, int64_t lda,
+        const void *b, int64_t ldb, void *c, int64_t ldc, int atype, int btype, int ctype) {
+    // Resolution occurs on the owner thread before the real graph is invoked.
+    // Never return false as an observer failure: that would change dispatch.
+    const Original function = original.load(std::memory_order_acquire);
+    if (!function) _exit(120);
+    const bool observed = active.load(std::memory_order_acquire);
+    const Context bound = observed ? context : Context{};
+    const bool accepted = function(params, m, n, k, a, lda, b, ldb, c, ldc, atype, btype, ctype);
+    const uint32_t index = call_count.fetch_add(1, std::memory_order_acq_rel);
+    if (!observed || index >= MAX_CALLS
+        || reinterpret_cast<uintptr_t>(a) < bound.a
+        || reinterpret_cast<uintptr_t>(b) < bound.b
+        || reinterpret_cast<uintptr_t>(c) < bound.c) {
+        failed.store(true, std::memory_order_release);
+    } else {
+        calls[index] = {bound.phase, bound.which, m, n, k, lda, ldb, ldc,
+                        reinterpret_cast<uintptr_t>(a) - bound.a,
+                        reinterpret_cast<uintptr_t>(b) - bound.b,
+                        reinterpret_cast<uintptr_t>(c) - bound.c, atype, btype, ctype, accepted};
+    }
+    return accepted;
+}
+
+extern "C" bool zg_alpha_trace_end(void) {
+    if (!active.exchange(false, std::memory_order_acq_rel) || case_count >= MAX_CASES) {
+        failed.store(true, std::memory_order_release); return false;
+    }
+    // The original synchronous graph-compute joins all worker calls first.
+    cases[case_count++] = {context.phase, context.which, context.width, context.first,
+                          call_count.load(std::memory_order_acquire)};
+    return !failed.load(std::memory_order_acquire);
+}
+
+extern "C" bool zg_alpha_trace_save(const char *path) {
+    if (!path || active.load(std::memory_order_acquire) || failed.load(std::memory_order_acquire)
+        || case_count != MAX_CASES || call_count.load(std::memory_order_acquire) > MAX_CALLS)
+        return false;
+    std::ostringstream out;
+    out << "{\"scope\":\"exact exported llamafile_sgemm entry passthrough; opaque params\","
+        << "\"internal_microkernel_observed\":false,\"math_replaced\":false,\"cpu\":\"" << CPU
+        << "\",\"symbol_offset\":1100016,\"case_count\":" << case_count
+        << ",\"call_count\":" << call_count.load(std::memory_order_acquire) << ",\"cases\":[";
+    for (uint32_t i = 0; i < case_count; ++i) {
+        const auto &r = cases[i];
+        if (i) out << ',';
+        out << "{\"phase\":" << r.phase << ",\"which\":" << r.which << ",\"width\":" << r.width
+            << ",\"first\":" << r.first << ",\"last\":" << r.last << '}';
+    }
+    out << "],\"calls\":[";
+    for (uint32_t i = 0; i < call_count.load(std::memory_order_acquire); ++i) {
+        const auto &r = calls[i];
+        if (i) out << ',';
+        out << "{\"phase\":" << r.phase << ",\"which\":" << r.which << ",\"m\":" << r.m
+            << ",\"n\":" << r.n << ",\"k\":" << r.k << ",\"lda\":" << r.lda << ",\"ldb\":" << r.ldb
+            << ",\"ldc\":" << r.ldc << ",\"a_offset\":" << r.a << ",\"b_offset\":" << r.b
+            << ",\"c_offset\":" << r.c << ",\"atype\":" << r.atype << ",\"btype\":" << r.btype
+            << ",\"ctype\":" << r.ctype << ",\"accepted\":" << (r.accepted ? "true" : "false") << '}';
+    }
+    out << "]}\n";
+    const std::string bytes = out.str();
+    if (bytes.size() > 1024 * 1024) return false;
+    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
+    if (fd < 0) return false;
+    const bool ok = write_all(fd, bytes.data(), bytes.size());
+    return close(fd) == 0 && ok;
+}
--- /dev/null
+++ original2B-alpha-public-replay/observer.h
@@ -0,0 +1,14 @@
+// SPDX-License-Identifier: Apache-2.0
+#pragma once
+#include <stdbool.h>
+#include <stdint.h>
+#ifdef __cplusplus
+extern "C" {
+#endif
+bool zg_alpha_trace_begin(uint32_t phase, uint32_t which, uint32_t width,
+                       const void *a, const void *b, const void *c);
+bool zg_alpha_trace_end(void);
+bool zg_alpha_trace_save(const char *new_path);
+#ifdef __cplusplus
+}
+#endif
--- /dev/null
+++ original2B-alpha-public-replay/prepare.py
@@ -0,0 +1,205 @@
+#!/usr/bin/env python3
+"""ROOT ONLY: byte-copy authentic alpha operands/results, no operator/model calls."""
+import argparse
+from functools import cache
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import struct
+
+HERE = Path(__file__).resolve().parent
+CAPTURE = HERE.parent / 'numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport'
+MODEL = HERE.parents[2] / 'target/models/qwen35-2b/model.gguf'
+CPU = '/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so'
+W_OFFSET, W_BYTES, X_BYTES = 459622304, 131072, 8192
+FIXTURE_CAP = 4 * 1024 * 1024
+
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+
+def digest(path):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
+        before = os.fstat(stream.fileno())
+        require(stat.S_ISREG(before.st_mode), 'regular immutable source')
+        total, h = 0, hashlib.sha256()
+        while chunk := stream.read(1024 * 1024):
+            total += len(chunk)
+            h.update(chunk)
+        after = os.fstat(stream.fileno())
+        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
+                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
+                and total == before.st_size, 'source changed during hash')
+    return {'bytes': total, 'sha256': h.hexdigest()}
+
+
+def bounded(path, cap):
+    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= cap, 'bounded regular source')
+    raw = path.read_bytes()
+    require(len(raw) <= cap, 'bounded immutable-length read')
+    return raw
+
+
+def jread(path, cap=1024 * 1024):
+    return json.loads(bounded(path, cap), object_pairs_hook=unique)
+
+
+def authenticate():
+    pins = jread(HERE / 'source-pins.json')
+    require(pins['capture'] == str(CAPTURE) and pins['cpu_dso'] == CPU, 'closed original capture/backend')
+    for path, expected in pins['inputs'].items():
+        require(digest(path) == expected, 'source/content pin changed: ' + path)
+    return pins
+
+
+@cache
+def metadata(mode):
+    require(mode in ['bulk64', 'ar1'], 'closed actual mode')
+    return [json.loads(line, object_pairs_hook=unique) for line in
+            bounded(CAPTURE / mode / 'records.jsonl', 16 * 1024 * 1024).splitlines()]
+
+
+def actual(mode, output, call):
+    stage = 'projection_' + ('output' if output else 'input') + '::blk.0.ssm_alpha.weight'
+    rows = [row for row in metadata(mode) if row['call'] == call and row['layer'] == 0
+            and row['stage'] == stage and row['occurrence'] == 0]
+    require(len(rows) == 1, 'unique complete original alpha field')
+    row = rows[0]
+    width, inner = (64 if mode == 'bulk64' else 1), (16 if output else 2048)
+    shape, strides = [inner, width, 1, 1], [4, inner*4, inner*width*4, inner*width*4]
+    require(row['mode'] == mode and type(row['call']) is int and row['call'] == call
+            and row['call_tokens'] == width and row['window_start'] == 0 and row['window_tokens'] == 64
+            and row['token_axis'] == 1 and row['scope'] == 'actual_tensor'
+            and row['masked_negative_infinity_allowed'] is False
+            and row['source_slice_start_elements'] == 0, 'actual unchanged full64 versus AR1 call')
+    require(row['dense_ne'] == row['native_ne'] == shape and row['native_nb'] == strides
+            and row['native_op'] == ('MUL_MAT' if output else 'MUL')
+            and row['elements'] == inner * width and '/' not in row['file'], 'full original F32 alpha geometry')
+    raw = bounded(CAPTURE / mode / row['file'], 524288)
+    require(len(raw) == inner * width * 4 and all(word & 0x7f800000 != 0x7f800000
+            for (word,) in struct.iter_unpack('<I', raw)), 'complete finite actual alpha words')
+    return row, raw
+
+
+def actual_weight():
+    rows = [json.loads(line, object_pairs_hook=unique) for line in
+            bounded(CAPTURE / 'learned-operands/records.jsonl', 128*1024).splitlines()]
+    selected = [row for row in rows if row['name'] == 'blk.0.ssm_alpha.weight']
+    require(len(rows) == 28 and len(selected) == 1, 'original learned role closure')
+    row = selected[0]
+    require(row['type'] == 0 and row['type_name'] == 'f32' and row['ne'] == [2048,16,1,1]
+            and row['nb'] == [4,8192,131072,131072] and row['bytes'] == W_BYTES
+            and row['encoding'] == 'GGUF_RAW' and row['buffer_name'] == 'CPU_Mapped'
+            and row['chunks'] == [{'file':'learned-4-0.bin','offset':0,'bytes':W_BYTES}],
+            'actual ordinary unmodified full-width F32 alpha weight; no repack')
+    raw = bounded(CAPTURE / 'learned-operands/learned-4-0.bin', W_BYTES)
+    require(len(raw) == W_BYTES, 'all actual learned weight bytes')
+    with os.fdopen(os.open(MODEL, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
+        source.seek(W_OFFSET)
+        require(source.read(W_BYTES) == raw, 'all physical weight bits equal approved GGUF source interval')
+    require(all(word & 0x7f800000 != 0x7f800000 for (word,) in struct.iter_unpack('<I', raw)), 'finite original F32 weight')
+    return row, raw
+
+
+def capture_guards():
+    done = jread(CAPTURE / 'complete.json')
+    require(done['status'] == 'TRANSPARENT_WINDOW_CAPTURE_NOT_CAUSE_QUALIFICATION'
+            and all(done[key] is True for key in ['common_prior_exact','bulk_instrumentation_complete_state_exact',
+                'ar1_instrumentation_complete_state_exact','both_next_AR1_full248320_and_whole_state_exact',
+                'all28_owners_seen_bulk_and_all64_AR1_calls','learned_unchanged_after_compute'])
+            and done['window_start'] == 0 and done['window_tokens'] == 64 and done['threads'] == 8
+            and done['learned_verified_boundaries'] == 2, 'full actual own-context transparency/readset guards')
+    bulk_row, bulk = actual('bulk64', False, 0)
+    producer = [row for row in metadata('bulk64') if row['stage'] == 'learned_op_output::blk.0.attn_norm.weight'
+                and row['layer'] == 0 and row['occurrence'] == 0]
+    output, _ = actual('bulk64', True, 0)
+    require(len(producer) == 1 and producer[0]['native_tensor_id'] == bulk_row['native_tensor_id']
+            == output['native_source_ids'][1] and producer[0]['native_op'] == 'MUL'
+            and bounded(CAPTURE/'bulk64'/producer[0]['file'],524288) == bulk, 'actual exact input producer binding')
+    for at in range(64):
+        row, raw = actual('ar1', False, at)
+        out, _ = actual('ar1', True, at)
+        require(raw == bulk[at*X_BYTES:(at+1)*X_BYTES] and row['native_tensor_id'] == out['native_source_ids'][1],
+                'all131072 actual input words and AR1 projection producer bindings close')
+
+
+def verify_fixtures(folder, pins):
+    manifest = jread(folder / 'manifest.json')
+    require(manifest['status'] == 'AUTHENTIC_BYTE_ONLY_ALPHA_FIXTURES' and len(manifest['cases']) == 65
+            and manifest['source_pins'] == digest(HERE/'source-pins.json')
+            and manifest['weight']['captured_row'] == actual_weight()[0], 'source-bound full alpha fixture manifest')
+    _, weight = actual_weight()
+    require(bounded(folder/'weight.f32le', W_BYTES) == weight
+            and digest(folder/'weight.f32le') == manifest['weight']['digest'], 'exact shared original weight fixture')
+    total = len(weight)
+    for which, descriptor in enumerate(manifest['cases']):
+        mode, at = ('bulk64', 0) if which == 0 else ('ar1', which-1)
+        key = 'bulk64' if which == 0 else f'ar1-{at}'
+        require(descriptor['case'] == key and descriptor['mode'] == mode and descriptor['call'] == at
+                and set(descriptor['fields']) == {'x','expected'}, 'same original65call population/order')
+        for suffix, output in [('x',False),('expected',True)]:
+            row, raw = actual(mode,output,at)
+            part = descriptor['fields'][suffix]
+            require(part['file'] == f'{key}.{suffix}.f32le' and part['captured_row'] == row
+                    and digest(folder/part['file']) == part['digest']
+                    and bounded(folder/part['file'],524288) == raw, 'authentic full actual fixture bytes')
+            total += len(raw)
+    require(total == manifest['payload_bytes'] == 1187840 and total <= FIXTURE_CAP, 'complete bounded fixture payload')
+    require(pins['cpu_dso'] == CPU, 'same public original backend fixture scope')
+    return digest(folder/'manifest.json')
+
+
+def prepare(output):
+    require(output.resolve().is_relative_to(HERE), 'owned exclusive fixture directory')
+    pins = authenticate()
+    capture_guards()
+    output.mkdir(mode=0o700)
+    row, weight = actual_weight()
+    with (output/'weight.f32le').open('xb') as stream:
+        stream.write(weight)
+    (output/'weight.f32le').chmod(0o600)
+    entries, total = [], len(weight)
+    for which in range(65):
+        mode, at = ('bulk64',0) if which == 0 else ('ar1',which-1)
+        key = 'bulk64' if which == 0 else f'ar1-{at}'
+        fields = {}
+        for suffix, output_field in [('x',False),('expected',True)]:
+            captured, raw = actual(mode,output_field,at)
+            target = output/f'{key}.{suffix}.f32le'
+            total += len(raw)
+            require(total <= FIXTURE_CAP, 'bounded raw fixture payload')
+            with target.open('xb') as stream:
+                stream.write(raw)
+            target.chmod(0o600)
+            fields[suffix] = {'file':target.name,'captured_row':captured,'digest':digest(target)}
+        entries.append({'case':key,'mode':mode,'call':at,'fields':fields})
+    manifest = {'status':'AUTHENTIC_BYTE_ONLY_ALPHA_FIXTURES','source_pins':digest(HERE/'source-pins.json'),
+                'weight':{'file':'weight.f32le','captured_row':row,'digest':digest(output/'weight.f32le')},
+                'cases':entries,'payload_bytes':total,'arithmetic_executed':False,
+                'expected_values':'unchanged complete actual original public alpha output bits',
+                'source_range':{'model':str(MODEL),'start':W_OFFSET,'bytes':W_BYTES,'encoding':'F32_GGUF_RAW'}}
+    raw = (json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
+    require(len(raw) <= 1024*1024, 'bounded fixture metadata')
+    with (output/'manifest.json').open('xb') as stream:
+        stream.write(raw)
+    authenticate()
+    verify_fixtures(output,pins)
+    print('AUTHENTIC_BYTE_ONLY_ALPHA_FIXTURES',len(entries),total)
+
+
+if __name__ == '__main__':
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('new_fixture_directory',type=Path)
+    prepare(parser.parse_args().new_fixture_directory)
--- /dev/null
+++ original2B-alpha-public-replay/run_replay.py
@@ -0,0 +1,196 @@
+#!/usr/bin/env python3
+"""ROOT ONLY: finite owned public-operator process, no model/provider server."""
+import argparse
+import json
+import os
+from pathlib import Path
+import selectors
+import signal
+import subprocess
+import time
+from prepare import HERE, CPU, authenticate, verify_fixtures, digest, jread, require, capture_guards
+
+LOADER = '/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib/ld-linux-x86-64.so.2'
+LIB = '/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib'
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
+
+
+def main():
+    p = argparse.ArgumentParser(description=__doc__)
+    p.add_argument('--binary', type=Path, required=True)
+    p.add_argument('--binary-sha256', required=True)
+    p.add_argument('--fixtures', type=Path, required=True)
+    p.add_argument('--new-output', type=Path, required=True)
+    p.add_argument('--new-log', type=Path, required=True)
+    p.add_argument('--new-receipt', type=Path, required=True)
+    p.add_argument('--preload', type=Path)
+    p.add_argument('--preload-sha256')
+    a = p.parse_args()
+    pins = authenticate()
+    capture_guards()
+    for path in [a.new_output, a.new_log, a.new_receipt]:
+        require(path.resolve().is_relative_to(HERE), 'owned ignored-run output path')
+    require(a.fixtures.resolve() == HERE / 'root-fixtures', 'exact accepted original fixtures')
+    before_fixtures = verify_fixtures(a.fixtures, pins)
+    before_binary = digest(a.binary)
+    require((a.preload is None) == (a.preload_sha256 is None), 'preload path and SHA paired')
+    before_preload = None
+    if a.preload is not None:
+        require(a.preload.resolve().is_relative_to(HERE), 'owned observer DSO only')
+        before_preload = digest(a.preload)
+        require(before_preload['bytes'] <= 8 * 1024 * 1024
+                and before_preload['sha256'] == a.preload_sha256, 'bounded pinned observer DSO')
+    require(before_binary['bytes'] <= 16*1024*1024 and before_binary['sha256'] == a.binary_sha256
+            and not a.new_output.exists(), 'bounded pinned newly built binary/new output')
+    require(a.new_log.parent.resolve() != a.new_output.resolve()
+            and a.new_receipt.parent.resolve() != a.new_output.resolve(), 'runner evidence outside native output')
+    argv = [LOADER, '--library-path', LIB, str(a.binary.resolve()), str(a.fixtures.resolve()),
+            str(a.new_output.resolve()), CPU, '8', '300']
+    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'CUDA_VISIBLE_DEVICES': '-1',
+           'NVIDIA_VISIBLE_DEVICES': 'void', 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
+    if a.preload is not None:
+        env['LD_PRELOAD'] = str(a.preload.resolve())
+        env['ZG_ALPHA_TRACE'] = '1'
+    started = time.monotonic()
+    receipt = {'argv': argv, 'binary': before_binary, 'fixture_manifest': before_fixtures,
+               'wall_cap_seconds': 300, 'address_space_cap_bytes': 8 * 1024**3,
+               'actual_gpu_or_model_execution': False, 'internal_dispatch_observed': False,
+               'preload': str(a.preload.resolve()) if a.preload else None, 'preload_digest': before_preload,
+               'source_profile': 'pinned llama.cpp0.6.0 haswell public alpha default/control diagnostic',
+               'exit_code': None, 'status': 'UNQUALIFIED'}
+    with a.new_receipt.open('x') as receipt_file, a.new_log.open('xb') as log:
+        process = None
+        selector = None
+        log_bytes = 0
+        primary_error = None
+        cleanup_error = None
+        try:
+            process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                                       stderr=subprocess.STDOUT, start_new_session=True)
+            receipt['owned_pid'] = process.pid
+            selector = selectors.DefaultSelector()
+            selector.register(process.stdout, selectors.EVENT_READ)
+            while selector.get_map():
+                require(time.monotonic() - started < 300, 'finite wall deadline')
+                for key, _ in selector.select(0.2):
+                    raw = os.read(key.fileobj.fileno(), 65536)
+                    if not raw:
+                        selector.unregister(key.fileobj)
+                        continue
+                    log_bytes += len(raw)
+                    require(log_bytes <= 8 * 1024 * 1024, 'bounded child log')
+                    log.write(raw)
+            receipt['exit_code'] = leader_exit_without_reaping(process, 5)
+            require(receipt['exit_code'] == 0, 'public alpha process refused')
+            require(digest(a.binary) == before_binary, 'binary changed during run')
+            require(before_preload is None or digest(a.preload) == before_preload, 'observer changed during run')
+            authenticate()
+            require(verify_fixtures(a.fixtures, pins) == before_fixtures, 'fixtures changed during run')
+            capture_guards()
+            receipt['status'] = 'PUBLIC_ALPHA_OBSERVER_TRANSPARENCY_VALIDATOR_REQUIRED'
+        except BaseException as error:
+            receipt['error'] = f'{type(error).__name__}: {error}'
+            primary_error = error
+        finally:
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
+                    try:
+                        close()
+                        receipt.setdefault('cleanup', {})[label] = True
+                    except BaseException as error:
+                        cleanup_error = cleanup_error or error
+                        receipt.setdefault('cleanup_errors', []).append(f'{label}: {type(error).__name__}: {error}')
+                        receipt['status'] = 'UNQUALIFIED'
+                if process is not None:
+                    receipt['exit_code'] = process.returncode
+            receipt['elapsed_seconds'] = time.monotonic() - started
+            json.dump(receipt, receipt_file, indent=2, allow_nan=False)
+            receipt_file.write('\n')
+            receipt_file.flush()
+        if primary_error is not None:
+            raise primary_error
+        if cleanup_error is not None:
+            raise cleanup_error
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ original2B-alpha-public-replay/validate.py
@@ -0,0 +1,164 @@
+#!/usr/bin/env python3
+"""ROOT ONLY: all default/output transparency before exported alpha route evidence."""
+import argparse
+import hashlib
+import json
+from pathlib import Path
+import struct
+from prepare import HERE, CPU, authenticate, actual, verify_fixtures, bounded, digest, jread, require, capture_guards
+
+
+def integer(value):
+    return type(value) is int
+
+
+def words_diff(left, right):
+    require(len(left) == len(right) and len(left) % 4 == 0, 'complete raw word comparison')
+    return sum(a != b for a, b in zip(struct.iter_unpack('<I', left), struct.iter_unpack('<I', right)))
+
+
+def run_receipt(path, output, observed):
+    record = jread(path)
+    require(record['status'] == 'PUBLIC_ALPHA_OBSERVER_TRANSPARENCY_VALIDATOR_REQUIRED'
+            and integer(record['exit_code']) and record['exit_code'] == 0
+            and record['actual_gpu_or_model_execution'] is False and record['internal_dispatch_observed'] is False,
+            'complete unchanged public alpha replay')
+    require(record['argv'][5] == str(output.resolve()) and record['argv'][-3:] == [CPU,'8','300'], 'exact owned original invocation')
+    require(all(record['cleanup'].get(key) is True for key in
+                ['leader_reaped','group_absent','pipe_eof','selector_closed','pipe_closed']), 'complete finite owned PGID/pipe cleanup')
+    require(record['wall_cap_seconds'] == 300 and record['address_space_cap_bytes'] == 8*1024**3,
+            'same bounded resource policy')
+    require(digest(record['argv'][3]) == record['binary'], 'same pinned public replay executable')
+    if observed:
+        require(type(record['preload']) is str and Path(record['preload']).resolve().is_relative_to(HERE)
+                and digest(record['preload']) == record['preload_digest'], 'actual own pinned observer')
+    else:
+        require(record['preload'] is None and record['preload_digest'] is None, 'genuine unobserved baseline')
+    return record
+
+
+def outputs(folder, observer):
+    require(folder.resolve().is_relative_to(HERE), 'owned new output directory')
+    done = jread(folder/'complete.json')
+    require(done['status'] == 'PUBLIC_ALPHA_REPLAY_DEFAULT_EXACT_CONTROL_DIAGNOSTIC'
+            and done['default_cases'] == done['control_cases'] == 65 and done['default_words'] == 2048
+            and integer(done['threads']) and done['threads'] == 8 and done['cause_qualified'] is False
+            and done['production_math_changed'] is False and done['internal_dispatch_observed'] is False,
+            'complete public65 default/65 control alpha output gate')
+    cases = [json.loads(line) for line in bounded(folder/'cases.jsonl',128*1024).splitlines()]
+    require(len(cases) == 130, 'complete source case order/count')
+    content, descriptors, defaults, total = {}, {}, {}, 0
+    for index,row in enumerate(cases):
+        which = index % 65
+        key = 'bulk64' if which == 0 else f'ar1-{which-1}'
+        control = index >= 65
+        file = ('ref-control-' if control else 'default-') + key + '.actual.f32le'
+        count = 1024 if which == 0 else 16
+        require(row['case'] == key and row['mode'] == ('PUBLIC_USE_REF_CONTROL' if control else 'DEFAULT_PUBLIC_ALPHA')
+                and integer(row['words']) and row['words'] == count, 'all complete raw alpha words/cases')
+        raw = bounded(folder/file,4096)
+        require(len(raw) == count*4 and all(word & 0x7f800000 != 0x7f800000
+                for (word,) in struct.iter_unpack('<I',raw)), 'exact full finite original operator output')
+        _, expected = actual('bulk64' if which == 0 else 'ar1',True,0 if which == 0 else which-1)
+        different = words_diff(raw,expected)
+        require(type(row['actual_capture_unequal_words']) is int and row['actual_capture_unequal_words'] == different,
+                'independently checked full capture comparison')
+        if not control:
+            require(different == 0, 'all65 default public projections exact to original capture')
+            defaults[which] = raw
+        content[file] = raw
+        descriptors[file] = {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
+        total += len(raw)
+    require(total == 16384 and len(defaults) == 65, 'all4096 alpha output words per run')
+    ref_bulk = content['ref-control-bulk64.actual.f32le']
+    ref_vs_ar = sum(words_diff(ref_bulk[at*64:(at+1)*64],defaults[at+1]) for at in range(64))
+    ref_ar_diff = sum(words_diff(content[f'ref-control-ar1-{at}.actual.f32le'],defaults[at+1]) for at in range(64))
+    require(cases[65]['ref_bulk_vs_all_default_ar1_unequal_words'] == ref_vs_ar,
+            'independently checked complete refbulk versus all originalAR1 comparison')
+    for phase in ['before','after']:
+        maps = bounded(folder/f'{phase}.maps',1024*1024).decode()
+        require(CPU in maps and all(CPU in line for line in maps.splitlines() if 'libggml-cpu-' in line),
+                'actual pinned Haswell backend only')
+        if observer:
+            require(observer in maps, 'actual observer mapped around computation')
+    return descriptors, content, {'ref_bulk_vs_all_default_AR1_unequal_words':ref_vs_ar,
+        'all_ref_AR1_vs_default_AR1_unequal_words':ref_ar_diff,'all65_defaults_exact':True}
+
+
+def routes(path):
+    trace = jread(path)
+    require(trace['cpu'] == CPU and trace['symbol_offset'] == 0x10c8f0
+            and trace['internal_microkernel_observed'] is False and trace['math_replaced'] is False,
+            'exact pinned exported entry, opaque params/no internal-kernel inference')
+    require(integer(trace['case_count']) and trace['case_count'] == 130 and len(trace['cases']) == 130
+            and integer(trace['call_count']) and trace['call_count'] == 520 and len(trace['calls']) == 520,
+            'complete bounded original public entry trace')
+    at = 0
+    for index,case in enumerate(trace['cases']):
+        phase,which = divmod(index,65)
+        width,count = (64 if which == 0 else 1),(0 if phase else 8)
+        require(all(integer(case[key]) for key in ['phase','which','width','first','last'])
+                and case == {'phase':phase,'which':which,'width':width,'first':at,'last':at+count},
+                'exact original public alpha route population/order')
+        for row in trace['calls'][at:at+count]:
+            require(all(integer(row[key]) for key in ['phase','which','m','n','k','lda','ldb','ldc',
+                    'a_offset','b_offset','c_offset','atype','btype','ctype']), 'exact integer entry descriptor')
+            require(row['phase'] == phase and row['which'] == which
+                    and (row['m'],row['n'],row['k']) == (16,width,2048)
+                    and (row['lda'],row['ldb'],row['ldc']) == (2048,2048,16)
+                    and (row['atype'],row['btype'],row['ctype']) == (0,0,0)
+                    and (row['a_offset'],row['b_offset'],row['c_offset']) == (0,0,0)
+                    and type(row['accepted']) is bool and row['accepted'] is (width == 64),
+                    'full original F32 alpha pointers/geometry and untouched public return')
+        at += count
+    require(at == 520, 'no omitted/trailing public call')
+    return {'default_bulk64':'8 exported calls, all true',
+            'default_AR1':'64x8 exported calls, all false',
+            'public_use_ref':'65 cases, zero exported calls','exported_calls':at}
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--plain-output',type=Path,required=True)
+    parser.add_argument('--plain-receipt',type=Path,required=True)
+    parser.add_argument('--observed-output',type=Path,required=True)
+    parser.add_argument('--observed-receipt',type=Path,required=True)
+    parser.add_argument('--new-report',type=Path,required=True)
+    args = parser.parse_args()
+    pins = authenticate()
+    capture_guards()
+    fixtures = verify_fixtures(HERE/'root-fixtures',pins)
+    plain = run_receipt(args.plain_receipt,args.plain_output,False)
+    observed = run_receipt(args.observed_receipt,args.observed_output,True)
+    require(plain['binary'] == observed['binary'] and plain['fixture_manifest'] == observed['fixture_manifest'] == fixtures
+            and plain['argv'] == observed['argv'][:5]+[str(args.plain_output.resolve())]+observed['argv'][6:],
+            'same exact original operator inputs/binary/conditions')
+    require(not (args.plain_output/'sgemm-trace.json').exists(), 'plain unobserved execution')
+    a,raw_a,summary = outputs(args.plain_output,None)
+    b,raw_b,observed_summary = outputs(args.observed_output,observed['preload'])
+    require(a == b and raw_a == raw_b and summary == observed_summary,
+            'all130 full plain/observed default AND control outputs bit-identical before route acceptance')
+    route = routes(args.observed_output/'sgemm-trace.json')
+    total = 0
+    for path in HERE.rglob('*'):
+        require(not path.is_symlink() and (path.is_file() or path.is_dir()),'regular own diagnostic artifact tree')
+        if path.is_file():
+            total += path.stat().st_size
+    require(total <= 512*1024*1024-128*1024,'bounded complete new-run artifacts with report reserve')
+    authenticate()
+    report = {'status':'TRANSPARENT_PUBLIC_ALPHA_EXPORTED_ENTRY_ROUTE_OBSERVED',
+        'all65_defaults_exact_to_actual_captures':True,'all130_plain_observed_outputs_exact':True,
+        'words_each_run':4096,'routes':route,'controls':summary,'outputs':a,
+        'input_closure':'same actual131072 F32 input words and full32768-word raw originalF32 alpha weights',
+        'source_pins':digest(HERE/'source-pins.json'),'binary':plain['binary'],'observer':observed['preload_digest'],
+        'pre_report_artifact_bytes':total,'internal_microkernel_observed':False,'full_runtime_closure_qualified':False,
+        'long_context_cause_qualified':False,'production_math_changed':False,'model_or_profile_eligibility':False}
+    raw = (json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
+    require(len(raw) <= 128*1024 and args.new_report.resolve().is_relative_to(HERE),'bounded new owned report')
+    with args.new_report.open('xb') as stream:
+        stream.write(raw)
+    print(report['status'],str(args.new_report))
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ original2B-alpha-public-replay/source-pins.json
@@ -0,0 +1,695 @@
+{
+  "scope": "SOURCE_ONLY_PUBLIC_ORIGINAL2B_ALPHA_REPLAY_OBSERVER_PREPARATION",
+  "capture": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport",
+  "original_source": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0",
+  "cpu_dso": "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so",
+  "public_symbol": "llamafile_sgemm",
+  "public_symbol_offset": 1100016,
+  "static_elf_evidence": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/source-pins.json",
+  "model_hash_provenance": "Previously independently approved exact GGUF; author only read bounded131072-byte alpha source interval; ROOT authenticate hashes whole model before/after",
+  "inputs": {
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/complete.json": {
+      "bytes": 1181,
+      "sha256": "0b73666812a6dd9a99558562c0460ab1810e05399f645fed96aa70d37725e1f3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/bulk64/records.jsonl": {
+      "bytes": 55822,
+      "sha256": "eacee1e7cf45fdac3ea40e48152432347f1f8f3915c633ecc2ce4c8f59e54437"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/records.jsonl": {
+      "bytes": 3378646,
+      "sha256": "114bbbf9464e288b225ece7e63f73ba07b9ffeae42343e45cc0b5d1b4fac2ed3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/learned-operands/records.jsonl": {
+      "bytes": 17570,
+      "sha256": "26d8a9d96194a38d6125ab339abc7564229cd98e65dc6ef7bee1751fa7bb7590"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/learned-operands/learned-4-0.bin": {
+      "bytes": 131072,
+      "sha256": "0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport-comparison.json": {
+      "bytes": 4809755,
+      "sha256": "0fd6266cb8437d7e1f1113944428a58fd31951bd44d2409d80ea9c3f0a638bae"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/learned-source-headers.json": {
+      "bytes": 6898,
+      "sha256": "5a0e2c8ee280d08c06252b25ddcc38fab300ccbaf56186104b3564ca864b9615"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-repack-transport-followup/native_window.cpp": {
+      "bytes": 33599,
+      "sha256": "43e3da3ceb6bd482e4e8419bbdc257687b7021720c0afcb35f8701ef784db614"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-repack-transport-followup/compare_window.py": {
+      "bytes": 24185,
+      "sha256": "1bc9fe24c2500017b339678884ad4a4113fc6271f5a315391b2f847868ba9b58"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/main.cpp": {
+      "bytes": 15528,
+      "sha256": "f14c8831a178bb89ee6e854fd84539d239d8605526acc14584b08b4e24434b94"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/observer.cpp": {
+      "bytes": 6793,
+      "sha256": "04ceca23c9c816fda083c27a748599b0c489baa31129559dba5689a8ffaec786"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/observer.h": {
+      "bytes": 373,
+      "sha256": "249cadd15ed550f6113edf48f457276850598871a84c5c01d518c6902e57a880"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/run_replay.py": {
+      "bytes": 9225,
+      "sha256": "c226112446b7d90f60363f9a08291680e467be49e171f146e663d31891a44b7c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/root-validation.json": {
+      "bytes": 1043,
+      "sha256": "aba581431ecbb32bc9d0d8f7cc85d1edded11da218fc81933e8ee79dc1def3d9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-dispatch-20261020/source-pins.json": {
+      "bytes": 3993,
+      "sha256": "51ffc0976cf38182e3fe0f355c2f820723566e6e1526e1726c2303408e08e82a"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/bulk64/field-15.f32le": {
+      "bytes": 524288,
+      "sha256": "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/bulk64/field-16.f32le": {
+      "bytes": 4096,
+      "sha256": "448f5626f2f06de0e5fe0f375a92589bd87b81833fcbb9daca530c20f8424b3e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-15.f32le": {
+      "bytes": 8192,
+      "sha256": "b1d1ac012f78247da9d59c268de262cb88d569fcc5f28df8e580ab3a1e837208"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-16.f32le": {
+      "bytes": 64,
+      "sha256": "c5966078b46b065ea9b7505b36e7fdfd4f769350f47ac4b52a8e9c3ddcc5f90c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-126.f32le": {
+      "bytes": 8192,
+      "sha256": "d8c1e375e97820914cd5fc4709879f042e7406a0a303f1fa2d6e2e13669d50a2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-127.f32le": {
+      "bytes": 64,
+      "sha256": "ad7c50ccc5731d098bffbe7b8c49b4740226037f074ce36a127ab401057f2b13"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-235.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-236.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-344.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-345.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-453.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-454.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-562.f32le": {
+      "bytes": 8192,
+      "sha256": "d6394c00fd7f2c1566e93699e79c0aa400568310aea2abaec4e31a6f478e7a8f"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-563.f32le": {
+      "bytes": 64,
+      "sha256": "532e7275e293b0dbc749eecd9808ce05f6ea5e1e6bf114adfe31bf2f6efcfc8f"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-671.f32le": {
+      "bytes": 8192,
+      "sha256": "4debe33280e810932c53160b23d82a923a004f84f59b543c5517eb9e3c310461"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-672.f32le": {
+      "bytes": 64,
+      "sha256": "e9d014c55d2dbb556818e78fd486606b75550c0bb12b018c2270f1186bd66797"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-780.f32le": {
+      "bytes": 8192,
+      "sha256": "e5f627101f0b7ea40d8d7d262a28916add245737fb551403e0e7240c214f0e28"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-781.f32le": {
+      "bytes": 64,
+      "sha256": "9fdf1df6a995d85b81c6a8182bc9a5d5af401603e7113f078b15d8cbdafb6a91"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-889.f32le": {
+      "bytes": 8192,
+      "sha256": "dd96115805b5424ba80832d1d0f98a53a1fa8762cdcfc1152b1d965d900b5e2e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-890.f32le": {
+      "bytes": 64,
+      "sha256": "52a936318dbe26ab7df850f47716b16b8da5f7423700f81929aca0c0ad82d6ce"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-998.f32le": {
+      "bytes": 8192,
+      "sha256": "04b06b0330f2d25de147a54c0685a030ce18d35f5d2ebba0262f1f2022bbf000"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-999.f32le": {
+      "bytes": 64,
+      "sha256": "f579ac32b7007efa37a7aa5da70f20896f865a7807afbd343216c91296fc70d1"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1107.f32le": {
+      "bytes": 8192,
+      "sha256": "179941df556fb79370c6190ee4d528c530935f9c4397a1cfca82c15d2de49ca5"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1108.f32le": {
+      "bytes": 64,
+      "sha256": "bb4e5a53a864a9aca9c369d7e84982f03d6cb4aacd3e83a51b7f39d6ce4d1a2c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1216.f32le": {
+      "bytes": 8192,
+      "sha256": "3d1d394ac0dd107357192ac9d000202fe01acfce20e6380518cf84fc103f8394"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1217.f32le": {
+      "bytes": 64,
+      "sha256": "c588cd2252500b74b9e93d981f4f5b4d928d29c8786db6e590882181febde8bc"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1325.f32le": {
+      "bytes": 8192,
+      "sha256": "86f6f7a22e288e49ad29ffc2e7635a169d281beb4333768fe88bca49c6663a6f"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1326.f32le": {
+      "bytes": 64,
+      "sha256": "18c0fbe8a620e8ac25b7015191a6135198ae128a9e654154c0be50a6d6702c1b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1434.f32le": {
+      "bytes": 8192,
+      "sha256": "c0adc6be7d6c5a74a0c398ea033b45bff455eaa4023e79f2700cf4b80f7c8226"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1435.f32le": {
+      "bytes": 64,
+      "sha256": "06c4fa2f9601385da9c2067c40473c705456b2d09abf4a686fd233d98169f1b4"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1543.f32le": {
+      "bytes": 8192,
+      "sha256": "5beb5350ecfda3a3ee1f4373ae4963fa1fe8446640da40417864b6e7cd45182b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1544.f32le": {
+      "bytes": 64,
+      "sha256": "f335de41562407dae49acfef40b894b8f589e46cb7b97496c79b3fe27ae92bd4"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1652.f32le": {
+      "bytes": 8192,
+      "sha256": "790c85507bc57a6a4953ace8d0ee163dd3c247b6b42575e6177ebf5551b76792"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1653.f32le": {
+      "bytes": 64,
+      "sha256": "263e18418766b9f3eb33b46bbf335fdb5bb161f839af61c38c7cb660f13b943b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1761.f32le": {
+      "bytes": 8192,
+      "sha256": "952eac836d2bbfc0889d5c0a4b7ed7f1d6fe82b00bd2bd1c7f32c3f6f577331e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1762.f32le": {
+      "bytes": 64,
+      "sha256": "44969c46575a1f474c8b427c6f5e17f24a0a854736a5c179bc66e76c68f405b7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1870.f32le": {
+      "bytes": 8192,
+      "sha256": "db1bc46461d226e632d5f90a896063fc962d5677b9ae841fabcc72a03574afac"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1871.f32le": {
+      "bytes": 64,
+      "sha256": "0c0eefc34cb33fc8ee34da8517e17796a2c98f91dbede5d1d06a0e940e4ed982"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1979.f32le": {
+      "bytes": 8192,
+      "sha256": "c81baccc0b256e13db4edd567f72095378e85b187924839fd4b5bce68fe870cd"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-1980.f32le": {
+      "bytes": 64,
+      "sha256": "96a67aa232ec9c1843b81bb0fdad48d60b5020f76040a7b1cbeb1aa02720ec4e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2088.f32le": {
+      "bytes": 8192,
+      "sha256": "93b61dc38c2d6cefc5d4a9c95651b2f98f45561512ee427bc5bba6952c87eacd"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2089.f32le": {
+      "bytes": 64,
+      "sha256": "3b12967b82c0c96808c1bf16ff2ebd82175dd9b0b74d5baaf4fccd7591d6eb31"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2197.f32le": {
+      "bytes": 8192,
+      "sha256": "06b6a0540a93872575ebe326330b19a0d881d0a94025fab9037f244843b8d30c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2198.f32le": {
+      "bytes": 64,
+      "sha256": "3709eadf0cc78f3a38f6f68ae2048afc96f21b0ad576d22a2902e55690d13c61"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2306.f32le": {
+      "bytes": 8192,
+      "sha256": "88a090dac7c622a215631a1920ad654760fa46e23f6149d0e89b675167fa7963"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2307.f32le": {
+      "bytes": 64,
+      "sha256": "76dc55427c84b5798eea8d9a4b8e892f4f0a3029a884277d461db0d993c0071a"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2415.f32le": {
+      "bytes": 8192,
+      "sha256": "655cc2c29feb1a794bc7aaa2ac93103104d717911e99f45eba93741de3e9fe02"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2416.f32le": {
+      "bytes": 64,
+      "sha256": "6541bf946984c14ae2fcec20e9a74c495dddf0146789b490889aca921f3ef84c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2524.f32le": {
+      "bytes": 8192,
+      "sha256": "88a090dac7c622a215631a1920ad654760fa46e23f6149d0e89b675167fa7963"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2525.f32le": {
+      "bytes": 64,
+      "sha256": "76dc55427c84b5798eea8d9a4b8e892f4f0a3029a884277d461db0d993c0071a"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2633.f32le": {
+      "bytes": 8192,
+      "sha256": "df57eba0c74a8f626e55ff0174bfdd5796c230a7ada1dc043ab89aeec30d962d"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2634.f32le": {
+      "bytes": 64,
+      "sha256": "401f04f19c9a802e2bb31e64d240550da1818d82159d2d7d7a69b0dd17217b78"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2742.f32le": {
+      "bytes": 8192,
+      "sha256": "5a748c996f44af066bfb0201952b91c81cc2a82a549f87b728e4461d6923e3fe"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2743.f32le": {
+      "bytes": 64,
+      "sha256": "4660e1b573834c90873514f00db01c119caa2bb7e8f6f60e4d4833820d1cad19"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2851.f32le": {
+      "bytes": 8192,
+      "sha256": "5bb0f960655d306707c9396463d78a209f4cf4eb18282cfa1782117eb44785c7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2852.f32le": {
+      "bytes": 64,
+      "sha256": "bac1a44db7be34bca0f66c7bdb29a57382563fa345d497e3300eaa188efefb6e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2960.f32le": {
+      "bytes": 8192,
+      "sha256": "07fc9735c3c73f304becc2501cacb24baf981795b3dd7732970384e2a4cc301e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-2961.f32le": {
+      "bytes": 64,
+      "sha256": "3108d244ef07bb9d3b6c2587a1c093bccb349635dfb94f7628b7734d786a5708"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3069.f32le": {
+      "bytes": 8192,
+      "sha256": "d8c1e375e97820914cd5fc4709879f042e7406a0a303f1fa2d6e2e13669d50a2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3070.f32le": {
+      "bytes": 64,
+      "sha256": "ad7c50ccc5731d098bffbe7b8c49b4740226037f074ce36a127ab401057f2b13"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3178.f32le": {
+      "bytes": 8192,
+      "sha256": "a60820c0bdaab2d062f82213fd23f2db10ad03b8748634fec024d6a3a4a5a5d5"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3179.f32le": {
+      "bytes": 64,
+      "sha256": "f05a4871f61a0826a7e9dd3cf472d23b165fd18fc42c380c9d43a289d178fe04"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3287.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3288.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3396.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3397.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3505.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3506.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3614.f32le": {
+      "bytes": 8192,
+      "sha256": "952eac836d2bbfc0889d5c0a4b7ed7f1d6fe82b00bd2bd1c7f32c3f6f577331e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3615.f32le": {
+      "bytes": 64,
+      "sha256": "44969c46575a1f474c8b427c6f5e17f24a0a854736a5c179bc66e76c68f405b7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3723.f32le": {
+      "bytes": 8192,
+      "sha256": "4debe33280e810932c53160b23d82a923a004f84f59b543c5517eb9e3c310461"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3724.f32le": {
+      "bytes": 64,
+      "sha256": "e9d014c55d2dbb556818e78fd486606b75550c0bb12b018c2270f1186bd66797"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3832.f32le": {
+      "bytes": 8192,
+      "sha256": "30279f1aff15ed5348dd29912b375e0bcc1763066a99e863233c08b959e1af13"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3833.f32le": {
+      "bytes": 64,
+      "sha256": "4bb0900b75f3a143e4b491045ceb67d3507bd2bb0bab69822689b94b910717b1"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3941.f32le": {
+      "bytes": 8192,
+      "sha256": "6c93b707dbf7ac24a57a67fbb1976703083263d05cd4ff5ec3f8b0ecf2a3177b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-3942.f32le": {
+      "bytes": 64,
+      "sha256": "f747628720fb5b6da6303c210988d76ecbc08e7d17e2e7397e8a39e0a8c63b73"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4050.f32le": {
+      "bytes": 8192,
+      "sha256": "4a7fc5a5f871ef5e796e0a197c33b6395dcc9f92543b56c2e4a4e581bd1b7a53"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4051.f32le": {
+      "bytes": 64,
+      "sha256": "0e2d3dcd80ebd53400cd5a1caf93371d8e551cb233b8e48a77d65f410b150e95"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4159.f32le": {
+      "bytes": 8192,
+      "sha256": "7bed4534f054fc0dc094483be0435a0a536c6361b11a26b07850558aa22e639b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4160.f32le": {
+      "bytes": 64,
+      "sha256": "cee0240edadad1e603de7a7d1363f871869be124bb925fb40c1a984efa5e7514"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4268.f32le": {
+      "bytes": 8192,
+      "sha256": "fdf64cc096f707625d60f8f46b05e0e353d408f01043c03ee3d7df06fc933dab"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4269.f32le": {
+      "bytes": 64,
+      "sha256": "322a6b71a51c5cd4cf98e3115c8935b75ac2b3a0d6a24c41b48259dd0521be53"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4377.f32le": {
+      "bytes": 8192,
+      "sha256": "640e7d893c2be8e3f2787d1dc2e64b9fcec3aafb530a2dd3e06e3d7ac18c1b30"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4378.f32le": {
+      "bytes": 64,
+      "sha256": "4356d54d83e0825fa281d7666335f207ea3464b6e80141fa520a4b08ae3794f5"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4486.f32le": {
+      "bytes": 8192,
+      "sha256": "70c58c99a9fa9e6b1c72ed04276dcc6a063c1bbbb059f5268b3c4aaafb7abe03"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4487.f32le": {
+      "bytes": 64,
+      "sha256": "7adf12e0f53418ad2734087b688dbeecc59bc55757bd31d0b9967c5ad2dcac7e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4595.f32le": {
+      "bytes": 8192,
+      "sha256": "d45800a66a8471f2f51e3e3d9596242623da6e2a40152368d143eb6f5488dc05"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4596.f32le": {
+      "bytes": 64,
+      "sha256": "c7594b20ac3b3560e62d0d216fd91f8535172eb51b952c79c6c479dee0309dce"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4704.f32le": {
+      "bytes": 8192,
+      "sha256": "bc7c46a182f858a9bfd0748d6f6b0d3b17bcb2be46178575b3301edfac0442be"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4705.f32le": {
+      "bytes": 64,
+      "sha256": "c1b20ad3e91ffab35d442276d6110bf8c8d86aba8a80ece89620a7bb701bdca8"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4813.f32le": {
+      "bytes": 8192,
+      "sha256": "8044a7109847783b4c7030a18a7d290d932fc425853e96ea86302a90d8aaceb1"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4814.f32le": {
+      "bytes": 64,
+      "sha256": "aaee68636f666392f947a954cd9c22574fd9c305a352cf6bdf5002fa56956ccb"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4922.f32le": {
+      "bytes": 8192,
+      "sha256": "ed177a9737fbaf32463e6f5d2341b4d17c57d96b20a463b66ac667272f0919ad"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-4923.f32le": {
+      "bytes": 64,
+      "sha256": "e39cf6c108dc99f31ec8a79ffa88a080b00e9ae3dfbe4c3a044d4a2af66982a2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5031.f32le": {
+      "bytes": 8192,
+      "sha256": "a772bbd8b3f2146ddf89913c465981b6a28fed96b3f42c5dd82bb55e4ae72e0b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5032.f32le": {
+      "bytes": 64,
+      "sha256": "ee9be70e64148e4b53d7579856a098d5a918dc5c2d7cfaa7e8e2d01cae6fb5c8"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5140.f32le": {
+      "bytes": 8192,
+      "sha256": "952eac836d2bbfc0889d5c0a4b7ed7f1d6fe82b00bd2bd1c7f32c3f6f577331e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5141.f32le": {
+      "bytes": 64,
+      "sha256": "44969c46575a1f474c8b427c6f5e17f24a0a854736a5c179bc66e76c68f405b7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5249.f32le": {
+      "bytes": 8192,
+      "sha256": "962cfcb9d4ada261808fc4019c96cd26e9acd8cf7a27fe68c4c14ff939ef473e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5250.f32le": {
+      "bytes": 64,
+      "sha256": "68c68406964ac3c551c8d350e58134eeef24b4b804b7ff50d2f34875716dfbb0"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5358.f32le": {
+      "bytes": 8192,
+      "sha256": "d3d5ebfdb9cc40e8df3fdb299615c6e09f41a7242c67a4cc20672417e977db15"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5359.f32le": {
+      "bytes": 64,
+      "sha256": "f920005af91a8e835dff529afc1f2da178b96a6ffa50c2aa3827770e945374c4"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5467.f32le": {
+      "bytes": 8192,
+      "sha256": "584ac46bc4fe4f9d13bf9bd02900b02642ef83e29977b481e1334e40b56f1b53"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5468.f32le": {
+      "bytes": 64,
+      "sha256": "53bca5f623355d27ba8f9a53c759835e6f087b49d5f407b1471d41e43765cf59"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5576.f32le": {
+      "bytes": 8192,
+      "sha256": "9acf49772686defd2d43e538c72938b05268eee49dc99da998ae548fed396e13"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5577.f32le": {
+      "bytes": 64,
+      "sha256": "5701629a49fdbb768ab65cbc350b7165bf5971ab23aff23b39416f447fc49946"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5685.f32le": {
+      "bytes": 8192,
+      "sha256": "a596e5691cb5bf07ad62af3f541be551e434490aea103e12d9e7778293886219"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5686.f32le": {
+      "bytes": 64,
+      "sha256": "6a87bba2339718192ee1bad1a8e73c24ef9c2f6de36da177958b2709fc7354a6"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5794.f32le": {
+      "bytes": 8192,
+      "sha256": "dd802d89f1f3c4c5cb75687f51fed3829e950b6ba1879103bb68a59576d8ba36"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5795.f32le": {
+      "bytes": 64,
+      "sha256": "98cceaea01bf5bbb0a4aaf33bc109456c9a0b680e75daad852ef982d1f78e453"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5903.f32le": {
+      "bytes": 8192,
+      "sha256": "b398f850a9e63d11595cb2e366466787f66cd983b5661fc8c32387a988fdc1bb"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-5904.f32le": {
+      "bytes": 64,
+      "sha256": "4b5759daab893b5c0b995fb3ec265ef6b213c1bfceead6bce7be20fe00352ea2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6012.f32le": {
+      "bytes": 8192,
+      "sha256": "5beb5350ecfda3a3ee1f4373ae4963fa1fe8446640da40417864b6e7cd45182b"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6013.f32le": {
+      "bytes": 64,
+      "sha256": "f335de41562407dae49acfef40b894b8f589e46cb7b97496c79b3fe27ae92bd4"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6121.f32le": {
+      "bytes": 8192,
+      "sha256": "3853943bbfeffbd4bdb2859473b7cc36cc0657fc202a8d58ba45b74f7104c4b2"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6122.f32le": {
+      "bytes": 64,
+      "sha256": "fa80cd553b89c503d1ff62ce6ddee8ef2c0fcab8effd92b66f026885ac73d5bc"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6230.f32le": {
+      "bytes": 8192,
+      "sha256": "952eac836d2bbfc0889d5c0a4b7ed7f1d6fe82b00bd2bd1c7f32c3f6f577331e"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6231.f32le": {
+      "bytes": 64,
+      "sha256": "44969c46575a1f474c8b427c6f5e17f24a0a854736a5c179bc66e76c68f405b7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6339.f32le": {
+      "bytes": 8192,
+      "sha256": "ccfe8c09fcb21c89534ab8985990b6ac83ded3add67c6551f1af8abf6b782338"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6340.f32le": {
+      "bytes": 64,
+      "sha256": "7337853eae863ab65390a6125b22a4076f038ada00c2db7b921df1bc34c5bb05"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6448.f32le": {
+      "bytes": 8192,
+      "sha256": "f3f61d0612993e4cd1ef4590041d3fe587df9c7156934ec9d1516cc7b6a277ca"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6449.f32le": {
+      "bytes": 64,
+      "sha256": "92835b8506605cfd4c8099586f954f5f17e892dd70f93e6c8e806f165e49939c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6557.f32le": {
+      "bytes": 8192,
+      "sha256": "b12c7354e8514958460d4a2d42bbe11536dc0cd9939ce8ee5ca8101a83c0c9cb"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6558.f32le": {
+      "bytes": 64,
+      "sha256": "a9007b8c96331a0bf30a9b4c35fff51fdafb41ac937c9283b3c8fadd8d7d5b5f"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6666.f32le": {
+      "bytes": 8192,
+      "sha256": "2a3ab0c598316d8d8bc4225622045664fe25bd736a446ba678c38a990b72fdfd"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6667.f32le": {
+      "bytes": 64,
+      "sha256": "e58345262462af5ed86b9d443dc652972338615a5637e8052f60bd117506d0c7"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6775.f32le": {
+      "bytes": 8192,
+      "sha256": "13be4a8f31a92fcefe48a38bcf9a40d0746847d9ed3cbdb7184061b8a7e0eee3"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6776.f32le": {
+      "bytes": 64,
+      "sha256": "2e79e57f00e2df6094fa23184571a2b2669255acd6d9e16cd8723a63b607e4f9"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6884.f32le": {
+      "bytes": 8192,
+      "sha256": "78e3cdb01cdafbba49da82158d2e6a48be24b24d62c54e0911c398a84f994a94"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/ar1/field-6885.f32le": {
+      "bytes": 64,
+      "sha256": "68806d6d15514e1c902a0f80b0275a754436b2135d04a4b4a385a5ee8292653c"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/bulk64/field-3.f32le": {
+      "bytes": 524288,
+      "sha256": "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include/llama.h": {
+      "bytes": 89978,
+      "sha256": "c9a5f025ad7dbf07eda6660c8ff2c32cd80c07ba440d80728707a9c9d13c1bc3"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/src/models/qwen35.cpp": {
+      "bytes": 28742,
+      "sha256": "deec1f546ed1562a8fb8811fe21ce823ef3ed66bae2dde7a3c791068e89a9efd"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/src/llama-graph.cpp": {
+      "bytes": 144062,
+      "sha256": "a9acd6050aad04e82dee31a13a013656195c83c56e2554a2f07d997e25d8cae4"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml.h": {
+      "bytes": 116144,
+      "sha256": "12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml-alloc.h": {
+      "bytes": 3689,
+      "sha256": "d114067f9099c4a0e2bbc0625c676d4894d9cf0ea27168ee072c7de0e71bb51d"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml-backend.h": {
+      "bytes": 25776,
+      "sha256": "5791edae1fb7622027621d15bc7a160198bd3b75ed8f96d06cb7b85e2b873756"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml.c": {
+      "bytes": 261112,
+      "sha256": "7e8bf2b60598b50d718f46aa3c36c9293b10f4e7dfde6844f0693e6fc976d505"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend.cpp": {
+      "bytes": 105561,
+      "sha256": "cdb79f2937aa147084a76f949adc0eb760ad1883e0ccc015bb4e0f599c89bd5f"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend-dl.cpp": {
+      "bytes": 1051,
+      "sha256": "acd76df8b83bcdef6807bc7b0520f57b379d61b4ff1027c04aa4d140d16ad07f"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend-reg.cpp": {
+      "bytes": 18555,
+      "sha256": "0f63c69ad083e0744872f57d59d049b069283303f5bbc646222907d779afe4d1"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.cpp": {
+      "bytes": 25005,
+      "sha256": "ccf382d3fbf126b5ddb5850a0a970656fb24e00587da4003419439277c91cd01"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.c": {
+      "bytes": 132440,
+      "sha256": "45b22f8c83b2cbccabfc2f81a38ef8a5810223e1ce366b7c6f77d42faa5cc6f3"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/repack.cpp": {
+      "bytes": 213604,
+      "sha256": "5fa9f24cc2631609f40fa47a64078f77dccbe9f327711af91af671cd29b8b87d"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/tiled/tiled.cpp": {
+      "bytes": 58049,
+      "sha256": "a5d92ce8f617696778e225de6658180649143fec3792c5cfef84170641f2d06e"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/llamafile/sgemm.h": {
+      "bytes": 524,
+      "sha256": "6ec12aa9244a8ffb13ae56b2e391f905b1d152ae25ff45da6a66e18daf650b3b"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/llamafile/sgemm.cpp": {
+      "bytes": 166263,
+      "sha256": "8413c9a0828ddea7289aee72a32c03bd56a23ee40179ec3600f9b0589a21e338"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/vec.cpp": {
+      "bytes": 25373,
+      "sha256": "31e43ec66a1e2920a549dd2f71e6ce474e68f2fc298c4d2942404ca327e763d1"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/simd-mappings.h": {
+      "bytes": 55790,
+      "sha256": "a936298928e4dcfc1b61fb9e99cf8a152572b413e30f38dc8e7c3211ee9cb6a0"
+    },
+    "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml.so.0.26.0": {
+      "bytes": 58424,
+      "sha256": "1df83a5f445c119d10cd4fcac260c150a32d1b2ddf01dd49c9686c8b958f5c70"
+    },
+    "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml-base.so.0.26.0": {
+      "bytes": 972584,
+      "sha256": "dc4856a2ad667979352c9f270274b84f8135575c7394bd3b00e415e9e371e903"
+    },
+    "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so": {
+      "bytes": 1459208,
+      "sha256": "ca75af8b6281c57114f411d8883120a4ee595d8cb05c998a0b5e40921417c935"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/target/models/qwen35-2b/model.gguf": {
+      "bytes": 1296764000,
+      "sha256": "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6"
+    },
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/INPUT-CLOSURE.json": {
+      "bytes": 1488,
+      "sha256": "8f73dd76505ec074f0aed61d96ea2cc22ba0404e0e2121df9e5bc13b465683cb"
+    },
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml-cpu.h": {
+      "bytes": 7753,
+      "sha256": "316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e"
+    }
+  },
+  "total_source_payload_copy_cap_bytes": 4194304,
+  "intended_route": "F32 tiled-ineligible; public sgemm n64 accepted versusn1 early refusal; public use_ref skips sgemm. Actual entry/control/output proof ROOT pending.",
+  "runtime_closure_scope": "Pins complete relevant source/ggml DSO files and records actualmaps; no entire host-runtime/internalmicrokernel qualification claim"
+}
--- /dev/null
+++ original2B-alpha-public-replay/INPUT-CLOSURE.json
@@ -0,0 +1,55 @@
+{
+  "scope": "Byte-only source/capture audit; no operator/model execution; first OBSERVED gap, not automatic global cause",
+  "preceding_exact_rows": 960,
+  "preceding_stage_families": 15,
+  "actual_input_words": 131072,
+  "actual_input_shape_bulk": [
+    2048,
+    64,
+    1,
+    1
+  ],
+  "actual_input_shape_AR1": [
+    2048,
+    1,
+    1,
+    1
+  ],
+  "all64_actual_inputs_byte_equal": true,
+  "input_sha256": "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6",
+  "weight_source_interval": {
+    "start": 459622304,
+    "bytes": 131072,
+    "end_exclusive": 459753376,
+    "shape": [
+      2048,
+      16,
+      1,
+      1
+    ],
+    "kind": "F32_GGUF_RAW",
+    "sha256": "0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876"
+  },
+  "first_observed_causal_scope_difference": {
+    "layer": 0,
+    "stage": "projection_output::blk.0.ssm_alpha.weight",
+    "occurrence": 0,
+    "position": 0,
+    "causal_scope": "all actual token-result words",
+    "all_raw_unequal_words": 13,
+    "causal_unequal_words": 13,
+    "first_raw_mismatch": {
+      "index": 0,
+      "bulk_bits": "bf1ef412",
+      "ar1_bits": "bf1ef414"
+    },
+    "first_causal_mismatch": {
+      "index": 0,
+      "bulk_bits": "bf1ef412",
+      "ar1_bits": "bf1ef414"
+    }
+  },
+  "weight_readset_scope": "Actual same ordinaryCPU_Mapped owner, fullbytes independently exact to source; no opaque repack enters this operator",
+  "internal_dispatch_observed": false,
+  "cause_qualified": false
+}
--- /dev/null
+++ original2B-alpha-public-replay/README.md
@@ -0,0 +1,123 @@
+# Original2B alpha projection public replay — ROOT EXECUTES
+
+Parent: numerics-and-context. Owner: spine_review. New ignored diagnostic only.
+No MAIN/plan/math edits, build, native/operator/model/container/GPU execution by
+this author. Preparation used source, metadata and bounded raw-byte comparison.
+Original capture/helpers/frozen attempts remain untouched.
+
+## Earliest actual input closure
+
+Accepted root-first64-repack-transport capture independently retains plain versus
+observed whole post-state and next real AR1 full248320/whole-state transparency.
+Its first observed gap is layer0 projection_output::blk.0.ssm_alpha.weight at
+position0:13/16 words, first bulkbf1ef412 versus AR1bf1ef414, before bias/softplus.
+
+- All960 preceding compared rows (15 captured stage families x64) are exact:
+  embedding, actual RMS/weighted norm, QKV, conv four-tap inputs/results, Q/K/V,
+  and every alpha input. This is captured-scope order, not an unobserved global
+  first-cause claim.
+- Independently read all131072 actual F32 alpha-input words. Bulk [2048,64,1,1]
+  contiguous bytes equal every corresponding64 AR1 [2048,1,1,1] input exactly;
+  SHA731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6.
+  Actual source tensorID5 attn_norm-0/MUL matches the captured producer and each
+  actual projection src1, checked before/after. No tokenizer/state reconstruction.
+- Full32768 F32 learned alpha words, shape[2048,16,1,1],131072B, SHA
+  0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876,
+  independently equal original approved GGUF bytes [459622304,459753376).
+  Actual source buffer is ordinary CPU_Mapped/GGUF_RAW; opaque CPU_REPACK bytes
+  from other model operators do not enter this standalone F32 projection.
+- Exact original GGUF SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6,
+  full source/header/capture metadata and relevant original source/DSO files are
+  pinned. Author read only bounded alpha source range; ROOT authenticate performs
+  whole-source before/after hashing. INPUT-CLOSURE.json records this audit.
+
+## Genuine public computation and observation
+
+- Same pinned original llama.cpp0.6.0 Haswell CPU backend and public
+  ggml_mul_mat(F32 weight[2048,16],F32 input[2048,64 or1]); all ne/nb/dtypes checked.
+  CPU payloads are populated with unchanged actual raw bits. No model load,
+  dequantization, custom dot loop, kernel replacement or mathematical fallback.
+  Input tensor/buffer ownership is independent diagnostic storage, not an alias
+  or substitute model/source identity.
+- Original64-row operator plus64 original AR1 operators,8 backend threads.
+  Every default output must bit-match its entire authentic capture:1024 bulk
+  words+64x16 AR1words=2048. Any default mismatch refuses BEFORE any control.
+- Only after all65 defaults pass, repeat65 calls using the explicitly public
+  ggml_backend_cpu_set_use_ref control. This is a named diagnostic condition,
+  not original model parity or permission to change Rust/production kernels.
+  Record every control output and complete refbulk versus defaultAR1 comparison.
+- Plain no-observer run first; separate same binary/inputs run with source-only
+  observer. ALL130 full default/control outputs in each run must be identical
+  across plain/observed before route evidence is accepted (4096words/run).
+- Optional LD_PRELOAD observer uses exact public sgemm.h13-argument ABI and
+  unchanged opaque params. Original public target is exact already-loaded
+  Haswell DSO via NOLOAD+dlsym+dladdr at authenticated exported offset0x10c8f0.
+  Call original exactly once, forward all pointers/scalars and return original
+  bool unchanged. No private params layout read, arithmetic or substitution.
+- Integer-only bounded2048-call ring;8 expected source entry calls/bulk and each
+  AR1 call, no control calls. Source-derived intended population:8 bulk64 true,
+  64x8 AR1 false,65 use_ref zero =520. Exact m16/n64-or1/k2048, lda/ldb2048,
+  ldc16, F32 types, zero bound operand offsets. ROOT must actually establish it.
+- Primary source: qwen35.cpp379-381/build_lora_mm uses this exact projection;
+  tiled.cpp rejects F32 source; ggml-cpu.c1308+ gates the public sgemm branch;
+  sgemm.cpp3911 returns false for n<2 and its F32 AVX branch selects tinyBLAS;
+  use_ref skips sgemm. These are source branch facts, not observed internal
+  instruction/microkernel routes. The wrapper records exported entry only.
+- Actual maps before/after must show only pinned Haswell CPU backend. Exact
+  relevant source and ggml DSO pins do not establish entire host runtime closure.
+  Internal microkernel/long-context cause/profile/model eligibility remain false.
+  Existing Rust declared AR1 kernels are untouched; no fitting nativebulk math.
+
+## Bounds and ROOT procedure
+
+Fresh private fixtures<4MiB (actual payload1,187,840B), expected values copied only
+from authentic actual captures, never candidate math. C++ opens expected payloads
+for comparison only after genuine public computation; outer source authentication
+hashes fixture files before/after. Every input is re-read in CPU storage and file
+form after each operator. F32 finite checks preserve all raw bits/signedzero.
+
+No network/GPU/model loading. Process AS<=8GiB, CPU deadline<=300x8+1s, wall300s,
+8 backend threads, backing<=2MiB/case, graph metadata2MiB/case. Owned PGID launcher
+is copied from accepted AV wrapper: WNOWAIT preserves leader identity, unconditional
+TERM/KILL/reap/group-absence and pipeEOF/closure happen before finalreceipt I/O,
+even if leader already exited. Logs<=8MiB/run; binary<=16MiB, observer<=8MiB.
+Entire new ignored-run artifact tree<=512MiB including reserved bounded report.
+No old capture/failed attempt is overwritten. Python AST only by author; no tests.
+
+ROOT verifies freeze and source pins, then builds with its accepted GCC/glibc2.44
+closure and strict warnings. No fast-math or upstream build. Root alone runs this
+after competing model/CPU/GPU work permits; placeholders require actual hashes.
+
+```sh
+TASK_ALPHA=.agents/runs/numerics-and-context-2b-alpha-replay-20261020
+TASK_NATIVE=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
+TASK_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
+TASK_C=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
+python3 "$TASK_ALPHA/prepare.py" "$TASK_ALPHA/root-fixtures"
+c++ -std=c++17 -O2 -Wall -Wextra -Werror \
+  -I "$TASK_NATIVE/ggml/include" "$TASK_ALPHA/main.cpp" \
+  -L "$TASK_LIB" -L "$TASK_C" -Wl,-rpath,"$TASK_LIB:$TASK_C" \
+  -Wl,--dynamic-linker,"$TASK_C/ld-linux-x86-64.so.2" \
+  -lggml -lggml-base -ldl -lm -o "$TASK_ALPHA/root-replay"
+c++ -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC \
+  -I "$TASK_NATIVE/ggml/src/ggml-cpu/llamafile" "$TASK_ALPHA/observer.cpp" \
+  -L "$TASK_C" -Wl,-rpath,"$TASK_C" -ldl -o "$TASK_ALPHA/root-observer.so"
+python3 "$TASK_ALPHA/run_replay.py" --binary "$TASK_ALPHA/root-replay" \
+  --binary-sha256 ACTUAL_REPLAY_SHA --fixtures "$TASK_ALPHA/root-fixtures" \
+  --new-output "$TASK_ALPHA/root-plain" --new-log "$TASK_ALPHA/root-plain.log" \
+  --new-receipt "$TASK_ALPHA/root-plain.invocation.json"
+python3 "$TASK_ALPHA/run_replay.py" --binary "$TASK_ALPHA/root-replay" \
+  --binary-sha256 ACTUAL_REPLAY_SHA --fixtures "$TASK_ALPHA/root-fixtures" \
+  --preload "$TASK_ALPHA/root-observer.so" --preload-sha256 ACTUAL_OBSERVER_SHA \
+  --new-output "$TASK_ALPHA/root-observed" --new-log "$TASK_ALPHA/root-observed.log" \
+  --new-receipt "$TASK_ALPHA/root-observed.invocation.json"
+python3 "$TASK_ALPHA/validate.py" --plain-output "$TASK_ALPHA/root-plain" \
+  --plain-receipt "$TASK_ALPHA/root-plain.invocation.json" \
+  --observed-output "$TASK_ALPHA/root-observed" \
+  --observed-receipt "$TASK_ALPHA/root-observed.invocation.json" \
+  --new-report "$TASK_ALPHA/root-validation.json"
+```
+
+Successful output means full public-operator replay/observer transparency and
+exported-entry diagnostic evidence. It does not close full runtime/internal
+kernel causality, historical8K corpus equivalence, HF/vLLM math or model support.
--- /dev/null
+++ original2B-alpha-public-replay/TODO.md
@@ -0,0 +1,18 @@
+# Original2B alpha first-gap public replay
+
+Parent: numerics-and-context. Owner: spine_review; ignored new run only.
+ROOT owns all builds/tests/native/operator/model/GPU/container execution.
+
+- [x] Independently inspect actual first64 capture/comparison;960 preceding rows
+  exact, full131072 alpha input words exact and actual producer binding closed.
+- [x] Independently byte-match all32768 alpha F32 weights to original GGUF
+  range[459622304,459753376); distinguish ordinary raw from opaque otherweights.
+- [x] Prepare genuine public ggml full64+64AR default replay, gated publicuse_ref
+  controls, unchanged ABI/return entry observer and full plain/observed validator.
+- [x] Preserve existing helpers/attempts; pin all relevant capture/source/DSO files,
+  bounded owned lifecycle, exact finite shapes/outputs and no internalcause claim.
+- [ ] Independent source review, ROOT strict build and byte-only fixture creation.
+- [ ] ROOT actual all65 default2048 words exact BEFORE65 controls; all130
+  plain/observed4096 words exact BEFORE520 exported-entry route population accepted.
+- [ ] Separately interpret actual control results/source branch proof. Do not alter
+  declared Rust AR1 math by fitting nativebulk; no automatic full-longcontext cause.
--- /dev/null
+++ original2B-alpha-public-replay/UPSTREAM_LICENSE
@@ -0,0 +1,21 @@
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

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/REPACK-TRANSPORT-FOLLOWUP.patch

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

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/SOURCE_ONLY_DELTA.patch

Original UTF-8 bytes: 37656; SHA256 `7b99ba68badb63c04c6db22594006589fe01d4a38118329b114e96935e600a6a`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/native_window.cpp
+++ b/native_window.cpp
@@ -5,10 +5,10 @@
 #undef main
 #include <map>
 
-constexpr size_t BULK_CAP = 128 * 1024 * 1024;
-constexpr size_t AR1_CAP = 192 * 1024 * 1024;
+constexpr size_t BULK_CAP = 112 * 1024 * 1024;
+constexpr size_t AR1_CAP = 112 * 1024 * 1024;
 constexpr size_t SNAPSHOTS_CAP = 136 * 1024 * 1024;
-constexpr size_t ATTEMPT_CAP = 512 * 1024 * 1024;
+constexpr size_t ATTEMPT_CAP = 480 * 1024 * 1024;
 constexpr size_t META_CAP = 16 * 1024 * 1024;
 static int window_start = 0, window_count = 0, call_index = 0, call_count = 0;
 static std::string provider_mode;
@@ -17,6 +17,11 @@
 static std::set<int> observed_layers;
 static int gdn_calls = 0;
 static size_t actual_context_capacity = 0;
+static std::map<const ggml_tensor*,size_t> tensor_ids;
+static size_t tensor_id(const ggml_tensor *t) {
+    if(!t)return 0;
+    return tensor_ids.emplace(t,tensor_ids.size()+1).first->second;
+}
 
 static std::vector<float> window_dense(const ggml_tensor *t,const char *site="selected_or_operand") {
     size_t count=1,span=sizeof(float);
@@ -31,19 +36,7 @@
             require(add<=std::numeric_limits<size_t>::max()-span,"window span addition overflow");span+=add;
         }
         require(count==size_t(ggml_nelements(t)) && span==ggml_nbytes(t),"window exact physical span");
-        if(span>16*1024*1024) {
-            // Pinned V storage is [capacity,head*dimension], then viewed and
-            // permuted. Read occupied logical rows, never the capacity gaps.
-            require(std::string(site)=="attention_V_cache_operand" && std::string(t->name).rfind("cache_v_l3",0)==0
-                && t->op==GGML_OP_PERMUTE && actual_context_capacity==8448
-                && t->ne[0]>=256 && t->ne[0]<=1024 && t->ne[0]%256==0
-                && t->ne[1]==256 && t->ne[2]==2 && t->ne[3]==1
-                && t->nb[1]==actual_context_capacity*4
-                && t->nb[2]==actual_context_capacity*256*4
-                && t->nb[3]==actual_context_capacity*512*4
-                && span==((512-1)*actual_context_capacity+size_t(t->ne[0]))*4
-                && span<=actual_context_capacity*512*4,"unsupported large window V-cache span");
-        }
+        require(span<=16*1024*1024,"bounded first-two-layer F32 physical span");
         require(t->data && (t->view_src ? t->view_src->buffer : t->buffer),"window tensor physical buffer");
         std::vector<float> result(count);size_t written=0;
         const size_t run=size_t(t->ne[0])*sizeof(float);
@@ -71,12 +64,12 @@
     if(end==std::string::npos)return -1;
     const auto part=name.substr(4,end-4);
     if(part.empty() || part.size()>2 || part.find_first_not_of("0123456789")!=std::string::npos)return -1;
-    const int layer=std::stoi(part);return layer<5 ? layer : -1;
+    const int layer=std::stoi(part);return layer<2 ? layer : -1;
 }
 static void record(int layer,const std::string &stage,const std::vector<float> &values,
                    const ggml_tensor *t,int token_axis,const char *scope="actual_tensor",
                    bool mask=false,const std::vector<int64_t> &shape={},size_t source_offset=0) {
-    require(layer>=-1 && layer<=4 && !values.empty() && values.size()<=2*1024*1024,"window field geometry");
+    require(layer>=-1 && layer<=1 && !values.empty() && values.size()<=2*1024*1024,"window field geometry");
     require(provider_mode=="bulk64" || provider_mode=="ar1","known window provider mode");
     const size_t cap=provider_mode=="bulk64" ? BULK_CAP : AR1_CAP;
     require(window_bytes<=cap && values.size()*4<=cap-window_bytes && window_records<16384,"window aggregate cap");
@@ -95,7 +88,9 @@
     for(int i=0;i<4;i++)records<<(i?",":"")<<(t?t->ne[i]:0);
     records << "],\"native_nb\":[";
     for(int i=0;i<4;i++)records<<(i?",":"")<<(t?t->nb[i]:0);
-    records << "],\"masked_negative_infinity_allowed\":"<<(mask?"true":"false")<<"}\n";
+    records << "],\"native_tensor_id\":"<<tensor_id(t)<<",\"native_source_ids\":[";
+    for(int i=0;i<GGML_MAX_SRC;i++)records<<(i?",":"")<<tensor_id(t?t->src[i]:nullptr);
+    records<<"],\"masked_negative_infinity_allowed\":"<<(mask?"true":"false")<<"}\n";
     require(bool(records) && records.tellp()>=0 && uint64_t(records.tellp())<=META_CAP,"window metadata cap/write");
     if(layer>=0)observed_layers.insert(layer);
 }
@@ -103,55 +98,121 @@
     const std::string name=t->name;
     if(name=="model.input_embed")return {-1,1};
     for(const auto &item:std::vector<std::pair<std::string,int>>{
-        {"attn_norm",1},{"attn_residual",1},{"attn_post_norm",1},{"ffn_out",1},{"l_out",1},
+        {"norm",-2},{"ffn_up",1},{"ffn_gate",1},{"ffn_swiglu",1},{"attn_norm",1},{"attn_residual",1},{"attn_post_norm",1},{"ffn_out",1},{"l_out",1},
         {"linear_attn_qkv_mixed",1},{"z",1},{"alpha",1},{"a_softplus",1},{"gate",1},
         {"beta",2},{"beta_sigmoid",2},{"conv_output_raw",1},{"conv_output_silu",1},
         {"q_conv_predelta",2},{"k_conv_predelta",2},{"v_conv_predelta",2},
         {"final_output",1},{"linear_attn_out",1}}) {
-        const int layer=named_layer(name,item.first);if(layer>=0 && layer<=4)return {layer,item.second};
-    }
-    for(const auto &item:std::vector<std::pair<std::string,int>>{
-        {"Qcur_full",2},{"Qcur_normed",2},{"Kcur_normed",2},{"Qcur",2},{"Kcur",2},{"Vcur",2},
-        {"gate_reshaped",1},{"gate_sigmoid",1},{"kq",1},{"kq_soft_max",1},{"kqv",1},
-        {"kqv_out",1},{"attn_pregate",1},{"attn_gated",1},{"attn_output",1}}) {
-        if(named_layer(name,item.first)==3)return {3,item.second};
+        const int layer=named_layer(name,item.first);if(layer>=0 && layer<=1)return {layer,item.second};
     }
     return {-2,-1};
 }
+
+constexpr size_t LEARNED_CAP=80*1024*1024, LEARNED_CHUNK=1024*1024;
+struct LearnedRole { const char *name; ggml_type type; int64_t columns,rows; };
+static const LearnedRole learned_roles[]={
+    {"attn_gate.weight",GGML_TYPE_Q4_0,2048,2048}, {"attn_norm.weight",GGML_TYPE_F32,2048,1},
+    {"attn_qkv.weight",GGML_TYPE_Q4_0,2048,6144}, {"ffn_down.weight",GGML_TYPE_Q4_1,6144,2048},
+    {"ffn_gate.weight",GGML_TYPE_Q4_0,2048,6144}, {"ffn_up.weight",GGML_TYPE_Q4_0,2048,6144},
+    {"post_attention_norm.weight",GGML_TYPE_F32,2048,1}, {"ssm_a",GGML_TYPE_F32,16,1},
+    {"ssm_alpha.weight",GGML_TYPE_F32,2048,16}, {"ssm_beta.weight",GGML_TYPE_F32,2048,16},
+    {"ssm_conv1d.weight",GGML_TYPE_F32,4,6144}, {"ssm_dt.bias",GGML_TYPE_F32,16,1},
+    {"ssm_norm.weight",GGML_TYPE_F32,128,1}, {"ssm_out.weight",GGML_TYPE_Q8_0,2048,2048},
+};
+struct Learned { const ggml_tensor *tensor; size_t bytes,index; void *data; ggml_backend_buffer_t buffer; };
+static std::map<std::string,Learned> learned;
+static fs::path learned_folder;
+static std::ofstream learned_records;
+static size_t learned_bytes=0;
+static void capture_learned(const ggml_tensor *t) {
+    const int layer=t ? tensor_layer(t->name) : -1;
+    if(layer<0)return;
+    const std::string name=t->name,prefix="blk."+std::to_string(layer)+".";
+    const LearnedRole *role=nullptr;
+    for(const auto &candidate:learned_roles)if(name==prefix+candidate.name)role=&candidate;
+    require(role && t->type==role->type && t->ne[0]==role->columns && t->ne[1]==role->rows
+        && t->ne[2]==1 && t->ne[3]==1 && t->op==GGML_OP_NONE && !t->view_src
+        && t->data && t->buffer && ggml_is_contiguous(t),"exact original2B learned physical operand");
+    const size_t bytes=ggml_nbytes(t);
+    require(bytes==ggml_row_size(role->type,role->columns)*size_t(role->rows)
+        && bytes<=8*1024*1024,"exact original packed learned bytes");
+    const auto found=learned.find(name);
+    if(found!=learned.end()) {
+        require(found->second.tensor==t && found->second.bytes==bytes && found->second.data==t->data
+            && found->second.buffer==t->buffer,"all contexts share same immutable learned owner");return;
+    }
+    require(provider_mode=="bulk64" && learned.size()<28 && learned_bytes<=LEARNED_CAP
+        && bytes<=LEARNED_CAP-learned_bytes,"bounded complete shared learned operands");
+    const size_t index=learned.size()+1;
+    learned.emplace(name,Learned{t,bytes,index,t->data,t->buffer});learned_bytes+=bytes;
+    learned_records<<"{\"name\":"<<quote(name)<<",\"type\":"<<int(t->type)
+        <<",\"type_name\":"<<quote(ggml_type_name(t->type))<<",\"bytes\":"<<bytes<<",\"ne\":[";
+    for(int i=0;i<4;i++)learned_records<<(i?",":"")<<t->ne[i];
+    learned_records<<"],\"nb\":[";for(int i=0;i<4;i++)learned_records<<(i?",":"")<<t->nb[i];
+    learned_records<<"],\"chunks\":[";
+    for(size_t at=0;at<bytes;at+=LEARNED_CHUNK) {
+        const size_t length=std::min(LEARNED_CHUNK,bytes-at);
+        std::vector<unsigned char> raw(length);ggml_backend_tensor_get(t,raw.data(),at,length);
+        const std::string file="learned-"+std::to_string(index)+"-"+std::to_string(at/LEARNED_CHUNK)+".bin";
+        write_bytes(learned_folder/file,raw.data(),raw.size());
+        if(at)learned_records<<',';
+        learned_records<<"{\"file\":"<<quote(file)<<",\"offset\":"<<at<<",\"bytes\":"<<length<<'}';
+    }
+    learned_records<<"]}\n";learned_records.flush();
+    require(bool(learned_records) && learned_records.tellp()>=0 && uint64_t(learned_records.tellp())<=128*1024,"learned metadata cap/write");
+}
+static void verify_learned_unchanged() {
+    require(learned.size()==28,"complete original2B first-two-layer learned closure");
+    for(const auto &entry:learned) {
+        const auto &r=entry.second;
+        require(r.tensor->data==r.data && r.tensor->buffer==r.buffer,"retained learned physical owner");
+        for(size_t at=0;at<r.bytes;at+=LEARNED_CHUNK) {
+            const size_t length=std::min(LEARNED_CHUNK,r.bytes-at);
+            std::vector<unsigned char> actual(length);ggml_backend_tensor_get(r.tensor,actual.data(),at,length);
+            const std::string file="learned-"+std::to_string(r.index)+"-"+std::to_string(at/LEARNED_CHUNK)+".bin";
+            const auto before=read_bounded(learned_folder/file,LEARNED_CHUNK);
+            require(before.size()==length && actual==before,"original learned operand changed during actual computations");
+        }
+    }
+}
+
 static bool window_observer(ggml_tensor *t,bool ask,void*) {
     if(!enabled || !t)return false;
     const auto pick=selected(t);
     const int projection=t->op==GGML_OP_MUL_MAT && t->src[0] ? tensor_layer(t->src[0]->name) : -1;
     const int gdn=t->op==GGML_OP_GATED_DELTA_NET && t->src[5] ? named_layer(t->src[5]->name,"state_predelta") : -1;
-    const bool wanted=pick.first!=-2 || projection>=0 || (gdn>=0 && gdn<=4);
+    bool learned_op=false;
+    for(int i=0;i<GGML_MAX_SRC;i++)learned_op=learned_op || (t->src[i] && tensor_layer(t->src[i]->name)>=0);
+    const bool wanted=pick.first!=-2 || projection>=0 || learned_op || (gdn>=0 && gdn<=1);
     if(ask)return wanted;
     require(wanted,"unexpected window callback");check_time();
+    for(int i=0;i<GGML_MAX_SRC;i++)capture_learned(t->src[i]);
+    if(learned_op && projection<0) {
+        const ggml_tensor *weight=nullptr;
+        for(int i=0;i<GGML_MAX_SRC;i++)if(t->src[i] && tensor_layer(t->src[i]->name)>=0) {
+            require(!weight,"one actual learned operand per closed operator");weight=t->src[i];
+        }
+        const int axis=t->ne[0]==128 && t->ne[1]==16 ? 2 : 1;
+        record(tensor_layer(weight->name),"learned_op_output::"+std::string(weight->name),window_dense(t),t,axis);
+    }
+    if(named_layer(t->name,"conv_output_raw")>=0 && named_layer(t->name,"conv_output_raw")<=1) {
+        require(t->op==GGML_OP_SSM_CONV && t->src[0] && t->src[1]
+            && t->src[0]->ne[0]==call_count+3 && t->src[0]->ne[1]==6144
+            && t->src[0]->ne[2]==1 && t->src[0]->ne[3]==1,"actual full conv causal-history geometry");
+        record(named_layer(t->name,"conv_output_raw"),"ssm_conv_input",window_dense(t->src[0]),t->src[0],-1,"actual_all_history_plus_current_rows");
+    }
     if(pick.first!=-2) {
         // Q/K/V labels occur on both original projection nodes and later
         // head-shaped views. Preserve the actual source geometry for each.
-        const int axis=t->op==GGML_OP_MUL_MAT ? 1 : pick.second;
+        const int axis=pick.second==-2 ? (t->ne[0]==128 && t->ne[1]==16 ? 2 : 1)
+                       : t->op==GGML_OP_MUL_MAT ? 1 : pick.second;
         record(pick.first,t->name,window_dense(t),t,axis);
     }
     if(projection>=0) {
         record(projection,"projection_input::"+std::string(t->src[0]->name),window_dense(t->src[1]),t->src[1],1);
         record(projection,"projection_output::"+std::string(t->src[0]->name),window_dense(t),t,1);
     }
-    if(named_layer(t->name,"kq")==3) {
-        require(t->op==GGML_OP_MUL_MAT && t->src[0] && t->src[1],"actual QK operator");
-        record(3,"attention_K_cache_operand",window_dense(t->src[0]),t->src[0],-1,"full_actual_padded_cache_raw; causal_rows_only_for_per_query_comparison");
-        record(3,"attention_Q_operand",window_dense(t->src[1]),t->src[1],1);
-    }
-    if(named_layer(t->name,"kq_soft_max")==3) {
-        require(t->op==GGML_OP_SOFT_MAX && t->src[0] && t->src[1],"actual softmax/mask operator");
-        record(3,"attention_softmax_scores_operand",window_dense(t->src[0]),t->src[0],1);
-        record(3,"attention_mask_operand",window_dense(t->src[1]),t->src[1],1,"actual_mask",true);
-    }
-    if(named_layer(t->name,"kqv")==3) {
-        require(t->op==GGML_OP_MUL_MAT && t->src[0] && t->src[1],"actual AV operator");
-        record(3,"attention_V_cache_operand",window_dense(t->src[0],"attention_V_cache_operand"),t->src[0],-1,"full_actual_padded_cache_raw; causal_rows_only_for_per_query_comparison");
-        record(3,"attention_probabilities_operand",window_dense(t->src[1]),t->src[1],1);
-    }
-    if(gdn>=0 && gdn<=4) {
+    if(gdn>=0 && gdn<=1) {
         require(gdn!=3 && t->src[0] && t->src[1] && t->src[2] && t->src[3] && t->src[4] && t->src[5],"actual GDN source binding");
         require(t->src[2]->ne[0]==128 && t->src[2]->ne[1]==16 && t->src[2]->ne[2]==call_count && t->src[2]->ne[3]==1,"actual unchanged GDN token width");
         int32_t snapshots=0;std::memcpy(&snapshots,t->op_params,4);require(snapshots==1,"original GDN final-state policy");
@@ -183,22 +244,29 @@
         }
     };
     one(-1,"model.input_embed");
-    for(int layer=0;layer<=4;layer++) {
-        for(const char *base:{"attn_norm","attn_residual","attn_post_norm","ffn_out","l_out"})one(layer,std::string(base)+"-"+std::to_string(layer));
+    for(int layer=0;layer<=1;layer++) {
+        for(const char *base:{"attn_norm","attn_residual","attn_post_norm","ffn_out","l_out",
+            "linear_attn_qkv_mixed","z","alpha","a_softplus","gate","beta","beta_sigmoid",
+            "conv_output_raw","conv_output_silu","q_conv_predelta","k_conv_predelta","v_conv_predelta",
+            "final_output","linear_attn_out"})one(layer,std::string(base)+"-"+std::to_string(layer));
         projections(layer,{"ffn_gate.weight","ffn_up.weight","ffn_down.weight"});
-        if(layer==3)projections(layer,{"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight"});
-        else projections(layer,{"attn_qkv.weight","ssm_alpha.weight","ssm_beta.weight","attn_gate.weight","ssm_out.weight"});
+        for(const char *base:{"ffn_up","ffn_gate","ffn_swiglu"})one(layer,std::string(base)+"-"+std::to_string(layer));
+        require(occurrences[{layer,"norm-"+std::to_string(layer)}]==3,"all three original RMS producer boundaries");
+        one(layer,"ssm_conv_input");
+        for(const char *role:{"attn_norm.weight","post_attention_norm.weight","ssm_a","ssm_dt.bias","ssm_conv1d.weight","ssm_norm.weight"})
+            one(layer,"learned_op_output::blk."+std::to_string(layer)+"."+role);
+        projections(layer,{"attn_qkv.weight","ssm_alpha.weight","ssm_beta.weight","attn_gate.weight","ssm_out.weight"});
         if(layer!=3) {
             for(const char *stage:{"gdn_q_operand","gdn_k_operand","gdn_v_operand","gdn_gate_operand","gdn_beta_operand","gdn_core_output"})one(layer,stage);
             if(provider_mode=="bulk64" || call_index==0)one(layer,"gdn_prior_state");
             if(provider_mode=="bulk64" || call_index+1==window_count)one(layer,"gdn_proposed_state");
         }
     }
-    for(const char *stage:{"Qcur_normed-3","Kcur_normed-3","kq-3","kq_soft_max-3","kqv-3","attn_pregate-3","attn_gated-3","attention_K_cache_operand","attention_Q_operand","attention_V_cache_operand","attention_softmax_scores_operand","attention_mask_operand","attention_probabilities_operand"})one(3,stage);
+
 }
 static size_t end_observation() {
-    require(observed_layers==std::set<int>({0,1,2,3,4}),"missing first-five-layer observations");
-    require(gdn_calls==4*(provider_mode=="bulk64" ? 1 : window_count),"missing actual GDN operations");
+    require(observed_layers==std::set<int>({0,1}),"missing first-two-layer observations");
+    require(gdn_calls==2*(provider_mode=="bulk64" ? 1 : window_count),"missing actual GDN operations");
     records.flush();require(bool(records),"window final records flush");records.close();return window_bytes;
 }
 static void guard_next(llama_context *plain,llama_context *observed,const std::vector<llama_token> &ids,int end,const fs::path &out) {
@@ -213,8 +281,8 @@
     try {
         require(argc==7,"usage: native_window GGUF ids.u32le aligned_start threads seconds NEW_OUTPUT");
         size_t consumed=0;const std::string start_arg=argv[3];const long start=std::stol(start_arg,&consumed);
-        require(consumed==start_arg.size() && start>=0 && start<=960 && start%64==0,"original1K aligned block start");
-        window_start=int(start);window_count=std::min(64,1023-window_start);
+        require(consumed==start_arg.size() && start>=0 && start<=960 && start%64==0,"bounded early aligned block of original8K prefix");
+        window_start=int(start);window_count=64;
         const int threads=integer(argv[4],8),seconds=integer(argv[5],7200);
         deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
         const uint32_t endian=1;require(*reinterpret_cast<const unsigned char*>(&endian)==1 && sizeof(float)==4,"little-endian F32 host");
@@ -225,7 +293,7 @@
         write_bytes(base/"input_ids.u32le",raw.data(),raw.size());
         llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
         std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
-        require(bool(model) && llama_model_n_layer(model.get())==24 && llama_model_n_embd(model.get())==1024 && llama_vocab_n_tokens(llama_model_get_vocab(model.get()))==248320,"original0.8B geometry");
+        require(bool(model) && llama_model_n_layer(model.get())==24 && llama_model_n_embd(model.get())==2048 && llama_vocab_n_tokens(llama_model_get_vocab(model.get()))==248320,"original2B geometry");
         auto params=llama_context_default_params();params.n_ctx=8320;params.n_batch=64;params.n_ubatch=64;
         params.n_threads=threads;params.n_threads_batch=threads;params.type_k=GGML_TYPE_F32;params.type_v=GGML_TYPE_F32;
         params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;params.offload_kqv=false;params.op_offload=false;params.no_perf=true;params.cb_eval=nullptr;
@@ -249,16 +317,20 @@
         write_bytes(base/"bulk.post-state.bin",after_bulk.data(),after_bulk.size());write_bytes(base/"ar1.post-state.bin",after_serial.data(),after_serial.size());
         params.cb_eval=window_observer;Context observed_bulk(llama_init_from_model(model.get(),params),llama_free);require(bool(observed_bulk),"observed bulk context");
         require(llama_state_set_data(observed_bulk.get(),before_bulk.data(),before_bulk.size())==before_bulk.size(),"same-source bulk prior restore");
-        begin_observation(base,"bulk64");enabled=true;call_index=0;call_count=window_count;occurrences.clear();
+        learned_folder=base/"learned-operands";require(fs::create_directory(learned_folder),"new learned operand directory");
+        require(chmod(learned_folder.c_str(),0700)==0,"learned operand permissions");
+        learned_records.open(learned_folder/"records.jsonl");require(bool(learned_records),"learned metadata open");
+        begin_observation(base,"bulk64");enabled=true;call_index=0;call_count=window_count;occurrences.clear();tensor_ids.clear();
         decode(observed_bulk.get(),ids,window_start,window_count,false);enabled=false;require_current_coverage();
         const size_t bulk_bytes=end_observation();require(state(observed_bulk.get())==after_bulk,"observer changed unchanged-width bulk complete state");
         guard_next(bulk.get(),observed_bulk.get(),ids,window_start+window_count,base/"bulk.next-logits.f32le");
         serial_params.cb_eval=window_observer;Context observed_ar(llama_init_from_model(model.get(),serial_params),llama_free);require(bool(observed_ar),"observed AR1 context");
         require(llama_state_set_data(observed_ar.get(),before_serial.data(),before_serial.size())==before_serial.size(),"same-source AR1 prior restore");
         begin_observation(base,"ar1");call_count=1;enabled=true;
-        for(int at=0;at<window_count;at++){call_index=at;occurrences.clear();decode(observed_ar.get(),ids,window_start+at,1,false);require_current_coverage();}
+        for(int at=0;at<window_count;at++){call_index=at;occurrences.clear();tensor_ids.clear();decode(observed_ar.get(),ids,window_start+at,1,false);require_current_coverage();}
         enabled=false;const size_t ar_bytes=end_observation();require(state(observed_ar.get())==after_serial,"observer changed AR1 complete state");
         guard_next(serial.get(),observed_ar.get(),ids,window_start+window_count,base/"ar1.next-logits.f32le");
+        verify_learned_unchanged();learned_records.close();
         size_t artifact_bytes=0;
         for(const auto &entry:fs::recursive_directory_iterator(base)) {
             require(!entry.is_symlink(),"no capture artifact symlink");
@@ -270,12 +342,12 @@
         }
         std::ofstream done(base/"complete.json");require(bool(done),"window complete record open");
         done<<"{\"status\":\"TRANSPARENT_WINDOW_CAPTURE_NOT_CAUSE_QUALIFICATION\",\"model_path\":"<<quote(fs::canonical(argv[1]).string())
-            <<",\"window_start\":"<<window_start<<",\"window_tokens\":"<<window_count<<",\"original_case_cut\":1024,\"threads\":"<<threads
+            <<",\"hidden_width\":2048,\"selected_layers\":[0,1],\"learned_operand_count\":28,\"learned_operand_bytes\":"<<learned_bytes<<",\"learned_unchanged_after_compute\":true,\"window_start\":"<<window_start<<",\"window_tokens\":"<<window_count<<",\"original_case_cut\":8192,\"threads\":"<<threads
             <<",\"bulk_actual_ubatch\":64,\"ar1_actual_ubatch\":1,\"common_prior_exact\":true,\"bulk_instrumentation_complete_state_exact\":true,\"ar1_instrumentation_complete_state_exact\":true"
             <<",\"both_next_AR1_full248320_and_whole_state_exact\":true,\"post_state_bulk_vs_ar1_equal\":"<<(after_bulk==after_serial?"true":"false")
             <<",\"bulk_payload_bytes\":"<<bulk_bytes<<",\"ar1_payload_bytes\":"<<ar_bytes
             <<",\"precompletion_artifact_bytes\":"<<artifact_bytes
-            <<",\"prefix_policy\":\"unchanged original15x64+63; no arbitrary cut binary-search inference\",\"operator_provider_internal_dispatch_observed\":false}\n";
+            <<",\"prefix_policy\":\"unchanged original8191-prefix127x64+63; early aligned64-row block, no arbitrary-cut inference\",\"operator_provider_internal_dispatch_observed\":false}\n";
         done.flush();require(bool(done),"window complete record write");require(chmod((base/"complete.json").c_str(),0600)==0,"window complete mode");
         std::cerr<<"TRANSPARENT_NATIVE_WINDOW_CAPTURE_COMPLETE start="<<window_start<<" tokens="<<window_count<<" cause_qualified=false\n";
         observed_ar.reset();observed_bulk.reset();serial.reset();bulk.reset();model.reset();llama_backend_free();return 0;
--- a/compare_window.py
+++ b/compare_window.py
@@ -4,12 +4,14 @@
 import hashlib
 import json
 import math
+import os
+import stat
 from pathlib import Path
 import re
 import struct
 import sys
 
-MODE_CAPS = {"bulk64": 128 * 1024 * 1024, "ar1": 192 * 1024 * 1024}
+MODE_CAPS = {"bulk64": 112 * 1024 * 1024, "ar1": 112 * 1024 * 1024}
 FILE_DIGESTS = {}
 
 
@@ -49,7 +51,7 @@
         require(row["mode"] == mode and type(row["call"]) is int
                 and 0 <= row["call"] < (1 if mode == "bulk64" else count)
                 and row["call_tokens"] == (count if mode == "bulk64" else 1), "actual call identity/width")
-        require(type(row["layer"]) is int and -1 <= row["layer"] <= 4
+        require(type(row["layer"]) is int and -1 <= row["layer"] <= 1
                 and isinstance(row["stage"], str) and len(row["stage"]) <= 256
                 and type(row["occurrence"]) is int and 0 <= row["occurrence"] <= 16,
                 "bounded field identity")
@@ -60,6 +62,10 @@
         require(isinstance(shape, list) and len(shape) == 4
                 and all(type(n) is int and n > 0 for n in shape)
                 and math.prod(shape) == row["elements"], "actual dense tensor shape")
+        require(type(row["native_tensor_id"]) is int and 0 < row["native_tensor_id"] < 1000000
+                and len(row["native_source_ids"]) == 10
+                and all(type(value) is int and 0 <= value < 1000000 for value in row["native_source_ids"]),
+                "actual per-call original tensor/operand identifiers")
         axis = row["token_axis"]
         require(type(axis) is int and -1 <= axis <= 3
                 and (axis == -1 or shape[axis] == row["call_tokens"]), "declared source token axis")
@@ -81,12 +87,19 @@
         require(lookup.get((call, layer, stage)) == [0], f"missing/duplicate required field: {call}/{layer}/{stage}")
     for call in range(1 if mode == "bulk64" else count):
         one(call, -1, "model.input_embed")
-        for layer in range(5):
-            for stage in ["attn_norm", "attn_residual", "attn_post_norm", "ffn_out", "l_out"]:
+        for layer in range(2):
+            for stage in ["attn_norm", "attn_residual", "attn_post_norm", "ffn_out", "l_out",
+                          "linear_attn_qkv_mixed", "z", "alpha", "a_softplus", "gate", "beta", "beta_sigmoid",
+                          "conv_output_raw", "conv_output_silu", "q_conv_predelta", "k_conv_predelta", "v_conv_predelta",
+                          "final_output", "linear_attn_out", "ffn_up", "ffn_gate", "ffn_swiglu"]:
                 one(call, layer, f"{stage}-{layer}")
             roles = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
-            roles += (["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"] if layer == 3 else
-                      ["attn_qkv.weight", "ssm_alpha.weight", "ssm_beta.weight", "attn_gate.weight", "ssm_out.weight"])
+            roles += ["attn_qkv.weight", "ssm_alpha.weight", "ssm_beta.weight", "attn_gate.weight", "ssm_out.weight"]
+            require(lookup.get((call, layer, f"norm-{layer}")) == [0, 1, 2], "all three real RMS producer outputs")
+            one(call, layer, "ssm_conv_input")
+            for role in ["attn_norm.weight", "post_attention_norm.weight", "ssm_a", "ssm_dt.bias",
+                         "ssm_conv1d.weight", "ssm_norm.weight"]:
+                one(call, layer, f"learned_op_output::blk.{layer}.{role}")
             for role in roles:
                 for prefix in ["projection_input::", "projection_output::"]:
                     one(call, layer, f"{prefix}blk.{layer}.{role}")
@@ -97,11 +110,7 @@
                     one(call, layer, "gdn_prior_state")
                 if mode == "bulk64" or call + 1 == count:
                     one(call, layer, "gdn_proposed_state")
-        for stage in ["Qcur_normed-3", "Kcur_normed-3", "kq-3", "kq_soft_max-3", "kqv-3",
-                      "attn_pregate-3", "attn_gated-3", "attention_K_cache_operand", "attention_Q_operand",
-                      "attention_V_cache_operand", "attention_softmax_scores_operand", "attention_mask_operand",
-                      "attention_probabilities_operand"]:
-            one(call, 3, stage)
+
 
 
 def field_bytes(folder, row):
@@ -146,6 +155,71 @@
                 first_raw_mismatch=first, first_causal_mismatch=first_causal)
 
 
+def digest_file(path):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
+        before = os.fstat(stream.fileno())
+        require(stat.S_ISREG(before.st_mode), "regular pinned source")
+        h = hashlib.sha256()
+        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
+            h.update(chunk)
+        after = os.fstat(stream.fileno())
+        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
+                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
+                "source changed during hash")
+    return {"bytes": before.st_size, "sha256": h.hexdigest()}
+
+
+def verify_learned(capture, done):
+    here = Path(__file__).resolve().parent
+    descriptors = jread(here / "learned-source-headers.json")
+    model = Path(descriptors["model"])
+    expected_source = {"bytes": descriptors["model_bytes"], "sha256": descriptors["model_sha256"]}
+    require(Path(done["model_path"]).resolve() == model.resolve() and digest_file(model) == expected_source,
+            "independently pinned approved original2B GGUF bytes")
+    ids = bounded(capture / "input_ids.u32le", 32768)
+    require(len(ids) == 32768 and hashlib.sha256(ids).hexdigest() ==
+            "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d", "exact original8192 teacher file")
+    expected = {row["name"]: row for row in descriptors["selected"]}
+    require(len(expected) == 28, "actual header first-two-layer closure")
+    rows = [json.loads(line, object_pairs_hook=unique) for line in
+            bounded(capture / "learned-operands/records.jsonl", 128 * 1024).splitlines()]
+    require(len(rows) == 28, "all actual learned operand descriptors")
+    seen, files, total, hashes = set(), set(), 0, {}
+    with os.fdopen(os.open(model, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
+        for row in rows:
+            name = row["name"]
+            require(name in expected and name not in seen, "unique original learned role")
+            seen.add(name)
+            original = expected[name]
+            require(row["type"] == original["type"] and row["ne"] == original["ne"]
+                    and row["bytes"] == original["bytes"] and 0 < row["bytes"] <= 8 * 1024**2,
+                    "exact original learned dtype/shape/byte geometry")
+            at, content = 0, hashlib.sha256()
+            for chunk in row["chunks"]:
+                require(re.fullmatch(r"learned-[1-9][0-9]?-[0-9]+\.bin", chunk["file"])
+                        and chunk["file"] not in files and type(chunk["offset"]) is int
+                        and chunk["offset"] == at and type(chunk["bytes"]) is int
+                        and chunk["bytes"] == min(1024**2, row["bytes"]-at), "complete bounded raw operand chunks")
+                files.add(chunk["file"])
+                raw = bounded(capture / "learned-operands" / chunk["file"], 1024**2)
+                require(len(raw) == chunk["bytes"], "exact raw source chunk")
+                source.seek(original["absolute_offset"] + at)
+                require(source.read(len(raw)) == raw, "actual physical learned operand equals approved GGUF bytes")
+                if row["type"] == 0:
+                    require(all(bits & 0x7f800000 != 0x7f800000 for (bits,) in struct.iter_unpack("<I", raw)),
+                            "finite actual F32 learned source")
+                content.update(raw)
+                at += len(raw)
+            require(at == row["bytes"], "complete actual packed learned operand")
+            total += at
+            hashes[name] = {"bytes": at, "sha256": content.hexdigest(), "type": row["type"], "ne": row["ne"]}
+    require(seen == set(expected) and total == descriptors["selected_raw_bytes"] == done["learned_operand_bytes"]
+            and total <= 80 * 1024**2, "all28 actual raw learned operands accounted")
+    require(digest_file(model) == expected_source, "original GGUF unchanged after actual readset validation")
+    return {"all28_equal_original_source_bytes": True, "bytes": total, "raw_operands": hashes,
+            "weight_dequantization_or_arithmetic_executed": False}
+
+
 def main():
     p = argparse.ArgumentParser(description=__doc__)
     p.add_argument("--capture", type=Path, required=True)
@@ -158,15 +232,20 @@
             and done["ar1_instrumentation_complete_state_exact"] is True
             and done["both_next_AR1_full248320_and_whole_state_exact"] is True,
             "mandatory unchanged-window/state/next-token transparency guards")
+    require(done["hidden_width"] == 2048 and done["selected_layers"] == [0, 1]
+            and done["original_case_cut"] == 8192 and done["learned_operand_count"] == 28
+            and done["learned_unchanged_after_compute"] is True and done["threads"] == 8,
+            "approved original2B/first-two-layer/8K-prefix source scope")
+    learned = verify_learned(a.capture, done)
     start, count = done["window_start"], done["window_tokens"]
     require(type(start) is int and type(count) is int and start in range(0, 961, 64)
-            and count == min(64, 1023 - start), "original15x64+63 segmentation")
+            and count == 64, "early aligned64-row block from original127x64+63 prefix")
     artifact_bytes = 0
     for path in a.capture.rglob("*"):
         require(not path.is_symlink() and (path.is_dir() or path.is_file()), "regular capture tree")
         if path.is_file() and path != a.capture / "complete.json":
             artifact_bytes += path.stat().st_size
-            require(artifact_bytes <= 512 * 1024 * 1024 - 128 * 1024, "whole capture artifact cap")
+            require(artifact_bytes <= 480 * 1024 * 1024 - 128 * 1024, "whole capture artifact cap")
     require(artifact_bytes == done["precompletion_artifact_bytes"], "complete artifact byte accounting")
     snapshots = ["common.prior-state.bin", "bulk.post-state.bin", "ar1.post-state.bin"]
     require(sum((a.capture / name).stat().st_size for name in snapshots) <= 136 * 1024 * 1024,
@@ -180,6 +259,23 @@
         require(key[0] == 0, "one unchanged actual bulk call")
         left = field_bytes(a.capture / "bulk64", row)
         layer, stage, occurrence = key[1:]
+        if stage == "ssm_conv_input":
+            require(row["dense_ne"] == [count+3, 6144, 1, 1] and row["native_op"] == "CONCAT",
+                    "actual full conv history/current sequence")
+            for at in range(count):
+                other_key = (at, layer, stage, occurrence)
+                require(other_key in serial, "all original AR1 conv causal operands")
+                other = serial[other_key]
+                require(other["dense_ne"] == [4, 6144, 1, 1], "original four-tap AR1 operand")
+                right = field_bytes(a.capture / "ar1", other)
+                # Byte-only actual-history codec; no convolution or generated expected arithmetic.
+                taps = b"".join(left[(channel*(count+3)+at)*4:(channel*(count+3)+at+4)*4]
+                                for channel in range(6144))
+                output.append(dict(layer=layer, stage=stage+"::actual4taps", occurrence=occurrence,
+                                   position=start+at, causal_scope="all four actual history/current taps",
+                                   **differences(taps, right)))
+                used_ar.add(other_key)
+            continue
         if row["token_axis"] == -1:
             # Caches contain the whole original block in bulk but only the
             # partial causal frontier in earlier AR1 calls. Compare whole cache
@@ -216,15 +312,14 @@
     # Intermediate AR cache operands are retained and intentionally not equal to
     # bulk's final whole-block cache. They are not silently treated as results.
     remaining = [key for key in serial if key not in used_ar]
-    require(all(serial[key]["stage"] in ["attention_K_cache_operand", "attention_V_cache_operand"]
-                and key[0] < count - 1 for key in remaining), "unmatched observed AR1 field")
+    require(not remaining, "unmatched observed first-two-layer AR1 field")
     report = dict(status="WINDOW_FIELD_COMPARISON_DIAGNOSTIC_ONLY", window_start=start, window_tokens=count,
         fields=output, first_observed_causal_scope_difference=next((r for r in output if r["causal_unequal_words"]), None),
         different_field_rows=sum(r["causal_unequal_words"] > 0 for r in output),
         unpaired_intermediate_actual_cache_operands=len(remaining),
         bulk_records_sha256=bulk_hash, ar1_records_sha256=serial_hash,
         complete_sha256=hashlib.sha256(bounded(a.capture / "complete.json", 128 * 1024)).hexdigest(),
-        actual_field_sha256=FILE_DIGESTS,
+        actual_field_sha256=FILE_DIGESTS, learned_operands=learned,
         tolerance="none; signed zero preserved; future causal-mask-excluded QK columns reported separately",
         cause_qualified=False, internal_dispatch_observed=False,
         scope="native unchanged-width bulk versus native AR1 actual tensors; no Rust parity, long-context eligibility or causal-operator attribution automatically inferred")

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/WINDOW-BINDING-FOLLOWUP.patch

Original UTF-8 bytes: 1892; SHA256 `371e8462a1c3f8059695edb1d3c50bf3bc1e738f5674cc15c639ec15515011af`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/compare_window.py
+++ b/compare_window.py
@@ -40,7 +40,7 @@
     return json.loads(bounded(path, cap), object_pairs_hook=unique)
 
 
-def read_mode(folder, mode, count):
+def read_mode(folder, mode, start, count):
     require(mode in MODE_CAPS, "known observation mode")
     raw = bounded(folder / "records.jsonl", 16 * 1024 * 1024)
     lines = raw.splitlines()
@@ -51,6 +51,9 @@
         require(row["mode"] == mode and type(row["call"]) is int
                 and 0 <= row["call"] < (1 if mode == "bulk64" else count)
                 and row["call_tokens"] == (count if mode == "bulk64" else 1), "actual call identity/width")
+        require(type(row["window_start"]) is int and row["window_start"] == start
+                and type(row["window_tokens"]) is int and row["window_tokens"] == count,
+                "exact completed original window binding")
         require(type(row["layer"]) is int and -1 <= row["layer"] <= 1
                 and isinstance(row["stage"], str) and len(row["stage"]) <= 256
                 and type(row["occurrence"]) is int and 0 <= row["occurrence"] <= 16,
@@ -250,8 +253,8 @@
     snapshots = ["common.prior-state.bin", "bulk.post-state.bin", "ar1.post-state.bin"]
     require(sum((a.capture / name).stat().st_size for name in snapshots) <= 136 * 1024 * 1024,
             "combined three-snapshot cap")
-    bulk, bulk_hash, bulk_bytes = read_mode(a.capture / "bulk64", "bulk64", count)
-    serial, serial_hash, ar_bytes = read_mode(a.capture / "ar1", "ar1", count)
+    bulk, bulk_hash, bulk_bytes = read_mode(a.capture / "bulk64", "bulk64", start, count)
+    serial, serial_hash, ar_bytes = read_mode(a.capture / "ar1", "ar1", start, count)
     require(bulk_bytes == done["bulk_payload_bytes"] and ar_bytes == done["ar1_payload_bytes"],
             "actual payload totals equal completion record")
     output, used_ar = [], set()

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-prepared/SOURCE_ONLY_DELTA.patch

Original UTF-8 bytes: 37656; SHA256 `7b99ba68badb63c04c6db22594006589fe01d4a38118329b114e96935e600a6a`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/native_window.cpp
+++ b/native_window.cpp
@@ -5,10 +5,10 @@
 #undef main
 #include <map>
 
-constexpr size_t BULK_CAP = 128 * 1024 * 1024;
-constexpr size_t AR1_CAP = 192 * 1024 * 1024;
+constexpr size_t BULK_CAP = 112 * 1024 * 1024;
+constexpr size_t AR1_CAP = 112 * 1024 * 1024;
 constexpr size_t SNAPSHOTS_CAP = 136 * 1024 * 1024;
-constexpr size_t ATTEMPT_CAP = 512 * 1024 * 1024;
+constexpr size_t ATTEMPT_CAP = 480 * 1024 * 1024;
 constexpr size_t META_CAP = 16 * 1024 * 1024;
 static int window_start = 0, window_count = 0, call_index = 0, call_count = 0;
 static std::string provider_mode;
@@ -17,6 +17,11 @@
 static std::set<int> observed_layers;
 static int gdn_calls = 0;
 static size_t actual_context_capacity = 0;
+static std::map<const ggml_tensor*,size_t> tensor_ids;
+static size_t tensor_id(const ggml_tensor *t) {
+    if(!t)return 0;
+    return tensor_ids.emplace(t,tensor_ids.size()+1).first->second;
+}
 
 static std::vector<float> window_dense(const ggml_tensor *t,const char *site="selected_or_operand") {
     size_t count=1,span=sizeof(float);
@@ -31,19 +36,7 @@
             require(add<=std::numeric_limits<size_t>::max()-span,"window span addition overflow");span+=add;
         }
         require(count==size_t(ggml_nelements(t)) && span==ggml_nbytes(t),"window exact physical span");
-        if(span>16*1024*1024) {
-            // Pinned V storage is [capacity,head*dimension], then viewed and
-            // permuted. Read occupied logical rows, never the capacity gaps.
-            require(std::string(site)=="attention_V_cache_operand" && std::string(t->name).rfind("cache_v_l3",0)==0
-                && t->op==GGML_OP_PERMUTE && actual_context_capacity==8448
-                && t->ne[0]>=256 && t->ne[0]<=1024 && t->ne[0]%256==0
-                && t->ne[1]==256 && t->ne[2]==2 && t->ne[3]==1
-                && t->nb[1]==actual_context_capacity*4
-                && t->nb[2]==actual_context_capacity*256*4
-                && t->nb[3]==actual_context_capacity*512*4
-                && span==((512-1)*actual_context_capacity+size_t(t->ne[0]))*4
-                && span<=actual_context_capacity*512*4,"unsupported large window V-cache span");
-        }
+        require(span<=16*1024*1024,"bounded first-two-layer F32 physical span");
         require(t->data && (t->view_src ? t->view_src->buffer : t->buffer),"window tensor physical buffer");
         std::vector<float> result(count);size_t written=0;
         const size_t run=size_t(t->ne[0])*sizeof(float);
@@ -71,12 +64,12 @@
     if(end==std::string::npos)return -1;
     const auto part=name.substr(4,end-4);
     if(part.empty() || part.size()>2 || part.find_first_not_of("0123456789")!=std::string::npos)return -1;
-    const int layer=std::stoi(part);return layer<5 ? layer : -1;
+    const int layer=std::stoi(part);return layer<2 ? layer : -1;
 }
 static void record(int layer,const std::string &stage,const std::vector<float> &values,
                    const ggml_tensor *t,int token_axis,const char *scope="actual_tensor",
                    bool mask=false,const std::vector<int64_t> &shape={},size_t source_offset=0) {
-    require(layer>=-1 && layer<=4 && !values.empty() && values.size()<=2*1024*1024,"window field geometry");
+    require(layer>=-1 && layer<=1 && !values.empty() && values.size()<=2*1024*1024,"window field geometry");
     require(provider_mode=="bulk64" || provider_mode=="ar1","known window provider mode");
     const size_t cap=provider_mode=="bulk64" ? BULK_CAP : AR1_CAP;
     require(window_bytes<=cap && values.size()*4<=cap-window_bytes && window_records<16384,"window aggregate cap");
@@ -95,7 +88,9 @@
     for(int i=0;i<4;i++)records<<(i?",":"")<<(t?t->ne[i]:0);
     records << "],\"native_nb\":[";
     for(int i=0;i<4;i++)records<<(i?",":"")<<(t?t->nb[i]:0);
-    records << "],\"masked_negative_infinity_allowed\":"<<(mask?"true":"false")<<"}\n";
+    records << "],\"native_tensor_id\":"<<tensor_id(t)<<",\"native_source_ids\":[";
+    for(int i=0;i<GGML_MAX_SRC;i++)records<<(i?",":"")<<tensor_id(t?t->src[i]:nullptr);
+    records<<"],\"masked_negative_infinity_allowed\":"<<(mask?"true":"false")<<"}\n";
     require(bool(records) && records.tellp()>=0 && uint64_t(records.tellp())<=META_CAP,"window metadata cap/write");
     if(layer>=0)observed_layers.insert(layer);
 }
@@ -103,55 +98,121 @@
     const std::string name=t->name;
     if(name=="model.input_embed")return {-1,1};
     for(const auto &item:std::vector<std::pair<std::string,int>>{
-        {"attn_norm",1},{"attn_residual",1},{"attn_post_norm",1},{"ffn_out",1},{"l_out",1},
+        {"norm",-2},{"ffn_up",1},{"ffn_gate",1},{"ffn_swiglu",1},{"attn_norm",1},{"attn_residual",1},{"attn_post_norm",1},{"ffn_out",1},{"l_out",1},
         {"linear_attn_qkv_mixed",1},{"z",1},{"alpha",1},{"a_softplus",1},{"gate",1},
         {"beta",2},{"beta_sigmoid",2},{"conv_output_raw",1},{"conv_output_silu",1},
         {"q_conv_predelta",2},{"k_conv_predelta",2},{"v_conv_predelta",2},
         {"final_output",1},{"linear_attn_out",1}}) {
-        const int layer=named_layer(name,item.first);if(layer>=0 && layer<=4)return {layer,item.second};
-    }
-    for(const auto &item:std::vector<std::pair<std::string,int>>{
-        {"Qcur_full",2},{"Qcur_normed",2},{"Kcur_normed",2},{"Qcur",2},{"Kcur",2},{"Vcur",2},
-        {"gate_reshaped",1},{"gate_sigmoid",1},{"kq",1},{"kq_soft_max",1},{"kqv",1},
-        {"kqv_out",1},{"attn_pregate",1},{"attn_gated",1},{"attn_output",1}}) {
-        if(named_layer(name,item.first)==3)return {3,item.second};
+        const int layer=named_layer(name,item.first);if(layer>=0 && layer<=1)return {layer,item.second};
     }
     return {-2,-1};
 }
+
+constexpr size_t LEARNED_CAP=80*1024*1024, LEARNED_CHUNK=1024*1024;
+struct LearnedRole { const char *name; ggml_type type; int64_t columns,rows; };
+static const LearnedRole learned_roles[]={
+    {"attn_gate.weight",GGML_TYPE_Q4_0,2048,2048}, {"attn_norm.weight",GGML_TYPE_F32,2048,1},
+    {"attn_qkv.weight",GGML_TYPE_Q4_0,2048,6144}, {"ffn_down.weight",GGML_TYPE_Q4_1,6144,2048},
+    {"ffn_gate.weight",GGML_TYPE_Q4_0,2048,6144}, {"ffn_up.weight",GGML_TYPE_Q4_0,2048,6144},
+    {"post_attention_norm.weight",GGML_TYPE_F32,2048,1}, {"ssm_a",GGML_TYPE_F32,16,1},
+    {"ssm_alpha.weight",GGML_TYPE_F32,2048,16}, {"ssm_beta.weight",GGML_TYPE_F32,2048,16},
+    {"ssm_conv1d.weight",GGML_TYPE_F32,4,6144}, {"ssm_dt.bias",GGML_TYPE_F32,16,1},
+    {"ssm_norm.weight",GGML_TYPE_F32,128,1}, {"ssm_out.weight",GGML_TYPE_Q8_0,2048,2048},
+};
+struct Learned { const ggml_tensor *tensor; size_t bytes,index; void *data; ggml_backend_buffer_t buffer; };
+static std::map<std::string,Learned> learned;
+static fs::path learned_folder;
+static std::ofstream learned_records;
+static size_t learned_bytes=0;
+static void capture_learned(const ggml_tensor *t) {
+    const int layer=t ? tensor_layer(t->name) : -1;
+    if(layer<0)return;
+    const std::string name=t->name,prefix="blk."+std::to_string(layer)+".";
+    const LearnedRole *role=nullptr;
+    for(const auto &candidate:learned_roles)if(name==prefix+candidate.name)role=&candidate;
+    require(role && t->type==role->type && t->ne[0]==role->columns && t->ne[1]==role->rows
+        && t->ne[2]==1 && t->ne[3]==1 && t->op==GGML_OP_NONE && !t->view_src
+        && t->data && t->buffer && ggml_is_contiguous(t),"exact original2B learned physical operand");
+    const size_t bytes=ggml_nbytes(t);
+    require(bytes==ggml_row_size(role->type,role->columns)*size_t(role->rows)
+        && bytes<=8*1024*1024,"exact original packed learned bytes");
+    const auto found=learned.find(name);
+    if(found!=learned.end()) {
+        require(found->second.tensor==t && found->second.bytes==bytes && found->second.data==t->data
+            && found->second.buffer==t->buffer,"all contexts share same immutable learned owner");return;
+    }
+    require(provider_mode=="bulk64" && learned.size()<28 && learned_bytes<=LEARNED_CAP
+        && bytes<=LEARNED_CAP-learned_bytes,"bounded complete shared learned operands");
+    const size_t index=learned.size()+1;
+    learned.emplace(name,Learned{t,bytes,index,t->data,t->buffer});learned_bytes+=bytes;
+    learned_records<<"{\"name\":"<<quote(name)<<",\"type\":"<<int(t->type)
+        <<",\"type_name\":"<<quote(ggml_type_name(t->type))<<",\"bytes\":"<<bytes<<",\"ne\":[";
+    for(int i=0;i<4;i++)learned_records<<(i?",":"")<<t->ne[i];
+    learned_records<<"],\"nb\":[";for(int i=0;i<4;i++)learned_records<<(i?",":"")<<t->nb[i];
+    learned_records<<"],\"chunks\":[";
+    for(size_t at=0;at<bytes;at+=LEARNED_CHUNK) {
+        const size_t length=std::min(LEARNED_CHUNK,bytes-at);
+        std::vector<unsigned char> raw(length);ggml_backend_tensor_get(t,raw.data(),at,length);
+        const std::string file="learned-"+std::to_string(index)+"-"+std::to_string(at/LEARNED_CHUNK)+".bin";
+        write_bytes(learned_folder/file,raw.data(),raw.size());
+        if(at)learned_records<<',';
+        learned_records<<"{\"file\":"<<quote(file)<<",\"offset\":"<<at<<",\"bytes\":"<<length<<'}';
+    }
+    learned_records<<"]}\n";learned_records.flush();
+    require(bool(learned_records) && learned_records.tellp()>=0 && uint64_t(learned_records.tellp())<=128*1024,"learned metadata cap/write");
+}
+static void verify_learned_unchanged() {
+    require(learned.size()==28,"complete original2B first-two-layer learned closure");
+    for(const auto &entry:learned) {
+        const auto &r=entry.second;
+        require(r.tensor->data==r.data && r.tensor->buffer==r.buffer,"retained learned physical owner");
+        for(size_t at=0;at<r.bytes;at+=LEARNED_CHUNK) {
+            const size_t length=std::min(LEARNED_CHUNK,r.bytes-at);
+            std::vector<unsigned char> actual(length);ggml_backend_tensor_get(r.tensor,actual.data(),at,length);
+            const std::string file="learned-"+std::to_string(r.index)+"-"+std::to_string(at/LEARNED_CHUNK)+".bin";
+            const auto before=read_bounded(learned_folder/file,LEARNED_CHUNK);
+            require(before.size()==length && actual==before,"original learned operand changed during actual computations");
+        }
+    }
+}
+
 static bool window_observer(ggml_tensor *t,bool ask,void*) {
     if(!enabled || !t)return false;
     const auto pick=selected(t);
     const int projection=t->op==GGML_OP_MUL_MAT && t->src[0] ? tensor_layer(t->src[0]->name) : -1;
     const int gdn=t->op==GGML_OP_GATED_DELTA_NET && t->src[5] ? named_layer(t->src[5]->name,"state_predelta") : -1;
-    const bool wanted=pick.first!=-2 || projection>=0 || (gdn>=0 && gdn<=4);
+    bool learned_op=false;
+    for(int i=0;i<GGML_MAX_SRC;i++)learned_op=learned_op || (t->src[i] && tensor_layer(t->src[i]->name)>=0);
+    const bool wanted=pick.first!=-2 || projection>=0 || learned_op || (gdn>=0 && gdn<=1);
     if(ask)return wanted;
     require(wanted,"unexpected window callback");check_time();
+    for(int i=0;i<GGML_MAX_SRC;i++)capture_learned(t->src[i]);
+    if(learned_op && projection<0) {
+        const ggml_tensor *weight=nullptr;
+        for(int i=0;i<GGML_MAX_SRC;i++)if(t->src[i] && tensor_layer(t->src[i]->name)>=0) {
+            require(!weight,"one actual learned operand per closed operator");weight=t->src[i];
+        }
+        const int axis=t->ne[0]==128 && t->ne[1]==16 ? 2 : 1;
+        record(tensor_layer(weight->name),"learned_op_output::"+std::string(weight->name),window_dense(t),t,axis);
+    }
+    if(named_layer(t->name,"conv_output_raw")>=0 && named_layer(t->name,"conv_output_raw")<=1) {
+        require(t->op==GGML_OP_SSM_CONV && t->src[0] && t->src[1]
+            && t->src[0]->ne[0]==call_count+3 && t->src[0]->ne[1]==6144
+            && t->src[0]->ne[2]==1 && t->src[0]->ne[3]==1,"actual full conv causal-history geometry");
+        record(named_layer(t->name,"conv_output_raw"),"ssm_conv_input",window_dense(t->src[0]),t->src[0],-1,"actual_all_history_plus_current_rows");
+    }
     if(pick.first!=-2) {
         // Q/K/V labels occur on both original projection nodes and later
         // head-shaped views. Preserve the actual source geometry for each.
-        const int axis=t->op==GGML_OP_MUL_MAT ? 1 : pick.second;
+        const int axis=pick.second==-2 ? (t->ne[0]==128 && t->ne[1]==16 ? 2 : 1)
+                       : t->op==GGML_OP_MUL_MAT ? 1 : pick.second;
         record(pick.first,t->name,window_dense(t),t,axis);
     }
     if(projection>=0) {
         record(projection,"projection_input::"+std::string(t->src[0]->name),window_dense(t->src[1]),t->src[1],1);
         record(projection,"projection_output::"+std::string(t->src[0]->name),window_dense(t),t,1);
     }
-    if(named_layer(t->name,"kq")==3) {
-        require(t->op==GGML_OP_MUL_MAT && t->src[0] && t->src[1],"actual QK operator");
-        record(3,"attention_K_cache_operand",window_dense(t->src[0]),t->src[0],-1,"full_actual_padded_cache_raw; causal_rows_only_for_per_query_comparison");
-        record(3,"attention_Q_operand",window_dense(t->src[1]),t->src[1],1);
-    }
-    if(named_layer(t->name,"kq_soft_max")==3) {
-        require(t->op==GGML_OP_SOFT_MAX && t->src[0] && t->src[1],"actual softmax/mask operator");
-        record(3,"attention_softmax_scores_operand",window_dense(t->src[0]),t->src[0],1);
-        record(3,"attention_mask_operand",window_dense(t->src[1]),t->src[1],1,"actual_mask",true);
-    }
-    if(named_layer(t->name,"kqv")==3) {
-        require(t->op==GGML_OP_MUL_MAT && t->src[0] && t->src[1],"actual AV operator");
-        record(3,"attention_V_cache_operand",window_dense(t->src[0],"attention_V_cache_operand"),t->src[0],-1,"full_actual_padded_cache_raw; causal_rows_only_for_per_query_comparison");
-        record(3,"attention_probabilities_operand",window_dense(t->src[1]),t->src[1],1);
-    }
-    if(gdn>=0 && gdn<=4) {
+    if(gdn>=0 && gdn<=1) {
         require(gdn!=3 && t->src[0] && t->src[1] && t->src[2] && t->src[3] && t->src[4] && t->src[5],"actual GDN source binding");
         require(t->src[2]->ne[0]==128 && t->src[2]->ne[1]==16 && t->src[2]->ne[2]==call_count && t->src[2]->ne[3]==1,"actual unchanged GDN token width");
         int32_t snapshots=0;std::memcpy(&snapshots,t->op_params,4);require(snapshots==1,"original GDN final-state policy");
@@ -183,22 +244,29 @@
         }
     };
     one(-1,"model.input_embed");
-    for(int layer=0;layer<=4;layer++) {
-        for(const char *base:{"attn_norm","attn_residual","attn_post_norm","ffn_out","l_out"})one(layer,std::string(base)+"-"+std::to_string(layer));
+    for(int layer=0;layer<=1;layer++) {
+        for(const char *base:{"attn_norm","attn_residual","attn_post_norm","ffn_out","l_out",
+            "linear_attn_qkv_mixed","z","alpha","a_softplus","gate","beta","beta_sigmoid",
+            "conv_output_raw","conv_output_silu","q_conv_predelta","k_conv_predelta","v_conv_predelta",
+            "final_output","linear_attn_out"})one(layer,std::string(base)+"-"+std::to_string(layer));
         projections(layer,{"ffn_gate.weight","ffn_up.weight","ffn_down.weight"});
-        if(layer==3)projections(layer,{"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight"});
-        else projections(layer,{"attn_qkv.weight","ssm_alpha.weight","ssm_beta.weight","attn_gate.weight","ssm_out.weight"});
+        for(const char *base:{"ffn_up","ffn_gate","ffn_swiglu"})one(layer,std::string(base)+"-"+std::to_string(layer));
+        require(occurrences[{layer,"norm-"+std::to_string(layer)}]==3,"all three original RMS producer boundaries");
+        one(layer,"ssm_conv_input");
+        for(const char *role:{"attn_norm.weight","post_attention_norm.weight","ssm_a","ssm_dt.bias","ssm_conv1d.weight","ssm_norm.weight"})
+            one(layer,"learned_op_output::blk."+std::to_string(layer)+"."+role);
+        projections(layer,{"attn_qkv.weight","ssm_alpha.weight","ssm_beta.weight","attn_gate.weight","ssm_out.weight"});
         if(layer!=3) {
             for(const char *stage:{"gdn_q_operand","gdn_k_operand","gdn_v_operand","gdn_gate_operand","gdn_beta_operand","gdn_core_output"})one(layer,stage);
             if(provider_mode=="bulk64" || call_index==0)one(layer,"gdn_prior_state");
             if(provider_mode=="bulk64" || call_index+1==window_count)one(layer,"gdn_proposed_state");
         }
     }
-    for(const char *stage:{"Qcur_normed-3","Kcur_normed-3","kq-3","kq_soft_max-3","kqv-3","attn_pregate-3","attn_gated-3","attention_K_cache_operand","attention_Q_operand","attention_V_cache_operand","attention_softmax_scores_operand","attention_mask_operand","attention_probabilities_operand"})one(3,stage);
+
 }
 static size_t end_observation() {
-    require(observed_layers==std::set<int>({0,1,2,3,4}),"missing first-five-layer observations");
-    require(gdn_calls==4*(provider_mode=="bulk64" ? 1 : window_count),"missing actual GDN operations");
+    require(observed_layers==std::set<int>({0,1}),"missing first-two-layer observations");
+    require(gdn_calls==2*(provider_mode=="bulk64" ? 1 : window_count),"missing actual GDN operations");
     records.flush();require(bool(records),"window final records flush");records.close();return window_bytes;
 }
 static void guard_next(llama_context *plain,llama_context *observed,const std::vector<llama_token> &ids,int end,const fs::path &out) {
@@ -213,8 +281,8 @@
     try {
         require(argc==7,"usage: native_window GGUF ids.u32le aligned_start threads seconds NEW_OUTPUT");
         size_t consumed=0;const std::string start_arg=argv[3];const long start=std::stol(start_arg,&consumed);
-        require(consumed==start_arg.size() && start>=0 && start<=960 && start%64==0,"original1K aligned block start");
-        window_start=int(start);window_count=std::min(64,1023-window_start);
+        require(consumed==start_arg.size() && start>=0 && start<=960 && start%64==0,"bounded early aligned block of original8K prefix");
+        window_start=int(start);window_count=64;
         const int threads=integer(argv[4],8),seconds=integer(argv[5],7200);
         deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
         const uint32_t endian=1;require(*reinterpret_cast<const unsigned char*>(&endian)==1 && sizeof(float)==4,"little-endian F32 host");
@@ -225,7 +293,7 @@
         write_bytes(base/"input_ids.u32le",raw.data(),raw.size());
         llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
         std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
-        require(bool(model) && llama_model_n_layer(model.get())==24 && llama_model_n_embd(model.get())==1024 && llama_vocab_n_tokens(llama_model_get_vocab(model.get()))==248320,"original0.8B geometry");
+        require(bool(model) && llama_model_n_layer(model.get())==24 && llama_model_n_embd(model.get())==2048 && llama_vocab_n_tokens(llama_model_get_vocab(model.get()))==248320,"original2B geometry");
         auto params=llama_context_default_params();params.n_ctx=8320;params.n_batch=64;params.n_ubatch=64;
         params.n_threads=threads;params.n_threads_batch=threads;params.type_k=GGML_TYPE_F32;params.type_v=GGML_TYPE_F32;
         params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;params.offload_kqv=false;params.op_offload=false;params.no_perf=true;params.cb_eval=nullptr;
@@ -249,16 +317,20 @@
         write_bytes(base/"bulk.post-state.bin",after_bulk.data(),after_bulk.size());write_bytes(base/"ar1.post-state.bin",after_serial.data(),after_serial.size());
         params.cb_eval=window_observer;Context observed_bulk(llama_init_from_model(model.get(),params),llama_free);require(bool(observed_bulk),"observed bulk context");
         require(llama_state_set_data(observed_bulk.get(),before_bulk.data(),before_bulk.size())==before_bulk.size(),"same-source bulk prior restore");
-        begin_observation(base,"bulk64");enabled=true;call_index=0;call_count=window_count;occurrences.clear();
+        learned_folder=base/"learned-operands";require(fs::create_directory(learned_folder),"new learned operand directory");
+        require(chmod(learned_folder.c_str(),0700)==0,"learned operand permissions");
+        learned_records.open(learned_folder/"records.jsonl");require(bool(learned_records),"learned metadata open");
+        begin_observation(base,"bulk64");enabled=true;call_index=0;call_count=window_count;occurrences.clear();tensor_ids.clear();
         decode(observed_bulk.get(),ids,window_start,window_count,false);enabled=false;require_current_coverage();
         const size_t bulk_bytes=end_observation();require(state(observed_bulk.get())==after_bulk,"observer changed unchanged-width bulk complete state");
         guard_next(bulk.get(),observed_bulk.get(),ids,window_start+window_count,base/"bulk.next-logits.f32le");
         serial_params.cb_eval=window_observer;Context observed_ar(llama_init_from_model(model.get(),serial_params),llama_free);require(bool(observed_ar),"observed AR1 context");
         require(llama_state_set_data(observed_ar.get(),before_serial.data(),before_serial.size())==before_serial.size(),"same-source AR1 prior restore");
         begin_observation(base,"ar1");call_count=1;enabled=true;
-        for(int at=0;at<window_count;at++){call_index=at;occurrences.clear();decode(observed_ar.get(),ids,window_start+at,1,false);require_current_coverage();}
+        for(int at=0;at<window_count;at++){call_index=at;occurrences.clear();tensor_ids.clear();decode(observed_ar.get(),ids,window_start+at,1,false);require_current_coverage();}
         enabled=false;const size_t ar_bytes=end_observation();require(state(observed_ar.get())==after_serial,"observer changed AR1 complete state");
         guard_next(serial.get(),observed_ar.get(),ids,window_start+window_count,base/"ar1.next-logits.f32le");
+        verify_learned_unchanged();learned_records.close();
         size_t artifact_bytes=0;
         for(const auto &entry:fs::recursive_directory_iterator(base)) {
             require(!entry.is_symlink(),"no capture artifact symlink");
@@ -270,12 +342,12 @@
         }
         std::ofstream done(base/"complete.json");require(bool(done),"window complete record open");
         done<<"{\"status\":\"TRANSPARENT_WINDOW_CAPTURE_NOT_CAUSE_QUALIFICATION\",\"model_path\":"<<quote(fs::canonical(argv[1]).string())
-            <<",\"window_start\":"<<window_start<<",\"window_tokens\":"<<window_count<<",\"original_case_cut\":1024,\"threads\":"<<threads
+            <<",\"hidden_width\":2048,\"selected_layers\":[0,1],\"learned_operand_count\":28,\"learned_operand_bytes\":"<<learned_bytes<<",\"learned_unchanged_after_compute\":true,\"window_start\":"<<window_start<<",\"window_tokens\":"<<window_count<<",\"original_case_cut\":8192,\"threads\":"<<threads
             <<",\"bulk_actual_ubatch\":64,\"ar1_actual_ubatch\":1,\"common_prior_exact\":true,\"bulk_instrumentation_complete_state_exact\":true,\"ar1_instrumentation_complete_state_exact\":true"
             <<",\"both_next_AR1_full248320_and_whole_state_exact\":true,\"post_state_bulk_vs_ar1_equal\":"<<(after_bulk==after_serial?"true":"false")
             <<",\"bulk_payload_bytes\":"<<bulk_bytes<<",\"ar1_payload_bytes\":"<<ar_bytes
             <<",\"precompletion_artifact_bytes\":"<<artifact_bytes
-            <<",\"prefix_policy\":\"unchanged original15x64+63; no arbitrary cut binary-search inference\",\"operator_provider_internal_dispatch_observed\":false}\n";
+            <<",\"prefix_policy\":\"unchanged original8191-prefix127x64+63; early aligned64-row block, no arbitrary-cut inference\",\"operator_provider_internal_dispatch_observed\":false}\n";
         done.flush();require(bool(done),"window complete record write");require(chmod((base/"complete.json").c_str(),0600)==0,"window complete mode");
         std::cerr<<"TRANSPARENT_NATIVE_WINDOW_CAPTURE_COMPLETE start="<<window_start<<" tokens="<<window_count<<" cause_qualified=false\n";
         observed_ar.reset();observed_bulk.reset();serial.reset();bulk.reset();model.reset();llama_backend_free();return 0;
--- a/compare_window.py
+++ b/compare_window.py
@@ -4,12 +4,14 @@
 import hashlib
 import json
 import math
+import os
+import stat
 from pathlib import Path
 import re
 import struct
 import sys
 
-MODE_CAPS = {"bulk64": 128 * 1024 * 1024, "ar1": 192 * 1024 * 1024}
+MODE_CAPS = {"bulk64": 112 * 1024 * 1024, "ar1": 112 * 1024 * 1024}
 FILE_DIGESTS = {}
 
 
@@ -49,7 +51,7 @@
         require(row["mode"] == mode and type(row["call"]) is int
                 and 0 <= row["call"] < (1 if mode == "bulk64" else count)
                 and row["call_tokens"] == (count if mode == "bulk64" else 1), "actual call identity/width")
-        require(type(row["layer"]) is int and -1 <= row["layer"] <= 4
+        require(type(row["layer"]) is int and -1 <= row["layer"] <= 1
                 and isinstance(row["stage"], str) and len(row["stage"]) <= 256
                 and type(row["occurrence"]) is int and 0 <= row["occurrence"] <= 16,
                 "bounded field identity")
@@ -60,6 +62,10 @@
         require(isinstance(shape, list) and len(shape) == 4
                 and all(type(n) is int and n > 0 for n in shape)
                 and math.prod(shape) == row["elements"], "actual dense tensor shape")
+        require(type(row["native_tensor_id"]) is int and 0 < row["native_tensor_id"] < 1000000
+                and len(row["native_source_ids"]) == 10
+                and all(type(value) is int and 0 <= value < 1000000 for value in row["native_source_ids"]),
+                "actual per-call original tensor/operand identifiers")
         axis = row["token_axis"]
         require(type(axis) is int and -1 <= axis <= 3
                 and (axis == -1 or shape[axis] == row["call_tokens"]), "declared source token axis")
@@ -81,12 +87,19 @@
         require(lookup.get((call, layer, stage)) == [0], f"missing/duplicate required field: {call}/{layer}/{stage}")
     for call in range(1 if mode == "bulk64" else count):
         one(call, -1, "model.input_embed")
-        for layer in range(5):
-            for stage in ["attn_norm", "attn_residual", "attn_post_norm", "ffn_out", "l_out"]:
+        for layer in range(2):
+            for stage in ["attn_norm", "attn_residual", "attn_post_norm", "ffn_out", "l_out",
+                          "linear_attn_qkv_mixed", "z", "alpha", "a_softplus", "gate", "beta", "beta_sigmoid",
+                          "conv_output_raw", "conv_output_silu", "q_conv_predelta", "k_conv_predelta", "v_conv_predelta",
+                          "final_output", "linear_attn_out", "ffn_up", "ffn_gate", "ffn_swiglu"]:
                 one(call, layer, f"{stage}-{layer}")
             roles = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
-            roles += (["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"] if layer == 3 else
-                      ["attn_qkv.weight", "ssm_alpha.weight", "ssm_beta.weight", "attn_gate.weight", "ssm_out.weight"])
+            roles += ["attn_qkv.weight", "ssm_alpha.weight", "ssm_beta.weight", "attn_gate.weight", "ssm_out.weight"]
+            require(lookup.get((call, layer, f"norm-{layer}")) == [0, 1, 2], "all three real RMS producer outputs")
+            one(call, layer, "ssm_conv_input")
+            for role in ["attn_norm.weight", "post_attention_norm.weight", "ssm_a", "ssm_dt.bias",
+                         "ssm_conv1d.weight", "ssm_norm.weight"]:
+                one(call, layer, f"learned_op_output::blk.{layer}.{role}")
             for role in roles:
                 for prefix in ["projection_input::", "projection_output::"]:
                     one(call, layer, f"{prefix}blk.{layer}.{role}")
@@ -97,11 +110,7 @@
                     one(call, layer, "gdn_prior_state")
                 if mode == "bulk64" or call + 1 == count:
                     one(call, layer, "gdn_proposed_state")
-        for stage in ["Qcur_normed-3", "Kcur_normed-3", "kq-3", "kq_soft_max-3", "kqv-3",
-                      "attn_pregate-3", "attn_gated-3", "attention_K_cache_operand", "attention_Q_operand",
-                      "attention_V_cache_operand", "attention_softmax_scores_operand", "attention_mask_operand",
-                      "attention_probabilities_operand"]:
-            one(call, 3, stage)
+
 
 
 def field_bytes(folder, row):
@@ -146,6 +155,71 @@
                 first_raw_mismatch=first, first_causal_mismatch=first_causal)
 
 
+def digest_file(path):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
+        before = os.fstat(stream.fileno())
+        require(stat.S_ISREG(before.st_mode), "regular pinned source")
+        h = hashlib.sha256()
+        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
+            h.update(chunk)
+        after = os.fstat(stream.fileno())
+        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
+                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
+                "source changed during hash")
+    return {"bytes": before.st_size, "sha256": h.hexdigest()}
+
+
+def verify_learned(capture, done):
+    here = Path(__file__).resolve().parent
+    descriptors = jread(here / "learned-source-headers.json")
+    model = Path(descriptors["model"])
+    expected_source = {"bytes": descriptors["model_bytes"], "sha256": descriptors["model_sha256"]}
+    require(Path(done["model_path"]).resolve() == model.resolve() and digest_file(model) == expected_source,
+            "independently pinned approved original2B GGUF bytes")
+    ids = bounded(capture / "input_ids.u32le", 32768)
+    require(len(ids) == 32768 and hashlib.sha256(ids).hexdigest() ==
+            "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d", "exact original8192 teacher file")
+    expected = {row["name"]: row for row in descriptors["selected"]}
+    require(len(expected) == 28, "actual header first-two-layer closure")
+    rows = [json.loads(line, object_pairs_hook=unique) for line in
+            bounded(capture / "learned-operands/records.jsonl", 128 * 1024).splitlines()]
+    require(len(rows) == 28, "all actual learned operand descriptors")
+    seen, files, total, hashes = set(), set(), 0, {}
+    with os.fdopen(os.open(model, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
+        for row in rows:
+            name = row["name"]
+            require(name in expected and name not in seen, "unique original learned role")
+            seen.add(name)
+            original = expected[name]
+            require(row["type"] == original["type"] and row["ne"] == original["ne"]
+                    and row["bytes"] == original["bytes"] and 0 < row["bytes"] <= 8 * 1024**2,
+                    "exact original learned dtype/shape/byte geometry")
+            at, content = 0, hashlib.sha256()
+            for chunk in row["chunks"]:
+                require(re.fullmatch(r"learned-[1-9][0-9]?-[0-9]+\.bin", chunk["file"])
+                        and chunk["file"] not in files and type(chunk["offset"]) is int
+                        and chunk["offset"] == at and type(chunk["bytes"]) is int
+                        and chunk["bytes"] == min(1024**2, row["bytes"]-at), "complete bounded raw operand chunks")
+                files.add(chunk["file"])
+                raw = bounded(capture / "learned-operands" / chunk["file"], 1024**2)
+                require(len(raw) == chunk["bytes"], "exact raw source chunk")
+                source.seek(original["absolute_offset"] + at)
+                require(source.read(len(raw)) == raw, "actual physical learned operand equals approved GGUF bytes")
+                if row["type"] == 0:
+                    require(all(bits & 0x7f800000 != 0x7f800000 for (bits,) in struct.iter_unpack("<I", raw)),
+                            "finite actual F32 learned source")
+                content.update(raw)
+                at += len(raw)
+            require(at == row["bytes"], "complete actual packed learned operand")
+            total += at
+            hashes[name] = {"bytes": at, "sha256": content.hexdigest(), "type": row["type"], "ne": row["ne"]}
+    require(seen == set(expected) and total == descriptors["selected_raw_bytes"] == done["learned_operand_bytes"]
+            and total <= 80 * 1024**2, "all28 actual raw learned operands accounted")
+    require(digest_file(model) == expected_source, "original GGUF unchanged after actual readset validation")
+    return {"all28_equal_original_source_bytes": True, "bytes": total, "raw_operands": hashes,
+            "weight_dequantization_or_arithmetic_executed": False}
+
+
 def main():
     p = argparse.ArgumentParser(description=__doc__)
     p.add_argument("--capture", type=Path, required=True)
@@ -158,15 +232,20 @@
             and done["ar1_instrumentation_complete_state_exact"] is True
             and done["both_next_AR1_full248320_and_whole_state_exact"] is True,
             "mandatory unchanged-window/state/next-token transparency guards")
+    require(done["hidden_width"] == 2048 and done["selected_layers"] == [0, 1]
+            and done["original_case_cut"] == 8192 and done["learned_operand_count"] == 28
+            and done["learned_unchanged_after_compute"] is True and done["threads"] == 8,
+            "approved original2B/first-two-layer/8K-prefix source scope")
+    learned = verify_learned(a.capture, done)
     start, count = done["window_start"], done["window_tokens"]
     require(type(start) is int and type(count) is int and start in range(0, 961, 64)
-            and count == min(64, 1023 - start), "original15x64+63 segmentation")
+            and count == 64, "early aligned64-row block from original127x64+63 prefix")
     artifact_bytes = 0
     for path in a.capture.rglob("*"):
         require(not path.is_symlink() and (path.is_dir() or path.is_file()), "regular capture tree")
         if path.is_file() and path != a.capture / "complete.json":
             artifact_bytes += path.stat().st_size
-            require(artifact_bytes <= 512 * 1024 * 1024 - 128 * 1024, "whole capture artifact cap")
+            require(artifact_bytes <= 480 * 1024 * 1024 - 128 * 1024, "whole capture artifact cap")
     require(artifact_bytes == done["precompletion_artifact_bytes"], "complete artifact byte accounting")
     snapshots = ["common.prior-state.bin", "bulk.post-state.bin", "ar1.post-state.bin"]
     require(sum((a.capture / name).stat().st_size for name in snapshots) <= 136 * 1024 * 1024,
@@ -180,6 +259,23 @@
         require(key[0] == 0, "one unchanged actual bulk call")
         left = field_bytes(a.capture / "bulk64", row)
         layer, stage, occurrence = key[1:]
+        if stage == "ssm_conv_input":
+            require(row["dense_ne"] == [count+3, 6144, 1, 1] and row["native_op"] == "CONCAT",
+                    "actual full conv history/current sequence")
+            for at in range(count):
+                other_key = (at, layer, stage, occurrence)
+                require(other_key in serial, "all original AR1 conv causal operands")
+                other = serial[other_key]
+                require(other["dense_ne"] == [4, 6144, 1, 1], "original four-tap AR1 operand")
+                right = field_bytes(a.capture / "ar1", other)
+                # Byte-only actual-history codec; no convolution or generated expected arithmetic.
+                taps = b"".join(left[(channel*(count+3)+at)*4:(channel*(count+3)+at+4)*4]
+                                for channel in range(6144))
+                output.append(dict(layer=layer, stage=stage+"::actual4taps", occurrence=occurrence,
+                                   position=start+at, causal_scope="all four actual history/current taps",
+                                   **differences(taps, right)))
+                used_ar.add(other_key)
+            continue
         if row["token_axis"] == -1:
             # Caches contain the whole original block in bulk but only the
             # partial causal frontier in earlier AR1 calls. Compare whole cache
@@ -216,15 +312,14 @@
     # Intermediate AR cache operands are retained and intentionally not equal to
     # bulk's final whole-block cache. They are not silently treated as results.
     remaining = [key for key in serial if key not in used_ar]
-    require(all(serial[key]["stage"] in ["attention_K_cache_operand", "attention_V_cache_operand"]
-                and key[0] < count - 1 for key in remaining), "unmatched observed AR1 field")
+    require(not remaining, "unmatched observed first-two-layer AR1 field")
     report = dict(status="WINDOW_FIELD_COMPARISON_DIAGNOSTIC_ONLY", window_start=start, window_tokens=count,
         fields=output, first_observed_causal_scope_difference=next((r for r in output if r["causal_unequal_words"]), None),
         different_field_rows=sum(r["causal_unequal_words"] > 0 for r in output),
         unpaired_intermediate_actual_cache_operands=len(remaining),
         bulk_records_sha256=bulk_hash, ar1_records_sha256=serial_hash,
         complete_sha256=hashlib.sha256(bounded(a.capture / "complete.json", 128 * 1024)).hexdigest(),
-        actual_field_sha256=FILE_DIGESTS,
+        actual_field_sha256=FILE_DIGESTS, learned_operands=learned,
         tolerance="none; signed zero preserved; future causal-mask-excluded QK columns reported separately",
         cause_qualified=False, internal_dispatch_observed=False,
         scope="native unchanged-width bulk versus native AR1 actual tensors; no Rust parity, long-context eligibility or causal-operator attribution automatically inferred")

````

