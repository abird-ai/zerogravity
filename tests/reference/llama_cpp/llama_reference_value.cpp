// Test-only native attention value capture and canonical-padding audit.
// Linked to the pinned original CPU library, never to Zerogravity.
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

static FILE *capture = nullptr;
static int position = 0;
static bool audit = false;
static bool gated = false;
static unsigned captured = 0, audited = 0;
static std::set<int> seen, checked;

[[noreturn]] static void fail(const char *message) {
    std::fprintf(stderr, "REFERENCE_VALUE_ERROR %s position=%d\n", message, position);
    std::exit(2);
}

static int named_layer(const ggml_tensor *tensor, const char *prefix) {
    if (!tensor) {
        return -1;
    }
    for (int layer : {3, 7, 11, 15, 19, 23}) {
        if (std::string(tensor->name) == std::string(prefix) + std::to_string(layer)) {
            return layer;
        }
    }
    return -1;
}

static bool wanted() {
    if (!audit) {
        return true;
    }
    for (int tokens : {1, 31, 32, 33, 224, 225, 255, 256, 257}) {
        if (position + 1 == tokens) {
            return true;
        }
    }
    return false;
}

static void require_positive_zero(const std::vector<unsigned char> &bytes, size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(uint32_t)) {
        fail("padding tensor stride");
    }
    uint32_t bits;
    std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
    if (bits != 0) {
        fail("nonzero/negative canonical padding");
    }
}

static void audit_padding(const ggml_tensor *tensor, int layer) {
    if (!checked.insert(layer).second) {
        fail("duplicate value matmul");
    }
    const auto *value = tensor->src[0];
    const auto *probability = tensor->src[1];
    const int valid = position + 1;
    const int padded = (valid + 255) / 256 * 256;
    if (!value || !probability || value->type != GGML_TYPE_F32 ||
        probability->type != GGML_TYPE_F32 || value->ne[0] != padded ||
        value->ne[1] != 256 || value->ne[2] != 2 || value->ne[3] != 1 ||
        probability->ne[0] != padded || probability->ne[1] != 1 ||
        probability->ne[2] != 8 || probability->ne[3] != 1) {
        fail("source padding geometry");
    }
    std::vector<unsigned char> values(ggml_nbytes(value));
    std::vector<unsigned char> probabilities(ggml_nbytes(probability));
    ggml_backend_tensor_get(value, values.data(), 0, values.size());
    ggml_backend_tensor_get(probability, probabilities.data(), 0, probabilities.size());
    size_t zeros = 0;
    for (int token = valid; token < padded; ++token) {
        for (int head = 0; head < 2; ++head) {
            for (int channel = 0; channel < 256; ++channel) {
                require_positive_zero(values, token * value->nb[0] + channel * value->nb[1] +
                                              head * value->nb[2]);
                ++zeros;
            }
        }
        for (int head = 0; head < 8; ++head) {
            require_positive_zero(probabilities,
                                  token * probability->nb[0] + head * probability->nb[2]);
            ++zeros;
        }
    }
    std::fprintf(stderr,
                 "SOURCE_VALUE_PADDING_PASS tokens=%d layer=%d padded=%d "
                 "checked_zero_fp32=%zu V_ne=%lld,%lld,%lld,%lld P_ne=%lld,%lld,%lld,%lld "
                 "V_nb=%zu,%zu,%zu,%zu P_nb=%zu,%zu,%zu,%zu\n",
                 valid, layer, padded, zeros,
                 static_cast<long long>(value->ne[0]), static_cast<long long>(value->ne[1]),
                 static_cast<long long>(value->ne[2]), static_cast<long long>(value->ne[3]),
                 static_cast<long long>(probability->ne[0]),
                 static_cast<long long>(probability->ne[1]),
                 static_cast<long long>(probability->ne[2]),
                 static_cast<long long>(probability->ne[3]),
                 value->nb[0], value->nb[1], value->nb[2], value->nb[3],
                 probability->nb[0], probability->nb[1], probability->nb[2], probability->nb[3]);
    ++audited;
}

static bool observe(ggml_tensor *tensor, bool ask, void *) {
    if (!tensor || !wanted()) {
        return false;
    }
    const int layer = named_layer(tensor, gated ? "attn_gated-" : "attn_pregate-");
    const int mat_layer = tensor->op == GGML_OP_MUL_MAT
                              ? named_layer(tensor->src[1], "kq_soft_max-")
                              : -1;
    if (layer < 0 && (!audit || mat_layer < 0)) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (audit && mat_layer >= 0) {
        audit_padding(tensor, mat_layer);
    }
    if (layer >= 0) {
        if (!seen.insert(layer).second || tensor->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(tensor) || ggml_nelements(tensor) != 2048) {
            fail("native pregate geometry/order");
        }
        std::vector<float> values(2048);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        std::fprintf(capture, "{\"position\":%d,\"layer\":%d,\"values\":[", position, layer);
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) {
                std::fputc(',', capture);
            }
            if (!std::isfinite(values[i])) {
                fail("nonfinite native pregate");
            }
            if (values[i] == 0.0f) {
                std::fputs(std::signbit(values[i]) ? "-0.0" : "0.0", capture);
            } else {
                std::fprintf(capture, "%.9g", values[i]);
            }
        }
        std::fputs("]}\n", capture);
        ++captured;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: value_capture GGUF output.jsonl [--padding-audit|--gated]\n");
        return 2;
    }
    audit = argc == 4 && std::string(argv[3]) == "--padding-audit";
    gated = argc == 4 && std::string(argv[3]) == "--gated";
    if (argc == 4 && !audit && !gated) {
        return 2;
    }
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        return 2;
    }
    auto params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = 128;
    params.n_ubatch = 128;
    params.n_threads = 8;
    params.n_threads_batch = 8;
    params.type_k = GGML_TYPE_F32;
    params.type_v = GGML_TYPE_F32;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.cb_eval = observe;
    params.no_perf = true;
    auto *ctx = llama_init_from_model(model, params);
    if (!ctx) {
        return 2;
    }
    if (llama_n_ctx(ctx) < 512) {
        fail("source context capacity below padding audit boundary");
    }
    capture = std::fopen(argv[2], "wb");
    if (!capture) {
        return 2;
    }
    const int ids[] = {11, 353, 17, 42};
    unsigned total = 0, total_audits = 0;
    for (int i = 0; i < (audit ? 257 : 4); ++i) {
        captured = 0;
        audited = 0;
        seen.clear();
        checked.clear();
        llama_token token = ids[i % 4];
        auto batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) {
            fail("source decode");
        }
        if (wanted() && (captured != 6 || (audit && audited != 6))) {
            fail("incomplete native capture/audit");
        }
        total += captured;
        total_audits += audited;
        ++position;
    }
    if (std::fclose(capture) != 0) {
        fail("capture file close");
    }
    std::fprintf(stderr, "INDEPENDENT_VALUE_CAPTURED=%u padding_audit=%d "
                         "padding_records=%u n_ctx=%u gated=%d\n",
                 total, audit ? 1 : 0, total_audits, llama_n_ctx(ctx), gated ? 1 : 0);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
