// Test-only original CPU MUL_MAT fixture for the observable sign of padded zero.
// No model or Zerogravity runtime dependency. Use the pinned Haswell CPU closure.
#include "llama.h"
#include "ggml-alloc.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static uint32_t bits(float value) {
    uint32_t result;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static float valid_only_value(int tokens, float minimum, float probability) {
    std::array<std::array<float, 8>, 4> sums{};
    for (int token = 0; token < tokens; ++token) {
        const int lane = token % 32;
        auto &sum = sums[lane / 8][lane % 8];
        sum = std::fma(-minimum, probability, sum);
    }
    std::array<float, 8> reduced{};
    for (int lane = 0; lane < 8; ++lane) {
        reduced[lane] = (sums[0][lane] + sums[2][lane]) + (sums[1][lane] + sums[3][lane]);
    }
    const std::array<float, 4> halves = {
        reduced[0] + reduced[4], reduced[1] + reduced[5],
        reduced[2] + reduced[6], reduced[3] + reduced[7],
    };
    return (halves[0] + halves[1]) + (halves[2] + halves[3]);
}

int main() {
    llama_backend_init();
    auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        return 2;
    }
    const float minimum = std::numeric_limits<float>::denorm_min();
    if (bits(minimum) != 1) {
        return 2;
    }
    unsigned cases = 0;
    for (int tokens : {1, 31, 32, 33, 224, 225, 255, 256, 257}) {
        const int padded = (tokens + 255) / 256 * 256;
        auto *ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        if (!ctx) {
            return 2;
        }
        auto *value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, 1);
        auto *probability = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, 1);
        auto *output = ggml_mul_mat(ctx, value, probability);
        if (!ggml_prec_set_acc(output, GGML_PREC_F32)) {
            return 2;
        }
        auto *graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            return 2;
        }
        std::vector<float> values(padded, 0.0f);
        std::vector<float> probabilities(padded, 0.0f);
        const float weight = 1.0f / static_cast<float>(tokens);
        for (int token = 0; token < tokens; ++token) {
            values[token] = -minimum;
            probabilities[token] = weight;
        }
        ggml_backend_tensor_set(value, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(probability, probabilities.data(), 0,
                                probabilities.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            return 2;
        }
        float result;
        ggml_backend_tensor_get(output, &result, 0, sizeof(result));
        const uint32_t actual = bits(result);
        const uint32_t valid_only = bits(valid_only_value(tokens, minimum, weight));
        // At T=1 the weight is one, so the negative minimum subnormal survives.
        // Other weights round each negative product to zero. Padding changes
        // the zero sign only when every reduction lane would otherwise be -0.
        const uint32_t expected = tokens == 1 ? 0x80000001u :
                                  tokens == 256 ? 0x80000000u : 0u;
        const uint32_t expected_valid_only = tokens == 1 ? 0x80000001u :
                                             tokens >= 32 ? 0x80000000u : 0u;
        if (actual != expected || valid_only != expected_valid_only) {
            std::fprintf(stderr,
                         "NATIVE_PADDED_VALUE_SIGN_FAIL tokens=%d padded=%d "
                         "bits=%08x expected=%08x valid_only=%08x expected_valid_only=%08x\n",
                         tokens, padded, static_cast<unsigned>(actual),
                         static_cast<unsigned>(expected), static_cast<unsigned>(valid_only),
                         static_cast<unsigned>(expected_valid_only));
            return 2;
        }
        std::printf("NATIVE_PADDED_VALUE_SIGN tokens=%d padded=%d bits=%08x "
                    "valid_only_bits=%08x expected_bits_asserted=1\n",
                    tokens, padded, static_cast<unsigned>(actual),
                    static_cast<unsigned>(valid_only));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ++cases;
    }
    ggml_backend_free(backend);
    llama_backend_free();
    std::printf("INDEPENDENT_SOURCE_VALUE_PADDING_FIXTURE_PASS cases=%u "
                "native_padded_matmul=1 signed_zeros_strict=1\n", cases);
}
