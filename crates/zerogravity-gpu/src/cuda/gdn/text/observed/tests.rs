use super::{CAPTURES, GdnKernelRecipe};
use zerogravity_core::{Error, Result};

fn specialize_packed_conv(cache_lines: u32) -> String {
    super::super::parameterized_conv::SOURCE
        .replace(
            "// Reversible candidate SHA256: 2165f48fc19f67b7d6570086cf179feb5d00717f5a816772ee32db716df655c9\n",
            "",
        )
        .replace(
            "struct alignas(8) ZgGdnConvScalarWords {\n    unsigned long long words[3];\n};\nstatic_assert(sizeof(ZgGdnConvScalarWords) == 24, \"GDN Conv scalar ABI\");\nstatic_assert(alignof(ZgGdnConvScalarWords) == 8, \"GDN Conv scalar alignment\");\n",
            "",
        )
        .replace(
            "void zg_gdn_conv_prefill_parameterized_v1(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, void* zg_arg5, void* zg_arg6, void* zg_arg7, void* zg_arg8, ZgGdnConvScalarWords zg_scalars)",
            "void zg_vllm_a905e193ffad4e19(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, void* zg_arg5, void* zg_arg6, void* zg_arg7, void* zg_arg8)",
        )
        .replace(
            "\\tmov.b64 %%zg_orig_rd18, %10;\\n",
            "\\tmov.b64 %%zg_orig_rd18, 8192;\\n",
        )
        .replace(
            "\\tmov.b32 %%zg_orig_r10, %9;\\n",
            &format!("\\tmov.b32 %%zg_orig_r10, {cache_lines};\\n"),
        )
        .replace(
            "\\tmov.b64 %%zg_orig_rd19, %11;\\n",
            "\\tmov.b64 %%zg_orig_rd19, 6144;\\n",
        )
        .replace(
            ", \"r\"(static_cast<unsigned int>(zg_scalars.words[0])), \"l\"(zg_scalars.words[1]), \"l\"(zg_scalars.words[2]) : \"memory\");",
            " : \"memory\");",
        )
}

#[test]
fn packed_conv_reversibly_specializes_to_every_retained_prefill_source() {
    use sha2::{Digest, Sha256};

    const LEGACY: [(u32, &str); 10] = [
        (492, include_str!("../qwen08_47/conv.cu")),
        (492, include_str!("../qwen08_57/conv.cu")),
        (492, include_str!("../qwen08_62/conv.cu")),
        (492, include_str!("../qwen08_67/conv.cu")),
        (492, include_str!("../qwen08_72/conv.cu")),
        (157, include_str!("../qwen2b_47/conv.cu")),
        (157, include_str!("../qwen2b_57/conv.cu")),
        (157, include_str!("../qwen2b_62/conv.cu")),
        (157, include_str!("../qwen2b_67/conv.cu")),
        (157, include_str!("../qwen2b_72/conv.cu")),
    ];
    for (cache_lines, legacy) in LEGACY {
        assert_eq!(specialize_packed_conv(cache_lines), legacy);
    }
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(super::super::parameterized_conv::SOURCE.as_bytes())
        ),
        super::super::parameterized_conv::PACKED_SOURCE_SHA256
    );
    assert_ne!(
        super::super::parameterized_conv::PACKED_SOURCE_SHA256,
        super::super::parameterized_conv::CANDIDATE_SOURCE_SHA256
    );
}

fn specialize_packed_prep(tokens: u32) -> String {
    super::super::parameterized_prefill::PREP_SOURCE
        .replace(
            "// Reversible candidate SHA256: e71a9758bc2d18a1d9301b3294650c096d738bec80d7630e56f12299d298fc09\n",
            "",
        )
        .replace(
            "struct alignas(8) ZgGdnPrepScalarWords {\n    unsigned long long words[7];\n};\nstatic_assert(sizeof(ZgGdnPrepScalarWords) == 56, \"GDN Prep scalar ABI\");\nstatic_assert(alignof(ZgGdnPrepScalarWords) == 8, \"GDN Prep scalar alignment\");\n",
            "",
        )
        .replace(
            "void zg_gdn_prep_prefill_parameterized_v1(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, void* zg_arg5, void* zg_arg6, void* zg_arg7, void* zg_arg8, void* zg_arg9, ZgGdnPrepScalarWords zg_scalars)",
            "void zg_vllm_78ff2ff19f5099b9(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, void* zg_arg5, void* zg_arg6, void* zg_arg7, void* zg_arg8, void* zg_arg9)",
        )
        .replace("\\tmov.b32 %%zg_orig_r14, %10;\\n", "\\tmov.b32 %%zg_orig_r14, 6144;\\n")
        .replace("\\tmov.b32 %%zg_orig_r15, %11;\\n", "\\tmov.b32 %%zg_orig_r15, 16;\\n")
        .replace("\\tmov.b32 %%zg_orig_r16, %12;\\n", "\\tmov.b32 %%zg_orig_r16, 16;\\n")
        .replace("\\tmov.b32 %%zg_orig_r17, %13;\\n", "\\tmov.b32 %%zg_orig_r17, 2048;\\n")
        .replace("\\tmov.b32 %%zg_orig_r18, %14;\\n", "\\tmov.b32 %%zg_orig_r18, 2048;\\n")
        .replace("\\tmov.b32 %%zg_orig_r19, %15;\\n", "\\tmov.b32 %%zg_orig_r19, 2048;\\n")
        .replace(
            "\\tmov.b32 %%zg_orig_r20, %16;\\n",
            &format!("\\tmov.b32 %%zg_orig_r20, {tokens};\\n"),
        )
        .replace(
            ", \"r\"(static_cast<unsigned int>(zg_scalars.words[0])), \"r\"(static_cast<unsigned int>(zg_scalars.words[1])), \"r\"(static_cast<unsigned int>(zg_scalars.words[2])), \"r\"(static_cast<unsigned int>(zg_scalars.words[3])), \"r\"(static_cast<unsigned int>(zg_scalars.words[4])), \"r\"(static_cast<unsigned int>(zg_scalars.words[5])), \"r\"(static_cast<unsigned int>(zg_scalars.words[6])) : \"memory\");",
            " : \"memory\");",
        )
}

fn specialize_pointer_source(source: &str, provenance: &str, from: &str, to: &str) -> String {
    source.replace(provenance, "").replace(from, to)
}

#[test]
fn parameterized_prefill_reversibly_specializes_to_all_retained_sources_and_labels() {
    use sha2::{Digest, Sha256};

    const PREP: [(u32, &str); 10] = [
        (47, include_str!("../qwen08_47/prep.cu")),
        (57, include_str!("../qwen08_57/prep.cu")),
        (62, include_str!("../qwen08_62/prep.cu")),
        (67, include_str!("../qwen08_67/prep.cu")),
        (72, include_str!("../qwen08_72/prep.cu")),
        (47, include_str!("../qwen2b_47/prep.cu")),
        (57, include_str!("../qwen2b_57/prep.cu")),
        (62, include_str!("../qwen2b_62/prep.cu")),
        (67, include_str!("../qwen2b_67/prep.cu")),
        (72, include_str!("../qwen2b_72/prep.cu")),
    ];
    for (tokens, legacy) in PREP {
        assert_eq!(specialize_packed_prep(tokens), legacy);
    }

    const CUMSUM: [&str; 10] = [
        include_str!("../qwen08_47/cumsum.cu"),
        include_str!("../qwen08_57/cumsum.cu"),
        include_str!("../qwen08_62/cumsum.cu"),
        include_str!("../qwen08_67/cumsum.cu"),
        include_str!("../qwen08_72/cumsum.cu"),
        include_str!("../qwen2b_47/cumsum.cu"),
        include_str!("../qwen2b_57/cumsum.cu"),
        include_str!("../qwen2b_62/cumsum.cu"),
        include_str!("../qwen2b_67/cumsum.cu"),
        include_str!("../qwen2b_72/cumsum.cu"),
    ];
    let cumsum = specialize_pointer_source(
        super::super::parameterized_prefill::CUMSUM_SOURCE,
        "// Reversible dead-T candidate SHA256: dfb31a1a569028c2775791b7d9e75230fac31bd8009ec977aebd6c595e05a311\n",
        "void zg_gdn_cumsum_prefill_parameterized_v1(",
        "void zg_vllm_a2ad2ccfa6e953f5(",
    );
    assert!(CUMSUM.into_iter().all(|legacy| cumsum == legacy));

    const SOLVE: [(&str, &str); 10] = [
        (
            "zg_vllm_116d2ffe705b6643",
            include_str!("../qwen08_47/solve.cu"),
        ),
        (
            "zg_vllm_b58e94d8cfdcf47d",
            include_str!("../qwen08_57/solve.cu"),
        ),
        (
            "zg_vllm_116d2ffe705b6643",
            include_str!("../qwen08_62/solve.cu"),
        ),
        (
            "zg_vllm_64d5095f294cb1c1",
            include_str!("../qwen08_67/solve.cu"),
        ),
        (
            "zg_vllm_b58e94d8cfdcf47d",
            include_str!("../qwen08_72/solve.cu"),
        ),
        (
            "zg_vllm_b58e94d8cfdcf47d",
            include_str!("../qwen2b_47/solve.cu"),
        ),
        (
            "zg_vllm_116d2ffe705b6643",
            include_str!("../qwen2b_57/solve.cu"),
        ),
        (
            "zg_vllm_116d2ffe705b6643",
            include_str!("../qwen2b_62/solve.cu"),
        ),
        (
            "zg_vllm_b58e94d8cfdcf47d",
            include_str!("../qwen2b_67/solve.cu"),
        ),
        (
            "zg_vllm_116d2ffe705b6643",
            include_str!("../qwen2b_72/solve.cu"),
        ),
    ];
    let provenance = "// Reversible dead-T candidate SHA256s: ffd4eeb0f041223092c6a198d70eb2eb1d3cf4fa8aff3166f85a42d83f05a9b6,00860d3b61c543fa9f9a5223d6a4a62aa16dd6e01eaa56105c538263c017d6e1,7178fd11687f132ffc6941a31027339aa6c997c95e16082694f8925e32be821e\n";
    for (entry, legacy) in SOLVE {
        assert_eq!(
            specialize_pointer_source(
                super::super::parameterized_prefill::SOLVE_SOURCE,
                provenance,
                "zg_gdn_solve_prefill_parameterized_v1",
                entry,
            ),
            legacy
        );
    }

    for (source, expected) in [
        (
            super::super::parameterized_prefill::PREP_SOURCE,
            super::super::parameterized_prefill::PREP_PACKED_SOURCE_SHA256,
        ),
        (
            super::super::parameterized_prefill::CUMSUM_SOURCE,
            super::super::parameterized_prefill::CUMSUM_POINTER_SOURCE_SHA256,
        ),
        (
            super::super::parameterized_prefill::SOLVE_SOURCE,
            super::super::parameterized_prefill::SOLVE_POINTER_SOURCE_SHA256,
        ),
    ] {
        assert_eq!(format!("{:x}", Sha256::digest(source.as_bytes())), expected);
    }
}

fn remove_packed_scalar_abi(
    source: &str,
    provenance: &str,
    declaration: &str,
    parameterized_formal: &str,
    original_formal: &str,
    scalar_operands: &str,
) -> String {
    source
        .replace(provenance, "")
        .replace(declaration, "")
        .replace(parameterized_formal, original_formal)
        .replace(scalar_operands, "")
}

#[test]
fn parameterized_gated_reversibly_specializes_to_all_retained_sources() {
    use sha2::{Digest, Sha256};

    let declaration = "struct alignas(8) ZgGdnGatedScalarWords {\n    unsigned long long words[5];\n};\nstatic_assert(sizeof(ZgGdnGatedScalarWords) == 40, \"ZgGdnGatedScalarWords ABI\");\nstatic_assert(alignof(ZgGdnGatedScalarWords) == 8, \"ZgGdnGatedScalarWords alignment\");\n";
    let formal = "void zg_gdn_gated_prefill_parameterized_v1(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, ZgGdnGatedScalarWords zg_scalars)";
    let original = "void zg_vllm_820c1a61c99ce579(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4)";
    let operands = ", \"r\"(static_cast<unsigned int>(zg_scalars.words[0])), \"r\"(static_cast<unsigned int>(zg_scalars.words[1])), \"r\"(static_cast<unsigned int>(zg_scalars.words[2])), \"r\"(static_cast<unsigned int>(zg_scalars.words[3])), \"r\"(static_cast<unsigned int>(zg_scalars.words[4]))";
    let provenance = format!(
        "// Reversible candidate SHA256: {}\n",
        super::super::parameterized_gated::PREFILL_CANDIDATE_SOURCE_SHA256
    );
    let base = remove_packed_scalar_abi(
        super::super::parameterized_gated::PREFILL_SOURCE,
        &provenance,
        declaration,
        formal,
        original,
        operands,
    );
    const LEGACY: [(u32, &str); 10] = [
        (752, include_str!("../qwen08_47/gated.cu")),
        (912, include_str!("../qwen08_57/gated.cu")),
        (992, include_str!("../qwen08_62/gated.cu")),
        (1072, include_str!("../qwen08_67/gated.cu")),
        (1152, include_str!("../qwen08_72/gated.cu")),
        (752, include_str!("../qwen2b_47/gated.cu")),
        (912, include_str!("../qwen2b_57/gated.cu")),
        (992, include_str!("../qwen2b_62/gated.cu")),
        (1072, include_str!("../qwen2b_67/gated.cu")),
        (1152, include_str!("../qwen2b_72/gated.cu")),
    ];
    for (m, legacy) in LEGACY {
        let specialized = base
            .replace(
                "\\tmov.b32 %%zg_orig_r34, %5;\\n",
                "\\tmov.b32 %%zg_orig_r34, 128;\\n",
            )
            .replace(
                "\\tmov.b32 %%zg_orig_r36, %6;\\n",
                "\\tmov.b32 %%zg_orig_r36, 128;\\n",
            )
            .replace(
                "\\tmov.b32 %%zg_orig_r38, %7;\\n",
                "\\tmov.b32 %%zg_orig_r38, 128;\\n",
            )
            .replace(
                "\\tmov.b32 %%zg_orig_r40, %8;\\n",
                &format!("\\tmov.b32 %%zg_orig_r40, {m};\\n"),
            )
            .replace(
                "\\tmov.b32 %%zg_orig_r42, %9;\\n",
                "\\tmov.b32 %%zg_orig_r42, 897988541;\\n",
            );
        assert_eq!(specialized, legacy);
    }
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(super::super::parameterized_gated::PREFILL_SOURCE.as_bytes())
        ),
        super::super::parameterized_gated::PREFILL_PACKED_SOURCE_SHA256
    );
}

#[test]
fn parameterized_cached_sources_reverse_to_both_retained_profiles() {
    use sha2::{Digest, Sha256};

    let conv_provenance = format!(
        "// Reversible candidate SHA256: {}\n",
        super::super::parameterized_cached::CONV_CANDIDATE_SOURCE_SHA256
    );
    let conv = remove_packed_scalar_abi(
        super::super::parameterized_cached::CONV_SOURCE,
        &conv_provenance,
        "struct alignas(8) ZgGdnCachedConvScalarWords {\n    unsigned long long words[4];\n};\nstatic_assert(sizeof(ZgGdnCachedConvScalarWords) == 32, \"ZgGdnCachedConvScalarWords ABI\");\nstatic_assert(alignof(ZgGdnCachedConvScalarWords) == 8, \"ZgGdnCachedConvScalarWords alignment\");\n",
        "void zg_gdn_conv_cached_parameterized_v1(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, ZgGdnCachedConvScalarWords zg_scalars)",
        "void zg_vllm_c8f85203ea21f242(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4)",
        ", \"r\"(static_cast<unsigned int>(zg_scalars.words[0])), \"r\"(static_cast<unsigned int>(zg_scalars.words[1])), \"l\"(zg_scalars.words[2]), \"l\"(zg_scalars.words[3])",
    );
    for (cache_lines, legacy) in [
        (492, include_str!("../qwen08_cached/conv.cu")),
        (157, include_str!("../qwen2b_cached/conv.cu")),
    ] {
        assert_eq!(
            conv.replace(
                "\\tmov.b32 %%zg_orig_r3, %5;\\n",
                "\\tmov.b32 %%zg_orig_r3, 1;\\n"
            )
            .replace(
                "\\tmov.b32 %%zg_orig_r2, %6;\\n",
                &format!("\\tmov.b32 %%zg_orig_r2, {cache_lines};\\n")
            )
            .replace(
                "\\tmov.b64 %%zg_orig_rd6, %7;\\n",
                "\\tmov.b64 %%zg_orig_rd6, 1;\\n"
            ),
            legacy
        );
    }

    let decode_provenance = format!(
        "// Reversible candidate SHA256: {}\n",
        super::super::parameterized_cached::DECODE_CANDIDATE_SOURCE_SHA256
    );
    let decode = remove_packed_scalar_abi(
        super::super::parameterized_cached::DECODE_SOURCE,
        &decode_provenance,
        "struct alignas(8) ZgGdnCachedDecodeScalarWords {\n    unsigned long long words[1];\n};\nstatic_assert(sizeof(ZgGdnCachedDecodeScalarWords) == 8, \"ZgGdnCachedDecodeScalarWords ABI\");\nstatic_assert(alignof(ZgGdnCachedDecodeScalarWords) == 8, \"ZgGdnCachedDecodeScalarWords alignment\");\n",
        "void zg_gdn_decode_cached_parameterized_v1(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, void* zg_arg5, void* zg_arg6, void* zg_arg7, void* zg_arg8, ZgGdnCachedDecodeScalarWords zg_scalars)",
        "void zg_vllm_ebe262b1b494d65e(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, void* zg_arg5, void* zg_arg6, void* zg_arg7, void* zg_arg8)",
        ", \"r\"(static_cast<unsigned int>(zg_scalars.words[0]))",
    )
    .replace("\\tmov.b32 %%zg_orig_r51, %10;\\n", "\\tmov.b32 %%zg_orig_r51, 1035273459;\\n");
    assert_eq!(decode, include_str!("../qwen08_cached/decode.cu"));
    assert_eq!(decode, include_str!("../qwen2b_cached/decode.cu"));

    let gated_provenance = format!(
        "// Reversible candidate SHA256: {}\n",
        super::super::parameterized_gated::CACHED_CANDIDATE_SOURCE_SHA256
    );
    let gated = remove_packed_scalar_abi(
        super::super::parameterized_gated::CACHED_SOURCE,
        &gated_provenance,
        "struct alignas(8) ZgGdnCachedGatedScalarWords {\n    unsigned long long words[5];\n};\nstatic_assert(sizeof(ZgGdnCachedGatedScalarWords) == 40, \"ZgGdnCachedGatedScalarWords ABI\");\nstatic_assert(alignof(ZgGdnCachedGatedScalarWords) == 8, \"ZgGdnCachedGatedScalarWords alignment\");\n",
        "void zg_gdn_gated_cached_parameterized_v1(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, ZgGdnCachedGatedScalarWords zg_scalars)",
        "void zg_vllm_c15586587b4a279e(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4)",
        ", \"r\"(static_cast<unsigned int>(zg_scalars.words[0])), \"r\"(static_cast<unsigned int>(zg_scalars.words[1])), \"r\"(static_cast<unsigned int>(zg_scalars.words[2])), \"r\"(static_cast<unsigned int>(zg_scalars.words[3])), \"r\"(static_cast<unsigned int>(zg_scalars.words[4]))",
    )
    .replace("\\tmov.b32 %%zg_orig_r16, %5;\\n", "\\tmov.b32 %%zg_orig_r16, 128;\\n")
    .replace("\\tmov.b32 %%zg_orig_r18, %6;\\n", "\\tmov.b32 %%zg_orig_r18, 128;\\n")
    .replace("\\tmov.b32 %%zg_orig_r20, %7;\\n", "\\tmov.b32 %%zg_orig_r20, 128;\\n")
    .replace("\\tmov.b32 %%zg_orig_r22, %8;\\n", "\\tmov.b32 %%zg_orig_r22, 16;\\n")
    .replace("\\tmov.b32 %%zg_orig_r23, %9;\\n", "\\tmov.b32 %%zg_orig_r23, 897988541;\\n");
    assert_eq!(gated, include_str!("../qwen08_cached/gated.cu"));
    assert_eq!(gated, include_str!("../qwen2b_cached/gated.cu"));

    for (source, expected) in [
        (
            super::super::parameterized_cached::CONV_SOURCE,
            super::super::parameterized_cached::CONV_PACKED_SOURCE_SHA256,
        ),
        (
            super::super::parameterized_cached::DECODE_SOURCE,
            super::super::parameterized_cached::DECODE_PACKED_SOURCE_SHA256,
        ),
        (
            super::super::parameterized_gated::CACHED_SOURCE,
            super::super::parameterized_gated::CACHED_PACKED_SOURCE_SHA256,
        ),
    ] {
        assert_eq!(format!("{:x}", Sha256::digest(source.as_bytes())), expected);
    }
}

impl GdnKernelRecipe {
    pub(in crate::cuda::gdn) fn fixture(
        catalogue: usize,
        input_tokens: usize,
        is_decode: bool,
    ) -> Result<Self> {
        let cell = catalogue
            .checked_mul(6)
            .and_then(|base| {
                CAPTURES
                    .get(base..base + 6)?
                    .iter()
                    .position(|c| c.input_tokens == input_tokens && c.is_decode == is_decode)
                    .map(|i| base + i)
            })
            .ok_or_else(|| {
                Error::Unsupported("actual original text fixture tuple absent".into())
            })?;
        Self::bind(CAPTURES[cell].recipe_sha256, input_tokens, is_decode)
    }
}
