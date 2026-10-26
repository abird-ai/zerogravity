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
