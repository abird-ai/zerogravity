//! Add bounds only to the existing Mean body; legacy CUDA bytes stay unchanged.
use zerogravity_core::{Error, Result};

const LOADS: &str = "  for (unsigned int index = x; index < ZG_COLUMNS / 4; index += ZG_MEAN_WIDTH) {\n    const unsigned int offset = row * ZG_COLUMNS + index * 4;\n    accum0 = __fadd_rn(accum0, source[offset]);\n    accum1 = __fadd_rn(accum1, source[offset + 1]);\n    accum2 = __fadd_rn(accum2, source[offset + 2]);\n    accum3 = __fadd_rn(accum3, source[offset + 3]);\n  }\n";
const STORE: &str = "  if (x == 0) output[row] = __fmul_rn(value, ZG_MEAN_FACTOR);";
const MASKED_STORE: &str =
    "  if (row < ZG_ROWS && x == 0) output[row] = __fmul_rn(value, ZG_MEAN_FACTOR);";

pub(super) fn fixed_prompt_source(original: &str) -> Result<String> {
    if original.matches(LOADS).count() != 1 || original.matches(STORE).count() != 1 {
        return Err(Error::State(
            "original RMS Mean source anchors changed".into(),
        ));
    }
    let guarded = format!("  if (row < ZG_ROWS) {{\n{LOADS}  }}\n");
    Ok(original
        .replace(LOADS, &guarded)
        .replace(STORE, MASKED_STORE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_mean_load_and_store_are_guarded_all_threads_keep_barriers() -> Result<()> {
        let original = super::super::SOURCE;
        let candidate = fixed_prompt_source(original)?;
        assert_eq!(candidate.matches("if (row < ZG_ROWS)").count(), 1);
        assert_eq!(candidate.matches("row < ZG_ROWS && x == 0").count(), 1);
        assert_eq!(
            candidate.matches("__syncthreads()").count(),
            original.matches("__syncthreads()").count()
        );
        assert_eq!(
            candidate
                .matches("__shfl_down_sync(0xffffffffu, value, offset)")
                .count(),
            1
        );
        assert!(!candidate.contains("return;"));
        let recovered = candidate
            .replace(&format!("  if (row < ZG_ROWS) {{\n{LOADS}  }}\n"), LOADS)
            .replace(MASKED_STORE, STORE);
        assert_eq!(recovered, original);
        for changed in [
            original.replace(LOADS, ""),
            original.replace(STORE, ""),
            format!("{original}{LOADS}"),
        ] {
            assert!(fixed_prompt_source(&changed).is_err());
        }
        Ok(())
    }
}
