//! Explicit launch metadata for trusted precompiled CUDA text.
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::CompilationError;

const PREFIX: &str = "// cubecl.cuda.dynamic_shared_bytes=";

pub(super) fn bytes(source: &str, maximum: usize) -> Result<usize, CompilationError> {
    let Some(value) = source.strip_prefix(PREFIX) else {
        return Ok(0);
    };
    let fail = || CompilationError::Generic {
        reason: "invalid precompiled CUDA dynamic shared-memory declaration".into(),
        backtrace: BackTrace::capture(),
    };
    let (value, rest) = value.split_once('\n').ok_or_else(fail)?;
    if value.is_empty()
        || value.len() > 6
        || !value.bytes().all(|b| b.is_ascii_digit())
        || rest.contains(PREFIX)
    {
        return Err(fail());
    }
    let bytes: usize = value.parse().map_err(|_| fail())?;
    if bytes > maximum {
        return Err(fail());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::bytes;

    #[test]
    fn exact_metadata_refuses_before_compilation_or_cache_insertion() {
        assert_eq!(bytes("ordinary CUDA source", 0).unwrap(), 0);
        for n in [0, 8, 49152, 49412, 98304] {
            let source = format!("// cubecl.cuda.dynamic_shared_bytes={n}\nbody");
            assert_eq!(bytes(&source, 98304).unwrap(), n);
            if n > 0 {
                assert!(bytes(&source, n - 1).is_err());
            }
        }
        for value in ["", "-1", "+1", "1 ", "1;", "1000000", "1\r", "１"] {
            assert!(bytes(&format!("// cubecl.cuda.dynamic_shared_bytes={value}\n"), 98304).is_err());
        }
        assert!(bytes("// cubecl.cuda.dynamic_shared_bytes=8", 98304).is_err());
        assert!(bytes("// cubecl.cuda.dynamic_shared_bytes=8\n// cubecl.cuda.dynamic_shared_bytes=8\n", 98304).is_err());
    }
}
