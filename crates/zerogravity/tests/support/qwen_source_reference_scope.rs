//! Original-reference testcase schedules, not production capability metadata.
use zerogravity::{Error, Result};

#[derive(Clone, Copy)]
pub struct ReferenceScope {
    pub layers: usize,
    pub contexts: &'static [usize],
}
impl ReferenceScope {
    pub fn for_repository(repository: &str) -> Result<Self> {
        match repository {
            "Qwen/Qwen3.5-0.8B" | "Qwen/Qwen3.5-2B" => Ok(Self {
                layers: 24,
                contexts: &[1024, 2048, 4096, 8192, 16384],
            }),
            "Qwen/Qwen3.8-27B" => Ok(Self {
                layers: 64,
                contexts: &[1024, 2048],
            }),
            _ => Err(Error::Unsupported(
                "original reference model/context scope".into(),
            )),
        }
    }
    pub fn active_context(self) -> usize {
        self.contexts[self.contexts.len() - 1]
    }
    pub fn populated_counts(self) -> (usize, usize) {
        // Two repeats; genuine64-row calls, then two natural decodes.
        let calls = 2 * self
            .contexts
            .iter()
            .map(|context| context / 64 + 2)
            .sum::<usize>();
        // Last prefill + both decodes: outputs and before/after state planes.
        let records = 2 * self.contexts.len() * 3 * (self.layers + 4 + 4 * self.layers);
        (calls, records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn populated_scope_keeps_all_calls_distinct_from_sparse_checkpoints() {
        for repository in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"] {
            let scope = ReferenceScope::for_repository(repository).unwrap();
            assert_eq!(scope.layers, 24);
            assert_eq!(scope.active_context(), 16384);
            assert_eq!(scope.populated_counts(), (1012, 3720));
        }
        let scope = ReferenceScope::for_repository("Qwen/Qwen3.8-27B").unwrap();
        assert_eq!(scope.layers, 64);
        assert_eq!(scope.contexts, [1024, 2048]);
        assert_eq!(scope.active_context(), 2048);
        assert_eq!(scope.populated_counts(), (104, 3888));
        assert!(ReferenceScope::for_repository("unqualified").is_err());
    }
}
