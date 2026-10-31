//! Scope-local state, not model/backend qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeStatus {
    pub closed: bool,
    pub loaded_models: usize,
    pub execution_adapters: usize,
}
