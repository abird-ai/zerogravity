//! Management intent only; no unauthenticated control route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdminOperation {
    ListModels,
    UnloadModel,
    Drain,
}
