pub(super) struct Pin {
    pub path: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}
// ROOT seals only after authentic source/owned/raw validation.
pub(super) const ACTUAL_REFERENCE: Option<Pin> = None;
