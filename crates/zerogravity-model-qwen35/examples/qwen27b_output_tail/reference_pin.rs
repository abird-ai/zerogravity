//! Filled only by ROOT after independent actual original-source/raw/lifetime validation.
pub(super) struct Pin {
    pub path: &'static str,
    pub bytes: usize,
    pub sha256: &'static str,
}
pub(super) const ACTUAL_REFERENCE: Option<Pin> = Some(Pin {
    path: "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/model-execution-qwen27b-output-tail-root-20261022/actual-reference-1.json",
    bytes: 13028,
    sha256: "735a06088e1ee4af933fb13c085e38bcc766abbaca52aa49c3324e77113efbe2",
});
