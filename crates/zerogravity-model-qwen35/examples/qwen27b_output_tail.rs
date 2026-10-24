//! ROOT-only learned final63 tail replay; no token loop, StateRoot or serving.
#[cfg(target_os = "linux")]
#[path = "qwen27b_output_tail/replay.rs"]
mod replay;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        replay::run()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("original27B source-window replay requires Linux".into())
    }
}
