//! Diagnostic original full64 replay; no production/native numerical permit.
#[cfg(target_os = "linux")]
#[path = "qwen27b_full_window/replay.rs"]
mod replay;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        replay::run()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("original full-window replay requires Linux".into())
    }
}
