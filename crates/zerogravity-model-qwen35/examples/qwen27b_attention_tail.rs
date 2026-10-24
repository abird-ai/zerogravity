//! ROOT-only learned final attention/cache/tail replay; no full-model producer.
#[cfg(target_os = "linux")]
#[path = "qwen27b_attention_tail/replay.rs"]
mod replay;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        replay::run()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("original attention-tail replay requires Linux".into())
    }
}
