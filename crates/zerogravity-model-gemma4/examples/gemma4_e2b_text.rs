//! Explicit learned-source diagnostic; analytic F32 remains unqualified.
#[cfg(target_os = "linux")]
#[path = "gemma4_e2b_text/run.rs"]
mod run;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        run::run()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("Gemma source-window diagnostic requires Linux".into())
    }
}
