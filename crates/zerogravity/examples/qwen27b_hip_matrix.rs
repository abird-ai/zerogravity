//! Explicit full64 finite teacher/source/StateRoot HIP qualification.
#[path = "qwen27b_hip_matrix/cache_import.rs"]
mod cache_import;
#[path = "qwen27b_hip_matrix/expected.rs"]
mod expected;
#[path = "gemma4_cpu_matrix/io.rs"]
mod io;
#[path = "qwen27b_hip_matrix/run.rs"]
mod run;
fn main() {
    if let Err(error) = run::run() {
        eprintln!("QWEN27B_HIP_MATRIX_REFUSED: {error}");
        std::process::exit(1);
    }
}
