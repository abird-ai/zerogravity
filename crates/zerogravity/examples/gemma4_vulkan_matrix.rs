//! Explicit complete original Vulkan/source/state qualification; no serving permit.
#[path = "gemma4_vulkan_matrix/cache_import.rs"]
mod cache_import;
#[path = "gemma4_cpu_matrix/comparison.rs"]
mod comparison;
#[path = "gemma4_vulkan_matrix/expected.rs"]
mod expected;
#[path = "gemma4_cpu_matrix/io.rs"]
mod io;
#[path = "gemma4_vulkan_matrix/run.rs"]
mod run;
fn main() {
    if let Err(error) = run::run() {
        eprintln!("GEMMA4_VULKAN_MATRIX_REFUSED: {error}");
        std::process::exit(1);
    }
}
