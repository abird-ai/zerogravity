//! Real managed source -> text -> SAME private greedy session, expected-only matrix.
#[path = "gemma4_cpu_matrix/comparison.rs"]
mod comparison;
#[path = "gemma4_cpu_matrix/io.rs"]
mod io;
#[path = "gemma4_cpu_matrix/reference.rs"]
mod reference;
#[path = "gemma4_cpu_matrix/run.rs"]
mod run;
fn main() {
    if let Err(e) = run::run() {
        eprintln!("GEMMA4_CPU_MATRIX_REFUSED: {e}");
        std::process::exit(1);
    }
}
