//! Explicit qualification frontend; production implementation is linked once.
mod models;
mod qualification;

fn main() {
    qualification::run();
}
