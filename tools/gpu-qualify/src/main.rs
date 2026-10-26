//! Explicit qualification frontend; production implementation is linked once.
mod qualification;
use zerogravity_gpu_qualify::models;

fn main() {
    qualification::run();
}
