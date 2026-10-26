//! Qwen qualification drivers; all execution uses the existing engine.
// Previous GPU modules: model_q4, model_mixed, model_resident, model_layer, model_sessions, model_workers, model_bench, activation_stress, operator_stress.
pub mod activation;
pub mod benchmark;
pub mod layer;
pub mod mixed;
pub mod operators;
pub mod q4;
pub mod resident;
pub mod rope;
pub mod sessions;
pub mod workers;
