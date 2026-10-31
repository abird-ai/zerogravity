//! Independently compilable server shell. HTTP/worker migration is F2.
//! S0 starts no listener and serves no synthetic responses.
pub use zerogravity_core::{Error, Result};
pub mod admin;
pub mod auth;
pub mod http;
pub mod ingress;
pub mod metrics;
pub mod openai;
pub mod realtime;
pub mod responses;
pub mod server;
pub mod stream;
pub use server::{Server, ServerBuilder, ServerHandle};
