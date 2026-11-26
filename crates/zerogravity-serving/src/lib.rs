//! Embeddable bounded async HTTP/1.1 transport over an existing text scheduler.
//! Cold builders; explicit loopback listeners; no process-global policy.
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
pub mod service;
pub use service::TextService;
pub mod stream;
#[cfg(test)]
mod test_support;
pub use server::{Server, ServerBuilder, ServerHandle};
