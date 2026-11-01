//! Embedded bounded HTTP transport; explicit start/stop over the caller's runtime.
pub use zerogravity_serving::{
    Server, ServerBuilder, ServerHandle, TextService, admin, auth, http, ingress, metrics, openai,
    realtime, responses, service, stream,
};
