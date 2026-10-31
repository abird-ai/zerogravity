//! Embedded server API. S0 builders are cold; start explicitly refuses until F2.
pub use zerogravity_serving::{
    Server, ServerBuilder, ServerHandle, admin, auth, http, ingress, metrics, openai, realtime,
    responses, stream,
};
