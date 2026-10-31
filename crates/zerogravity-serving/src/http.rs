//! Protocol intent, not an active HTTP parser or endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}
