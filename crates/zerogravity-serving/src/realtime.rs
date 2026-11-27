//! Transport availability does not imply a qualified loaded-model capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealtimeTransport {
    WebSocket,
    WebRtc,
}
impl RealtimeTransport {
    pub fn require_implemented(self) -> crate::Result<()> {
        match self {
            Self::WebSocket => Ok(()),
            Self::WebRtc => Err(crate::Error::Unsupported(
                "Realtime WebRTC transport is not implemented".into(),
            )),
        }
    }
}
