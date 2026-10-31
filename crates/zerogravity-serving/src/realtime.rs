//! Future transports; network ACK is not playback or neural commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealtimeTransport {
    WebSocket,
    WebRtc,
}
impl RealtimeTransport {
    pub fn require_implemented(self) -> crate::Result<()> {
        Err(crate::Error::Unsupported(format!(
            "Fabric {self:?} transport is not implemented"
        )))
    }
}
