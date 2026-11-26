//! Native action ticket and presentation metadata. The shared host owns the
//! actual worker, cancellation, output and terminal result.
use super::{library::Key, settings};
pub(super) use crate::control_host::work::{Completed, Owner, Prompt, Request, Response};
pub(super) use crate::control_host::{Document, combine, retire};

impl Request {
    pub fn correlation(&self) -> String {
        match self {
            Self::Invoke { owner, .. } | Self::Unload(owner) => {
                format!("registration {}", owner.registration)
            }
            Self::Stop(generation) => format!("generation {generation}"),
            Self::Inspect(digest)
            | Self::Verify(digest)
            | Self::Reclaim { digest, .. }
            | Self::Alias { digest, .. } => format!("sha256:{digest}"),
            Self::Settings(_) => "configuration".into(),
            _ => "local".into(),
        }
    }
    fn detail_key(&self) -> Option<Key> {
        match self {
            Self::Inspect(digest)
            | Self::Verify(digest)
            | Self::Reclaim { digest, .. }
            | Self::Alias { digest, .. } => Some(Key::Snapshot(digest.clone())),
            Self::CheckFit { device, .. } => Some(Key::Device(device.clone())),
            _ => None,
        }
    }
}
pub(super) struct Job {
    pub id: u64,
    pub title: &'static str,
    pub detail_key: Option<Key>,
    pub log_id: u64,
    pub retry: Option<settings::Action>,
}
impl Job {
    pub fn ticket(request: &Request, log_id: u64) -> Self {
        let retry = match request {
            Request::Settings(
                action @ (settings::Action::Set { .. } | settings::Action::Replace { .. }),
            ) => Some(action.clone()),
            _ => None,
        };
        Self {
            id: 0,
            title: request.title(),
            detail_key: request.detail_key(),
            log_id,
            retry,
        }
    }
}
