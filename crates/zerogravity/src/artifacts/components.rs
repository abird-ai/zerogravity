//! Bounded component document and explicit label-to-transport bindings.
use super::acquisition;
use crate::{Error, Result};
use acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
use acquisition::{
    ComponentKind, ComponentPullRequest, ComponentRequest, ComponentTransport, Provider,
    PullRequest, Selection,
};
use serde::Deserialize;
use std::{collections::BTreeMap, io::Read, path::Path, sync::atomic::AtomicBool};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanFile {
    schema: String,
    components: Vec<Input>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    label: String,
    role: String,
    provider: String,
    repository: String,
    revision: Option<String>,
    files: Option<Vec<String>>,
    subfolder: Option<String>,
    release: Option<String>,
    asset: Option<String>,
    snapshot: Option<String>,
}
fn invalid() -> Error {
    Error::Malformed(
        "invalid component plan; use the fixed pull components schema (no credentials or URLs)"
            .into(),
    )
}
pub struct ComponentDocument {
    request: ComponentPullRequest,
    bindings: Vec<(String, String)>,
}
impl ComponentDocument {
    pub fn open(path: &Path, bounds: &acquisition::PullLimits) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes, bounds)
    }
    pub fn from_bytes(bytes: &[u8], bounds: &acquisition::PullLimits) -> Result<Self> {
        bounds.validate()?;
        if bytes.len() > 1024 * 1024 {
            return Err(Error::Limit("component plan exceeds1MiB".into()));
        }
        let raw: PlanFile = serde_json::from_slice(bytes).map_err(|_| invalid())?;
        if raw.schema != "zerogravity.pull.components.v1"
            || raw.components.is_empty()
            || raw.components.len() > 16
        {
            return Err(invalid());
        }
        let mut requests = Vec::new();
        let mut bindings = Vec::new();
        for input in raw.components {
            let role = match input.role.as_str() {
                "weights" => ComponentKind::Weights,
                "config" => ComponentKind::Config,
                "tokenizer" => ComponentKind::Tokenizer,
                "auxiliary" => ComponentKind::Auxiliary,
                _ => return Err(invalid()),
            };
            let provider = match input.provider.as_str() {
                "huggingface" => Provider::HuggingFace,
                "github" => Provider::GitHub,
                _ => return Err(invalid()),
            };
            let (revision, selection) =
                match (input.revision, input.files, input.release, input.asset) {
                    (Some(revision), Some(paths), None, None) => (
                        revision,
                        Selection::Files {
                            paths,
                            subfolder: input.subfolder,
                        },
                    ),
                    (None, None, Some(tag), Some(name)) if input.subfolder.is_none() => {
                        (tag, Selection::ReleaseAsset { name })
                    }
                    _ => return Err(invalid()),
                };
            if let Some(snapshot) = input.snapshot {
                if snapshot.is_empty()
                    || snapshot.len() > 4096
                    || snapshot.chars().any(char::is_control)
                {
                    return Err(invalid());
                }
                bindings.push((input.label.clone(), snapshot));
            }
            requests.push(ComponentRequest::new(
                input.label,
                role,
                PullRequest {
                    provider,
                    repository: input.repository,
                    revision,
                    selection,
                },
            )?);
        }
        Ok(Self {
            request: ComponentPullRequest::new(requests, bounds)?,
            bindings,
        })
    }
    pub fn request(&self) -> &ComponentPullRequest {
        &self.request
    }
    pub fn bindings(&self) -> &[(String, String)] {
        &self.bindings
    }
    pub fn online_request(&self) -> Result<&ComponentPullRequest> {
        if !self.bindings.is_empty() {
            return Err(Error::Malformed(
                "component snapshot selectors are offline-only".into(),
            ));
        }
        Ok(&self.request)
    }
}
pub struct ComponentClients<T>(BTreeMap<String, T>);
impl<T> ComponentClients<T> {
    /// Explicit bindings only; no environment lookup or transport construction.
    pub fn new(clients: BTreeMap<String, T>) -> Self {
        Self(clients)
    }
}
impl<T: HttpTransport> ComponentTransport for ComponentClients<T> {
    fn head(
        &self,
        label: &str,
        request: &HttpRequest,
        cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        self.0
            .get(label)
            .ok_or_else(|| Error::State("component client binding missing".into()))?
            .head(request, cancel)
    }

    fn get(&self, label: &str, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        self.0
            .get(label)
            .ok_or_else(|| Error::State("component client binding missing".into()))?
            .get(request, cancel)
    }
}
