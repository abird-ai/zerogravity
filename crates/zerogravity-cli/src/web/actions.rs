//! Bounded browser action grammar. Never accept arbitrary argv, local file
//! paths, remote URLs, credential fields, or caller-supplied runtime policy.
use super::Shared;
use crate::control_host::{
    Origin,
    work::{Owner, Prompt, Request},
};
use serde::Deserialize;
use serde_json::{Value, json};
use zerogravity::{Runtime, run_policy::SourceSelector};

pub(super) const MAX_BODY: usize = 256 * 1024;
const MAX_PROMPT: usize = 32 * 1024;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Input {
    Load {
        source: String,
    },
    RunSource {
        source: String,
        prompt: String,
        chat: bool,
    },
    Invoke {
        alias: String,
        registration_id: String,
        prompt: String,
        chat: bool,
    },
    Unload {
        alias: String,
        registration_id: String,
        confirm: bool,
    },
    Start {
        address: String,
    },
    Stop {
        generation: String,
        confirm: bool,
    },
    Cancel {
        id: String,
    },
}

fn id(value: &str) -> Option<u64> {
    let parsed = value.parse::<u64>().ok()?;
    (parsed.to_string() == value).then_some(parsed)
}
fn owner(alias: String, registration_id: String) -> Option<Owner> {
    zerogravity::runtime::text::validate_alias(&alias).ok()?;
    Some(Owner {
        alias,
        registration: id(&registration_id)?,
    })
}
fn prompt(text: String, chat: bool) -> Option<Prompt> {
    if text.trim().is_empty() || text.len() > MAX_PROMPT || text.contains('\0') {
        return None;
    }
    Some(Prompt { text, chat })
}
fn part(value: &str) -> bool {
    !value.is_empty()
        && !matches!(value, "." | "..")
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
fn pinned(value: &str) -> bool {
    let Some((repo, commit)) = value.split_once('@') else {
        return false;
    };
    let Some((owner, name)) = repo.split_once('/') else {
        return false;
    };
    part(owner)
        && part(name)
        && commit.len() == 40
        && commit
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn source(value: String) -> Option<String> {
    if value.len() > 4096 || value.chars().any(char::is_control) {
        return None;
    }
    let value = value.trim();
    if value.is_empty() {
        return Some(String::new());
    } // trusted selected document
    let selector = if let Some((alias, source)) = value.split_once('=') {
        zerogravity::runtime::text::validate_alias(alias).ok()?;
        source
    } else {
        value
    };
    let allowed = match SourceSelector::parse(selector).ok()? {
        SourceSelector::Name(_) | SourceSelector::Snapshot(_) | SourceSelector::ByName(_) => true,
        SourceSelector::HuggingFace(locator) | SourceSelector::GitHub(locator) => pinned(&locator),
        SourceSelector::Catalog(reference) => {
            // Catalog reference segments, not filesystem authority.
            reference.split('/').all(part)
        }
        SourceSelector::Local(_) | SourceSelector::Http(_) => false,
    };
    allowed.then(|| value.to_owned())
}

enum Intent {
    Start(Request),
    Cancel(u64),
}
impl Input {
    fn validate(self) -> Option<Intent> {
        let request = match self {
            Self::Load { source: value } => Request::Load {
                source: source(value)?,
                prompt: None,
                device: None,
            },
            Self::RunSource {
                source: value,
                prompt: text,
                chat,
            } => Request::Load {
                source: source(value)?,
                prompt: Some(prompt(text, chat)?),
                device: None,
            },
            Self::Invoke {
                alias,
                registration_id,
                prompt: text,
                chat,
            } => Request::Invoke {
                owner: owner(alias, registration_id)?,
                prompt: prompt(text, chat)?,
            },
            Self::Unload {
                alias,
                registration_id,
                confirm: true,
            } => Request::Unload(owner(alias, registration_id)?),
            Self::Start { address } => {
                if !cfg!(feature = "serve")
                    || address.len() > 128
                    || address.chars().any(char::is_control)
                {
                    return None;
                }
                let address = address.trim();
                if !address.is_empty()
                    && !address
                        .parse::<std::net::SocketAddr>()
                        .is_ok_and(|address| address.ip().is_loopback())
                {
                    return None;
                }
                Request::Start(address.to_owned())
            }
            Self::Stop {
                generation,
                confirm: true,
            } if cfg!(feature = "serve") => Request::Stop(id(&generation)?),
            Self::Cancel { id: value } => return Some(Intent::Cancel(id(&value)?)),
            _ => return None,
        };
        Some(Intent::Start(request))
    }
}

pub(super) fn submit(
    runtime: &Runtime,
    shared: &Shared,
    body: &[u8],
) -> Result<(&'static str, Value), &'static str> {
    let intent = serde_json::from_slice::<Input>(body)
        .ok()
        .and_then(Input::validate)
        .ok_or("400 Bad Request")?;
    if shared.stop.load(std::sync::atomic::Ordering::Acquire) {
        return Err("503 Service Unavailable");
    }
    match intent {
        Intent::Start(request) => {
            let id = shared
                .actions
                .start(runtime, &shared.selection, request, Origin::Web)
                .map_err(|error| match error {
                    zerogravity::Error::State(_) => "409 Conflict",
                    _ => "503 Service Unavailable",
                })?;
            Ok(("202 Accepted", json!({"id": id.to_string()})))
        }
        Intent::Cancel(id) => {
            shared
                .actions
                .cancel(id, Some(Origin::Web))
                .map_err(|_| "409 Conflict")?;
            Ok((
                "202 Accepted",
                json!({"id": id.to_string(), "cancel_requested": true}),
            ))
        }
    }
}

pub(super) fn result(shared: &Shared, identity: &str) -> Result<Value, &'static str> {
    let id = id(identity).ok_or("404 Not Found")?;
    let result = shared.actions.web_result(id).ok_or("404 Not Found")?;
    let response = result.response.map(|response| json!({
        "revision": response.revision.to_string(),
        "owner": response.owner.map(|owner| json!({"alias": owner.alias, "registration_id": owner.registration.to_string()})),
        "text": response.text,
        "clipped": response.clipped,
        "usage": response.usage,
    }));
    Ok(json!({
        "id": result.summary.id.to_string(),
        "title": result.summary.title,
        "state": result.state,
        "cancel_requested": result.summary.cancel_requested,
        "message": result.message,
        "response": response,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_web_intent_refuses_paths_credentials_stale_id_coercions_and_missing_consent() {
        for value in [
            "",
            "qwen3.5-2b",
            "by-name:local",
            "route=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "hf:org/repo@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "catalog:huggingface/org/repo/revision",
        ] {
            assert!(source(value.into()).is_some(), "{value}");
        }
        for value in [
            "/tmp/model.gguf",
            "../model.gguf",
            "route=/tmp/model.gguf",
            "https://host/file",
            "hf:org/repo@main",
            "catalog:../model.gguf",
            "catalog:/etc/passwd",
            "hf:org/../repo@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "route=hf:org/repo@token?key=secret",
        ] {
            assert!(source(value.into()).is_none(), "{value}");
        }
        for bytes in [
            br#"{"kind":"load","source":"","api_key":"secret"}"#.as_slice(),
            br#"{"kind":"load","source":"","source":"other"}"#,
            br#"{"kind":"load","kind":"invoke","source":""}"#,
            br#"{"kind":"unload","alias":"model","registration_id":1,"confirm":true}"#,
            br#"{"kind":"unload","alias":"model","registration_id":"01","confirm":true}"#,
            br#"{"kind":"unload","alias":"model","registration_id":"1","confirm":false}"#,
            br#"{"kind":"unload","alias":"model","registration_id":"1"}"#,
            br#"{"kind":"invoke","alias":"model","registration_id":"1","prompt":"","chat":false}"#,
        ] {
            assert!(
                serde_json::from_slice::<Input>(bytes)
                    .ok()
                    .and_then(Input::validate)
                    .is_none()
            );
        }
        assert!(matches!(serde_json::from_str::<Input>(r#"{"kind":"unload","alias":"model","registration_id":"18446744073709551615","confirm":true}"#).unwrap().validate(), Some(Intent::Start(Request::Unload(Owner { registration: u64::MAX, .. })))));
    }
}
