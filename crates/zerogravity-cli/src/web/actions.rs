//! Bounded browser action grammar. Never accept arbitrary argv, local file
//! paths, remote URLs, credential fields, or caller-supplied runtime policy.
use super::Shared;
use crate::control_host::{
    Origin,
    execution::Execution,
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
        execution: Option<ExecutionInput>,
    },
    RunSource {
        source: String,
        prompt: String,
        chat: bool,
        execution: Option<ExecutionInput>,
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
    Inspect {
        digest: String,
    },
    Browse {
        source: String,
        query: String,
        page: usize,
        cursor: Option<String>,
        online: bool,
    },
    Repository {
        locator: String,
    },
    SaveSources {
        expected: Vec<zerogravity::discovery::Source>,
        entries: Vec<zerogravity::discovery::Source>,
    },
    Verify {
        digest: String,
    },
    RemovalPreview {
        digest: String,
    },
    Remove {
        digest: String,
        confirm: bool,
    },
    Alias {
        digest: String,
        name: String,
    },
    CheckFit {
        digest: String,
        device: String,
    },
    LoadOnDevice {
        digest: String,
        device: String,
    },
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    PullPreview {
        source: String,
        selection: String,
    },
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    Pull {
        source: String,
        selection: String,
        confirm: bool,
    },
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    Download {
        operation_id: String,
        command: DownloadCommand,
    },
    Settings {
        action: super::settings::Input,
    },
    Cancel {
        id: String,
    },
}

#[cfg(all(feature = "acquisition", target_os = "linux"))]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum DownloadCommand {
    Pause,
    Resume,
    Cancel,
}
#[cfg(all(feature = "acquisition", target_os = "linux"))]
impl From<DownloadCommand> for zerogravity::control::DownloadCommand {
    fn from(command: DownloadCommand) -> Self {
        match command {
            DownloadCommand::Pause => Self::Pause,
            DownloadCommand::Resume => Self::Resume,
            DownloadCommand::Cancel => Self::Cancel,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionInput {
    backend: String,
    device: Option<String>,
}

fn execution(value: Option<ExecutionInput>) -> Option<Execution> {
    let Some(value) = value else {
        return Some(Execution::default());
    };
    Some(Execution {
        backend: Some(zerogravity::run_policy::Backend::parse(&value.backend).ok()?),
        device: match value.device {
            Some(id) => Some(device(id)?),
            None => None,
        },
    })
}

fn digest(value: String) -> Option<String> {
    (cfg!(all(feature = "cache", target_os = "linux"))
        && SourceSelector::is_snapshot_digest(&value))
    .then_some(value)
}
fn device(value: String) -> Option<String> {
    // Only an exact physical inventory identity. No ordinal remapping, node
    // path or remote host. Explicit backend choices are a separate finite field.
    let valid = value == "cpu:0"
        || value.strip_prefix("pci:").is_some_and(|pci| {
            zerogravity::devices::normalize_pci(pci).is_ok_and(|canonical| canonical == pci)
        });
    valid.then_some(value)
}
#[cfg(all(feature = "acquisition", target_os = "linux"))]
fn pull(source: &str, selection: &str) -> Option<crate::pull::native::Intent> {
    if source.len() > 4096
        || selection.len() > 64 * 1024
        || selection
            .chars()
            .any(|c| c.is_control() && !matches!(c, '\r' | '\n' | '\t'))
    {
        return None;
    }
    let normalized = zerogravity::discovery::normalize_locator(source);
    let valid = match SourceSelector::parse(&normalized).ok()? {
        SourceSelector::Name(_) => true,
        SourceSelector::HuggingFace(value) | SourceSelector::GitHub(value) => pinned(&value),
        _ => false,
    };
    if !valid {
        return None;
    }
    crate::pull::native::Intent::parse(source, selection).ok()
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
    Settings(super::settings::Input),
    Start(Request),
    Cancel(u64),
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    Download(u64, zerogravity::control::DownloadCommand),
}
impl Input {
    fn validate(self) -> Option<Intent> {
        let request = match self {
            Self::Load {
                source: value,
                execution: selected,
            } => Request::Load {
                source: source(value)?,
                prompt: None,
                execution: execution(selected)?,
            },
            Self::RunSource {
                source: value,
                prompt: text,
                chat,
                execution: selected,
            } => Request::Load {
                source: source(value)?,
                prompt: Some(prompt(text, chat)?),
                execution: execution(selected)?,
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
            Self::Inspect { digest: value } => Request::Inspect(digest(value)?),
            Self::Browse {
                source,
                query,
                page,
                cursor,
                online,
            } if part(&source) && query.len() <= 256 && !query.chars().any(char::is_control) => {
                Request::Browse {
                    source,
                    query,
                    page,
                    cursor,
                    online,
                }
            }
            Self::Repository { locator } if locator.len() <= 4096 => {
                Request::Repository { locator }
            }
            Self::SaveSources { expected, entries } => {
                zerogravity::discovery::validate(&expected).ok()?;
                zerogravity::discovery::validate(&entries).ok()?;
                Request::SaveSources { expected, entries }
            }
            Self::Verify { digest: value } => Request::Verify(digest(value)?),
            Self::RemovalPreview { digest: value } => Request::Reclaim {
                digest: digest(value)?,
                apply: false,
            },
            Self::Remove {
                digest: value,
                confirm: true,
            } => Request::Reclaim {
                digest: digest(value)?,
                apply: true,
            },
            Self::Alias {
                digest: value,
                name,
            } if SourceSelector::is_human_name(&name) => Request::Alias {
                digest: digest(value)?,
                name,
            },
            Self::CheckFit {
                digest: value,
                device: selected,
            } => Request::CheckFit {
                source: format!("sha256:{}", digest(value)?),
                device: device(selected)?,
            },
            Self::LoadOnDevice {
                digest: value,
                device: selected,
            } => Request::Load {
                source: format!("sha256:{}", digest(value)?),
                execution: Execution::on_device(device(selected)?),
                prompt: None,
            },
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            Self::PullPreview { source, selection } => {
                Request::PullPreview(pull(&source, &selection)?)
            }
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            Self::Pull {
                source,
                selection,
                confirm: true,
            } => Request::Pull(pull(&source, &selection)?),
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            Self::Download {
                operation_id,
                command,
            } => return Some(Intent::Download(id(&operation_id)?, command.into())),
            Self::Settings { action } => return Some(Intent::Settings(action)),
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
    let intent = match intent {
        Intent::Settings(input) => Intent::Start(Request::Settings(input.action(shared)?)),
        other => other,
    };
    match intent {
        Intent::Settings(_) => unreachable!("settings resolved before admission"),
        Intent::Start(request) => {
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            if matches!(&request, Request::Pull(_)) && shared.selection.snapshot().offline {
                return Err("403 Forbidden");
            }
            let id = shared
                .actions
                .start(runtime, &shared.selection, request, Origin::Web)
                .map_err(|error| match error {
                    zerogravity::Error::State(_) => "409 Conflict",
                    _ => "503 Service Unavailable",
                })?;
            Ok(("202 Accepted", json!({"id": id.to_string()})))
        }
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        Intent::Download(id, command) => {
            // Direct control of the original active transfer, independent of
            // the shared action slot and its private native/Web receipts.
            runtime
                .control()
                .control_download(id, command)
                .map_err(|_| "409 Conflict")?;
            Ok(("202 Accepted", json!({"operation_id": id.to_string()})))
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
        "discovery": result.discovery,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn execution_intent_is_bounded_and_preserves_exact_pair() {
        use zerogravity::run_policy::Backend;
        let decode = |value: &str| {
            serde_json::from_str::<Input>(value)
                .ok()
                .and_then(Input::validate)
        };
        assert!(matches!(decode(r#"{"kind":"load","source":"model"}"#),
            Some(Intent::Start(Request::Load { execution, .. })) if execution == Execution::default()));
        assert!(
            matches!(decode(r#"{"kind":"run_source","source":"model","prompt":"hi","chat":false,"execution":{"backend":"rocm","device":"pci:0000:01:00.0"}}"#),
            Some(Intent::Start(Request::Load { execution, prompt: Some(_), .. }))
            if execution.backend == Some(Backend::Rocm) && execution.device.as_deref() == Some("pci:0000:01:00.0"))
        );
        for value in [
            r#"{"kind":"load","source":"model","execution":{"backend":"unknown"}}"#,
            r#"{"kind":"load","source":"model","execution":{"backend":"cuda","device":"gpu:0"}}"#,
            r#"{"kind":"load","source":"model","execution":{"backend":"auto","device":"/dev/dri/card0"}}"#,
            r#"{"kind":"load","source":"model","execution":{"backend":"cpu","headroom":"0"}}"#,
        ] {
            assert!(decode(value).is_none(), "{value}");
        }
    }

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
