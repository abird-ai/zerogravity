//! Explicit browser policy projection over the original ConfigFile snapshot.
//! Native-only values are neither disclosed nor writable through this seam.
use super::Shared;
use crate::control_host::{
    Target,
    settings::{self, Action, Snapshot},
};
use serde::Deserialize;
use serde_json::{Value, json};
use serde_yaml::Value as Yaml;
use zerogravity::{Error, Result, run_policy::Headroom};

pub(super) type Observed = Option<(String, Snapshot)>;

fn profile(name: &str) -> bool {
    zerogravity::run::config::validate_profile_name(name).is_ok()
}
pub(super) fn target(name: &str, shared: &Shared) -> Option<Target> {
    if name == "@active" {
        Some(shared.selection.snapshot().target)
    } else if profile(name) {
        Target::profile(name).ok()
    } else {
        None
    }
}
pub(super) fn profiles(shared: &Shared) -> Result<Value> {
    Ok(json!({
        "profiles": settings::profiles()?,
        "active": shared.selection.snapshot().target.profile,
    }))
}

fn editable(key: &str) -> bool {
    matches!(
        key,
        "run.backend"
            | "run.device"
            | "run.headroom"
            | "run.numerical"
            | "run.speculation"
            | "run.draft_device"
            | "run.task.context"
            | "run.task.max_tokens"
            | "run.task.raw"
            | "run.task.prefill_chunk"
            | "run.diagnostics.explain"
            | "run.diagnostics.dry_run"
            | "serve.listen"
            | "serve.workers"
            | "serve.active_per_worker"
            | "serve.max_context"
            | "serve.max_output"
            | "serve.host_mib"
            | "serve.max_pending_requests"
            | "serve.enable_per_request_metrics"
            | "ui.theme"
            | "ui.color"
            | "ui.progress"
            | "ui.interactive"
            | "downloads.ask_before_pull"
    )
}
fn safe_value(key: &str, value: &Yaml) -> bool {
    if !editable(key) {
        return false;
    }
    if value.is_null() {
        return true;
    }
    let text = value.as_str().unwrap_or("");
    match key {
        "run.backend" => matches!(text, "auto" | "cpu" | "cuda" | "rocm" | "vulkan"),
        "run.numerical" => matches!(text, "source" | "native"),
        "run.speculation" => matches!(text, "auto" | "off" | "mtp"),
        "run.headroom" => text.len() <= 32 && Headroom::parse(text).is_ok(),
        "run.device" | "run.draft_device" => value.as_sequence().is_some_and(|items| {
            items.len() <= 64
                && items.iter().all(|item| {
                    item.as_str().is_some_and(|id| {
                        matches!(id, "auto" | "cpu" | "cpu:0")
                            || id.strip_prefix("pci:").is_some_and(|pci| {
                                zerogravity::devices::normalize_pci(pci)
                                    .is_ok_and(|canonical| canonical == pci)
                            })
                    })
                })
        }),
        "serve.listen" => {
            text.len() <= 128
                && text
                    .parse::<std::net::SocketAddr>()
                    .is_ok_and(|address| address.ip().is_loopback())
        }
        "ui.theme" => zerogravity::configuration::ThemeMode::parse(text).is_some(),
        "ui.color" | "ui.progress" | "ui.interactive" => {
            matches!(text, "auto" | "always" | "never")
        }
        "run.task.raw"
        | "run.diagnostics.explain"
        | "run.diagnostics.dry_run"
        | "serve.enable_per_request_metrics"
        | "downloads.ask_before_pull" => value.as_bool().is_some(),
        _ => value.as_u64().is_some(),
    }
}
fn lookup<'a>(node: &'a Yaml, key: &str) -> Option<&'a Yaml> {
    key.split('.').try_fold(node, |node, part| node.get(part))
}
// Raw YAML is an opt-in normalized view of a wholly browser-safe document.
// No comments (which may contain secrets), graph, arbitrary text or paths.
fn safe_document(node: &Yaml, prefix: &str) -> bool {
    node.as_mapping().is_some_and(|map| {
        map.iter().all(|(key, value)| {
            let Some(key) = key.as_str() else {
                return false;
            };
            let path = if prefix.is_empty() {
                key.to_owned()
            } else {
                format!("{prefix}.{key}")
            };
            match path.as_str() {
                "schema" => matches!(
                    value.as_str(),
                    Some("zerogravity.pipeline.v2" | "zerogravity.pipeline.v1")
                ),
                "run" | "run.task" | "run.diagnostics" | "serve" | "ui" | "downloads" => {
                    value.is_null() || safe_document(value, &path)
                }
                "model" | "pipeline" | "route" => value.as_sequence().is_some_and(Vec::is_empty),
                _ => safe_value(&path, value),
            }
        })
    })
}
fn encode(value: &Yaml) -> Result<String> {
    serde_yaml::to_string(value).map_err(|_| Error::State("configuration encoding".into()))
}

pub(super) fn observe(shared: &Shared, target: Target) -> Result<Value> {
    let observed = settings::observe(target)?;
    let valid = observed.warning.is_none();
    let raw = if let Some(node) = &observed.node {
        if safe_document(node, "") {
            Some(encode(node)?)
        } else {
            None
        }
    } else {
        None
    };
    let rows = observed
        .rows
        .into_iter()
        .map(|row| {
            // Effective defaults are produced by the shared resolver, not by
            // browser code. Protect even existing typed strings outside our grammar.
            let allowed = editable(&row.key) && (!row.explicit || safe_value(&row.key, &row.value));
            let value = if allowed {
                encode(&row.value).ok()
            } else {
                None
            };
            let origin = if row.origin == "unset" {
                "unset"
            } else if row.origin == "built-in" {
                "built-in"
            } else {
                "document"
            };
            json!({"key": row.key, "value": value, "explicit": row.explicit,
            "editable": allowed, "origin": origin, "help": row.help})
        })
        .collect::<Vec<_>>();
    let active = shared.selection.snapshot().target;
    let selected = &observed.snapshot.target;
    let is_active = selected.path == active.path;
    let removable = selected
        .profile
        .as_deref()
        .is_some_and(|name| name != "default")
        && !is_active;
    let exists = observed.snapshot.exists();
    let name = selected.profile.clone();
    let mut retained = shared
        .config_snapshot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !retained
        .as_ref()
        .is_some_and(|(_, snapshot)| snapshot == &observed.snapshot)
    {
        let mut random = [0u8; 16];
        getrandom::fill(&mut random)
            .map_err(|_| Error::State("configuration observation identity".into()))?;
        let token = random
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        *retained = Some((token, observed.snapshot));
    }
    let token = &retained.as_ref().expect("retained snapshot").0;
    Ok(
        json!({"snapshot": token, "profile": name, "active": is_active, "exists": exists,
        "removable": removable, "valid": valid, "raw": raw, "rows": rows,
        "warning": (!valid).then_some("Invalid configuration; repair in the native editor.")}),
    )
}

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Input {
    Create {
        name: String,
    },
    Copy {
        snapshot: String,
        name: String,
    },
    Remove {
        snapshot: String,
        confirm: bool,
    },
    Activate {
        snapshot: String,
    },
    Validate {
        snapshot: String,
    },
    Set {
        snapshot: String,
        key: String,
        value: String,
    },
    Unset {
        snapshot: String,
        key: String,
        confirm: bool,
    },
    Replace {
        snapshot: String,
        yaml: String,
    },
}
impl Input {
    pub(super) fn action(self, shared: &Shared) -> std::result::Result<Action, &'static str> {
        let bad = "400 Bad Request";
        let observed = |token: &str| {
            if token.len() != 32
                || !token
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(bad);
            }
            shared
                .config_snapshot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_ref()
                .filter(|(id, _)| id == token)
                .map(|(_, file)| file.clone())
                .ok_or("409 Conflict")
        };
        let name = |name: String| {
            if profile(&name) && name != "default" {
                Ok(name)
            } else {
                Err(bad)
            }
        };
        let editable_file = |file: &Snapshot, key: &str| -> std::result::Result<(), &'static str> {
            let node = file.node().map_err(|_| bad)?;
            if editable(key) && lookup(&node, key).is_none_or(|value| safe_value(key, value)) {
                Ok(())
            } else {
                Err("403 Forbidden")
            }
        };
        Ok(match self {
            Self::Create { name: new } => Action::Create(name(new)?),
            Self::Copy {
                snapshot,
                name: new,
            } => Action::Copy {
                source: observed(&snapshot)?,
                name: name(new)?,
            },
            Self::Remove {
                snapshot,
                confirm: true,
            } => Action::Remove(observed(&snapshot)?),
            Self::Activate { snapshot } => Action::Activate(observed(&snapshot)?),
            Self::Validate { snapshot } => Action::Validate(observed(&snapshot)?),
            Self::Set {
                snapshot,
                key,
                value,
            } => {
                let file = observed(&snapshot)?;
                editable_file(&file, &key)?;
                if value.len() > 8192 {
                    return Err(bad);
                }
                let node: Yaml = serde_yaml::from_str(&value).map_err(|_| bad)?;
                if !safe_value(&key, &node) {
                    return Err("403 Forbidden");
                }
                Action::Set {
                    file,
                    key,
                    value: encode(&node).map_err(|_| bad)?,
                }
            }
            Self::Unset {
                snapshot,
                key,
                confirm: true,
            } => {
                let file = observed(&snapshot)?;
                editable_file(&file, &key)?;
                Action::Unset { file, key }
            }
            Self::Replace { snapshot, yaml } => {
                let file = observed(&snapshot)?;
                if !safe_document(&file.node().map_err(|_| bad)?, "") {
                    return Err("403 Forbidden");
                }
                let node =
                    zerogravity::configuration::document_value(&file.target.path, yaml.as_bytes())
                        .map_err(|_| bad)?;
                if !safe_document(&node, "") {
                    return Err("403 Forbidden");
                }
                Action::Replace {
                    file,
                    yaml: encode(&node).map_err(|_| bad)?,
                }
            }
            _ => return Err(bad),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_policy_is_not_a_path_or_arbitrary_text_grant() {
        for yaml in [
            "schema: zerogravity.pipeline.v2\nrun: {backend: cpu, device: ['cpu:0'], task: {context: 1024}}",
            "schema: zerogravity.pipeline.v2\nserve: {listen: '127.0.0.1:0'}\nui: {color: never}",
        ] {
            assert!(safe_document(&serde_yaml::from_str(yaml).unwrap(), ""));
        }
        for yaml in [
            "run: {task: {prompt: secret}}",
            "run: {diagnostics: {audit_logits_jsonl: /tmp/file}}",
            "run: {device: [/dev/dri/renderD128]}",
            "run: {speculation: 'draft:/tmp/weights'}",
            "model: [{source: /tmp/weights}]",
            "serve: {listen: '0.0.0.0:80'}",
            "api_key: secret",
        ] {
            assert!(
                !safe_document(&serde_yaml::from_str(yaml).unwrap(), ""),
                "{yaml}"
            );
        }
    }
}
