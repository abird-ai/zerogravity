//! Named discovery sources, independent of cache publication and execution.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    HuggingFace,
    Github,
    Local,
}
impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "Hugging Face",
            Self::Github => "GitHub",
            Self::Local => "Local folder",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub name: String,
    pub kind: Kind,
    /// Remote namespace or absolute local folder. None means whole hub.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(default = "enabled")]
    pub enabled: bool,
}
fn enabled() -> bool {
    true
}

pub fn defaults() -> Vec<Source> {
    [
        ("hf", Kind::HuggingFace),
        ("github", Kind::Github),
        ("local", Kind::Local),
    ]
    .into_iter()
    .map(|(name, kind)| Source {
        name: name.into(),
        kind,
        scope: None,
        enabled: true,
    })
    .collect()
}

fn bad(message: &str) -> Error {
    Error::Malformed(message.into())
}
fn component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 200
        && !matches!(value, "." | "..")
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
fn repo(value: &str) -> bool {
    value
        .split_once('/')
        .is_some_and(|(owner, name)| component(owner) && component(name))
}
pub fn validate(sources: &[Source]) -> Result<()> {
    let mut names = BTreeSet::new();
    for source in sources {
        if !component(&source.name) || !names.insert(&source.name) {
            return Err(bad("source names must be unique nonempty identifiers"));
        }
        if let Some(scope) = &source.scope {
            if scope.is_empty() || scope.len() > 4096 || scope.chars().any(char::is_control) {
                return Err(bad("invalid discovery source scope"));
            }
            match source.kind {
                Kind::Local if !Path::new(scope).is_absolute() => {
                    return Err(bad("local source requires an absolute directory"));
                }
                Kind::HuggingFace | Kind::Github if !component(scope) && !repo(scope) => {
                    return Err(bad("remote source scope must be a namespace or OWNER/REPO"));
                }
                _ => {}
            }
        }
    }
    Ok(())
}
pub fn sources(doc: &crate::configuration::ConfigDocument) -> Vec<Source> {
    doc.sources.clone().unwrap_or_else(defaults)
}
/// Change the existing configuration document atomically. The caller supplies
/// the observed file, so stale edits cannot overwrite another writer.
pub fn replace(file: &crate::configuration::ConfigFile, entries: &[Source]) -> Result<()> {
    validate(entries)?;
    let yaml =
        serde_yaml::to_string(entries).map_err(|_| bad("cannot encode discovery sources"))?;
    file.change(crate::configuration::Change::Set {
        path: "sources",
        yaml: &yaml,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub name: String,
    /// Original source selector. Not an acquisition or load permit.
    pub locator: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curated: Option<String>,
    pub payload_hold: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Page {
    pub entries: Vec<Entry>,
    /// More results may exist. No false completeness claim.
    pub more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}
const PAGE: usize = 30;

fn check(source: &Source, cancel: &AtomicBool) -> Result<()> {
    validate(std::slice::from_ref(source))?;
    if !source.enabled {
        return Err(Error::Unsupported("discovery source is disabled".into()));
    }
    if cancel.load(Ordering::Acquire) {
        return Err(Error::State("discovery canceled".into()));
    }
    Ok(())
}
fn in_scope(source: &Source, repository: &str) -> bool {
    source.scope.as_ref().is_none_or(|scope| {
        repository == scope
            || repository
                .strip_prefix(scope)
                .is_some_and(|rest| rest.starts_with('/'))
    })
}
fn entry(kind: Kind, name: String) -> Entry {
    let target = if kind == Kind::HuggingFace {
        crate::targets::MODEL_TARGETS
            .iter()
            .find(|t| t.repository == Some(name.as_str()))
    } else {
        None
    };
    Entry {
        locator: format!(
            "{}:{name}",
            if kind == Kind::HuggingFace {
                "hf"
            } else {
                "gh"
            }
        ),
        curated: target.map(|t| t.slug.into()),
        payload_hold: kind == Kind::HuggingFace
            && crate::targets::MODEL_TARGETS.iter().any(|t| {
                t.payload_hold
                    && t.repository
                        .is_some_and(|repo| repo.eq_ignore_ascii_case(&name))
            }),
        name,
        bytes: None,
    }
}
pub fn curated(source: &Source) -> Vec<Entry> {
    if !source.enabled || source.kind == Kind::Local {
        return Vec::new();
    }
    if let Some(scope) = &source.scope
        && repo(scope)
    {
        return vec![entry(source.kind, scope.clone())];
    }
    if source.kind != Kind::HuggingFace {
        return Vec::new();
    }
    crate::targets::MODEL_TARGETS
        .iter()
        .filter_map(|target| {
            let name = target.repository?;
            (repo(name) && in_scope(source, name)).then(|| entry(source.kind, name.into()))
        })
        .collect()
}
/// Normalize direct repository URLs only. Arbitrary HTTPS object URLs retain
/// their original locator and must use the explicit object acquisition path.
pub fn normalize_locator(value: &str) -> String {
    for (prefix, tag) in [
        ("https://huggingface.co/", "hf"),
        ("https://github.com/", "gh"),
    ] {
        if let Some(name) = value.strip_prefix(prefix).map(|s| s.trim_end_matches('/'))
            && repo(name)
        {
            return format!("{tag}:{name}");
        }
    }
    value.to_owned()
}
/// Browse one local folder without recursive traversal or following symlinks.
/// A directory containing model metadata is an observation, not source closure.
pub fn local(source: &Source, query: &str, page: usize, cancel: &AtomicBool) -> Result<Page> {
    check(source, cancel)?;
    if source.kind != Kind::Local {
        return Err(bad("local source required"));
    }
    let Some(root) = &source.scope else {
        return Ok(Page {
            entries: vec![],
            more: false,
            next_cursor: None,
        });
    };
    let mut entries = Vec::new();
    let needle = query.to_lowercase();
    for (index, item) in std::fs::read_dir(root)?.enumerate() {
        if cancel.load(Ordering::Acquire) {
            return Err(Error::State("discovery canceled".into()));
        }
        if index >= 65536 {
            return Err(Error::Limit(
                "local discovery inventory; select a narrower folder".into(),
            ));
        }
        let item = item?;
        let file_type = item.file_type()?;
        let name = item.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.chars().any(char::is_control) || !name.to_lowercase().contains(&needle) {
            continue;
        }
        let path = item.path();
        if !(file_type.is_file() && name.to_ascii_lowercase().ends_with(".gguf")
            || file_type.is_dir()
                && path
                    .join("config.json")
                    .symlink_metadata()
                    .is_ok_and(|m| m.is_file()))
        {
            continue;
        }
        let Some(locator) = path.to_str() else {
            continue;
        };
        entries.push(Entry {
            name: name.into(),
            locator: locator.into(),
            curated: None,
            payload_hold: false,
            bytes: if file_type.is_file() {
                Some(item.metadata()?.len())
            } else {
                None
            },
        });
        if entries.len() > 16384 {
            return Err(Error::Limit(
                "local discovery inventory; select a narrower folder".into(),
            ));
        }
    }
    entries.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    let offset = page
        .checked_mul(PAGE)
        .ok_or_else(|| bad("invalid discovery page"))?;
    let more = entries.len() > offset.saturating_add(PAGE);
    Ok(Page {
        entries: entries.into_iter().skip(offset).take(PAGE).collect(),
        more,
        next_cursor: None,
    })
}

#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod remote;
#[cfg(all(feature = "acquisition", target_os = "linux"))]
pub use remote::{File, Repository, browse, files};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_browse_is_scoped_paged_and_cancellable() {
        use std::fs;
        let root = std::env::temp_dir().join(format!("zg-discovery-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        for i in 0..33 {
            fs::write(root.join(format!("model-{i:02}.gguf")), b"fixture").unwrap();
        }
        fs::write(root.join("not-a-model.txt"), b"fixture").unwrap();
        fs::create_dir(root.join("directory")).unwrap();
        fs::write(root.join("directory/config.json"), b"{}").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(root.join("model-00.gguf"), root.join("link.gguf")).unwrap();
        let source = Source {
            name: "models".into(),
            kind: Kind::Local,
            scope: Some(root.to_str().unwrap().into()),
            enabled: true,
        };
        let cancel = AtomicBool::new(false);
        let first = local(&source, "", 0, &cancel).unwrap();
        assert_eq!(first.entries.len(), PAGE);
        assert!(first.more);
        assert_eq!(first.entries[0].name, "directory");
        let last = local(&source, "", 1, &cancel).unwrap();
        assert_eq!(last.entries.len(), 4);
        assert!(!last.more);
        let filtered = local(&source, "model-01", 0, &cancel).unwrap();
        assert_eq!(filtered.entries.len(), 1);
        assert_eq!(filtered.entries[0].bytes, Some(7));
        cancel.store(true, Ordering::Release);
        assert!(local(&source, "", 0, &cancel).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn configuration_and_curated_scope_are_independent_of_execution() {
        let mut source = defaults().remove(0);
        source.scope = Some("Qwen".into());
        assert!(curated(&source).len() > 1);
        assert!(curated(&source).iter().all(|e| e.name.starts_with("Qwen/")));
        assert!(validate(&[source.clone(), source.clone()]).is_err());
        source.scope = Some("https://example.com/".into());
        assert!(validate(&[source]).is_err());
        assert_eq!(
            normalize_locator("https://huggingface.co/Qwen/model/"),
            "hf:Qwen/model"
        );
        assert_eq!(
            normalize_locator("https://huggingface.co/Qwen/model/resolve/main/x.gguf"),
            "https://huggingface.co/Qwen/model/resolve/main/x.gguf"
        );
    }
}
