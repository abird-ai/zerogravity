//! Shared startup selection declarations. No context, allocator, source loader,
//! network, neural fallback or completion authority lives in this module.
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
}
impl Backend {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "auto" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "rocm" => Self::Rocm,
            "vulkan" => Self::Vulkan,
            _ => return Err(Error::Unsupported(format!("unregistered run backend {s}"))),
        })
    }
    pub fn label(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Vulkan => "vulkan",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceSelector {
    Local(PathBuf),
    Catalog(String),
    HuggingFace(String),
    GitHub(String),
    Http(String),
}
impl SourceSelector {
    pub fn parse(s: &str) -> Result<Self> {
        if s.is_empty() || s.len() > 4096 || s.chars().any(char::is_control) {
            return Err(Error::Malformed(
                "bounded nonempty model source required".into(),
            ));
        }
        if let Some(rest) = s.strip_prefix("hf:") {
            if rest.is_empty() {
                return Err(Error::Malformed("hf: repository required".into()));
            }
            return Ok(Self::HuggingFace(rest.into()));
        }
        if let Some(rest) = s.strip_prefix("gh:") {
            if rest.is_empty() {
                return Err(Error::Malformed("gh: repository required".into()));
            }
            return Ok(Self::GitHub(rest.into()));
        }
        if let Some(reference) = s.strip_prefix("catalog:") {
            return catalog_reference(reference);
        }
        if s.starts_with("http:") || s.starts_with("https:") {
            return Ok(Self::Http(s.into()));
        }
        if s.contains("://") {
            return Err(Error::Unsupported(
                "source scheme unavailable; use an exact local acquired source".into(),
            ));
        }
        let p = std::path::Path::new(s);
        // An absolute or explicitly relative path is always local, including
        // a local directory named like a provider catalogue namespace.
        if p.is_absolute() || s.starts_with('.') {
            return Ok(Self::Local(p.into()));
        }
        // `models pull` default references are <namespace>/REPOSITORY/REVISION.
        if let Some((namespace, reference)) = s.split_once('/')
            && !reference.is_empty()
            && matches!(
                namespace,
                "huggingface" | "huggingface-bundle" | "github" | "github-bundle"
            )
        {
            return catalog_reference(s);
        }
        if s.contains('/')
            || p.extension().is_some_and(|e| {
                matches!(
                    e.to_str(),
                    Some("gguf" | "safetensors" | "json" | "yaml" | "yml")
                )
            })
        {
            Ok(Self::Local(p.into()))
        } else {
            Ok(Self::Catalog(s.into()))
        }
    }
    /// Positional run-document classification shared by the run frontend.
    /// Only a local filesystem path whose final component is .yaml/.yml
    /// selects a run document; provider/catalog/URL schemes are sources
    /// regardless of any file-extension spelling. Invalid selectors fail
    /// here with their exact selector errors, before any config load.
    pub fn implies_run_document(s: &str) -> Result<bool> {
        Ok(matches!(
            Self::parse(s)?,
            Self::Local(p)
                if p.extension().is_some_and(|e| e == "yaml" || e == "yml")
        ))
    }
    /// Narrow provider-scheme prefix test only; NOT the run-document
    /// classification authority. Positional model/config classification
    /// derives from the full shared selector (`implies_run_document`) so
    /// pull-shaped catalog namespaces and unavailable schemes never load
    /// as run documents.
    pub fn has_provider_scheme(s: &str) -> bool {
        ["hf:", "gh:", "catalog:", "http:", "https:"]
            .iter()
            .any(|prefix| s.starts_with(prefix))
    }
}
/// Same bounded name rules as an existing cache reference; no filesystem or
/// provider fallback is attempted for either spelling.
fn catalog_reference(reference: &str) -> Result<SourceSelector> {
    if reference.is_empty() || reference.len() > 4096 || reference.chars().any(char::is_control) {
        return Err(Error::Malformed(
            "bounded nonempty catalog reference required".into(),
        ));
    }
    Ok(SourceSelector::Catalog(reference.into()))
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entrypoint {
    Model(String),
    Pipeline(String),
}
impl Entrypoint {
    pub fn parse(s: &str) -> Result<Self> {
        let (kind, id) = s.split_once(':').ok_or_else(|| {
            Error::Malformed("entrypoint requires model:ID or pipeline:NAME".into())
        })?;
        if id.is_empty() || id.len() > 128 || id.chars().any(char::is_control) {
            return Err(Error::Malformed("invalid entrypoint identity".into()));
        }
        match kind {
            "model" => Ok(Self::Model(id.into())),
            "pipeline" => Ok(Self::Pipeline(id.into())),
            _ => Err(Error::Malformed(
                "entrypoint requires model:ID or pipeline:NAME".into(),
            )),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRef {
    pub host: String,
    pub selector: String,
}
impl DeviceRef {
    pub fn parse(s: &str) -> Result<Self> {
        // Absolute device nodes select the local physical GPU. Keep the path
        // literal until host inventory binds its character-device identity to
        // the existing PCI resource domain (no device open or fallback here).
        let (host, selector) = if s.starts_with("/dev/") {
            ("local", s)
        } else {
            s.split_once('/').unwrap_or(("local", s))
        };
        // CPU has exactly one local device. Keep GPUs physical/explicit:
        // an unqualified "gpu" shortcut would hide device identity.
        let selector = if selector == "cpu" { "cpu:0" } else { selector };
        if host.is_empty()
            || host.len() > 128
            || !host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            || selector.is_empty()
            || selector.len() > 256
            || selector.chars().any(char::is_control)
        {
            return Err(Error::Malformed("invalid bounded device reference".into()));
        }
        let local_node = selector.starts_with("/dev/")
            && selector
                .split('/')
                .skip(1)
                .all(|component| !component.is_empty() && component != "." && component != "..");
        if selector.starts_with("/dev/") {
            if !local_node {
                return Err(Error::Malformed(
                    "device node path must be absolute and normalized".into(),
                ));
            }
        } else if !["cpu:", "gpu:", "npu:", "pci:", "luid:", "registry:"]
            .iter()
            .any(|prefix| selector.starts_with(prefix) && selector.len() > prefix.len())
        {
            return Err(Error::Malformed("unknown physical device selector".into()));
        }
        if let Some((class, index)) = selector.split_once(':')
            && matches!(class, "cpu" | "gpu" | "npu")
        {
            index
                .parse::<usize>()
                .map_err(|_| Error::Malformed("device class index must be an integer".into()))?;
        }
        Ok(Self {
            host: host.into(),
            selector: selector.into(),
        })
    }
    pub fn label(&self) -> String {
        format!("{}/{}", self.host, self.selector)
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Devices {
    Auto,
    Ordered(Vec<DeviceRef>),
}
impl Devices {
    pub fn parse(values: &[String]) -> Result<Self> {
        let parts = values.iter().flat_map(|s| s.split(',')).collect::<Vec<_>>();
        if parts == ["auto"] {
            return Ok(Self::Auto);
        }
        if parts.is_empty()
            || parts.len() > 64
            || parts.iter().any(|s| s.is_empty() || *s == "auto")
        {
            return Err(Error::Malformed(
                "device auto must stand alone; list is bounded and nonempty".into(),
            ));
        }
        let mut result = Vec::new();
        for value in parts {
            let device = DeviceRef::parse(value)?;
            if !result.contains(&device) {
                result.push(device);
            }
        }
        if let Some(index) = result.iter().position(|d| d.selector.starts_with("cpu:"))
            && result[index + 1..]
                .iter()
                .any(|d| !d.selector.starts_with("cpu:"))
        {
            return Err(Error::Malformed(
                "CPU must be terminal in ordered devices".into(),
            ));
        }
        Ok(Self::Ordered(result))
    }
    pub fn validate_backend(&self, backend: &Backend) -> Result<()> {
        if let Self::Ordered(devices) = self {
            for d in devices {
                if d.selector.starts_with("cpu:")
                    && !matches!(backend, Backend::Auto | Backend::Cpu)
                {
                    return Err(Error::Malformed(
                        "forced GPU backend contradicts explicit CPU compute".into(),
                    ));
                }
                if !d.selector.starts_with("cpu:") && *backend == Backend::Cpu {
                    return Err(Error::Malformed(
                        "run-cpu contradicts explicit accelerator compute".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Headroom {
    Default,
    Percent(u8),
    Bytes(u64),
}
impl Headroom {
    pub fn parse(s: &str) -> Result<Self> {
        if let Some(p) = s.strip_suffix('%') {
            return p
                .parse::<u8>()
                .ok()
                .filter(|p| *p < 100)
                .map(Self::Percent)
                .ok_or_else(|| Error::Malformed("headroom must be a finite0..99 percent".into()));
        }
        for (suffix, mult) in [
            ("GiB", 1024u64.pow(3)),
            ("MiB", 1024u64.pow(2)),
            ("KiB", 1024u64),
            ("B", 1),
        ] {
            if let Some(n) = s.strip_suffix(suffix) {
                return n
                    .parse::<u64>()
                    .ok()
                    .and_then(|n| n.checked_mul(mult))
                    .map(Self::Bytes)
                    .ok_or_else(|| Error::Limit("invalid or overflowing headroom bytes".into()));
            }
        }
        s.parse::<u64>().map(Self::Bytes).map_err(|_| {
            Error::Malformed("headroom requires percent or integer bytes/KiB/MiB/GiB".into())
        })
    }
    pub fn bytes(&self, ceiling: u64, host: bool) -> Result<u64> {
        Ok(match self {
            Self::Default => (ceiling / 10).max(if host {
                1024u64.pow(3)
            } else {
                256 * 1024u64.pow(2)
            }),
            Self::Percent(p) => {
                ceiling / 100 * u64::from(*p) + (ceiling % 100) * u64::from(*p) / 100
            }
            Self::Bytes(n) => *n,
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Speculation {
    Auto,
    Off,
    Mtp,
    Draft(String),
}
impl Speculation {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            "mtp" => Ok(Self::Mtp),
            _ => s
                .strip_prefix("draft:")
                .filter(|s| !s.is_empty())
                .map(|s| Self::Draft(s.into()))
                .ok_or_else(|| {
                    Error::Malformed("speculation requires auto/off/mtp/draft:MODEL".into())
                }),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    pub prompt: Option<String>,
    pub context: usize,
    pub max_tokens: usize,
    pub raw: bool,
    pub teacher_ids: Option<Vec<u32>>,
    pub prefill_chunk: usize,
}
impl Default for Task {
    fn default() -> Self {
        Self {
            prompt: None,
            context: 32768,
            max_tokens: 16,
            raw: false,
            teacher_ids: None,
            prefill_chunk: 128,
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Diagnostics {
    pub explain: bool,
    pub dry_run: bool,
    pub audit_logits_jsonl: Option<PathBuf>,
    pub audit_layers_jsonl: Option<PathBuf>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    Cli,
    Config(PathBuf),
    Builtin,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub graph: crate::pipeline::Config,
    pub entrypoint: Entrypoint,
    pub task: Task,
    pub backend: Backend,
    pub devices: Devices,
    pub headroom: Headroom,
    pub numerical: Option<String>,
    pub speculation: Speculation,
    pub draft_device: Option<Devices>,
    pub diagnostics: Diagnostics,
    pub provenance: BTreeMap<String, Origin>,
    pub vendor_filter: Option<String>,
}
impl Request {
    pub fn validate_policy(&self) -> Result<()> {
        self.devices.validate_backend(&self.backend)?;
        for m in &self.graph.models {
            for path in [&m.model_config, &m.tokenizer, &m.native_artifacts]
                .into_iter()
                .flatten()
            {
                if path.is_empty() || path.len() > 4096 || path.chars().any(char::is_control) {
                    return Err(Error::Malformed(
                        "bounded nonempty model sidecar/native path required".into(),
                    ));
                }
            }
            if m.snapshot.as_ref().is_some_and(|s| {
                s.len() != 64
                    || !s
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            }) {
                return Err(Error::Malformed(
                    "model snapshot must be an exact lowercase SHA256".into(),
                ));
            }
        }
        if self.task.context == 0
            || self.task.context > 32768
            || self.task.max_tokens == 0
            || self.task.max_tokens > 128
            || self.task.prefill_chunk == 0
            || self.task.prefill_chunk > 32768
            || self
                .task
                .prompt
                .as_ref()
                .is_some_and(|p| p.len() > 1024 * 1024)
        {
            return Err(Error::Limit(
                "bounded context/output/prompt/prefill request".into(),
            ));
        }
        if let Some(ids) = &self.task.teacher_ids
            && (ids.len() != self.task.max_tokens || self.diagnostics.audit_logits_jsonl.is_none())
        {
            return Err(Error::Malformed(
                "teacher IDs require matching output count and audit logits destination".into(),
            ));
        }
        if let Some(draft) = &self.draft_device {
            draft.validate_backend(&self.backend)?;
            let Devices::Ordered(draft) = draft else {
                return Err(Error::Malformed(
                    "draft device requires explicit subset of run devices".into(),
                ));
            };
            let Devices::Ordered(all) = &self.devices else {
                return Err(Error::Malformed(
                    "draft device requires explicit whole-run allowlist".into(),
                ));
            };
            if draft.iter().any(|d| !all.contains(d)) {
                return Err(Error::Malformed(
                    "draft device must be in whole-run allowlist".into(),
                ));
            }
        }
        Ok(())
    }
}
