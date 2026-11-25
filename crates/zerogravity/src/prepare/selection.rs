//! Selection metadata has no SourceIdentity or execution eligibility. Runtime
//! reopens exact immutable snapshots through the existing verified bridge.
use crate::artifact::Artifact;
#[cfg(all(feature = "cache", target_os = "linux"))]
use crate::artifact::ArtifactInfo;
use crate::qwen35::config::ModelGeometry;
use crate::{Error, Result, run_policy::SourceSelector};
use std::io::Read;
use std::path::{Path, PathBuf};
enum Resolved {
    Local {
        weights: Artifact,
        config: Vec<u8>,
        tokenizer: std::fs::File,
        tokenizer_bytes: u64,
    },
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Cached {
        root: PathBuf,
        disk: u64,
        snapshot: String,
        weights: String,
        config: String,
        tokenizer: String,
        layout: ArtifactInfo,
    },
}
pub struct Selection {
    context: usize,
    resolved: Resolved,
    geometry: ModelGeometry,
    planning_bytes: u64,
    model_target: String,
    source_label: String,
    raw_ticket_peak: u64,
}
fn bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let f = std::fs::File::open(path)?;
    let m = f.metadata()?;
    if !m.is_file() || m.len() > limit {
        return Err(Error::Limit("source routing metadata size".into()));
    }
    let mut bytes = Vec::with_capacity(m.len() as usize);
    f.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != m.len() {
        return Err(Error::State("source metadata changed while reading".into()));
    }
    Ok(bytes)
}
fn resolve(
    model: &crate::pipeline::ModelInstance,
    context: usize,
    cache_root: Option<&Path>,
) -> Result<Selection> {
    if !(1..=32768).contains(&context) {
        return Err(Error::Limit("CPU source planning context1..32768".into()));
    }
    let selector = SourceSelector::parse(model.source.as_deref().ok_or_else(|| {
        Error::Malformed("selected model requires an explicit acquired source".into())
    })?)?;
    match &selector {
        SourceSelector::HuggingFace(value) | SourceSelector::GitHub(value) => {
            #[cfg(all(feature = "cache", target_os = "linux"))]
            {
                let provider = if matches!(selector, SourceSelector::HuggingFace(_)) {
                    "huggingface"
                } else {
                    "github"
                };
                crate::cache::provenance::ProviderLocator::parse(provider, value)?;
            }
            #[cfg(not(all(feature = "cache", target_os = "linux")))]
            {
                let _ = value;
                return Err(Error::Unsupported("provider cache selection requires Linux and cache; no acquisition is performed".into()));
            }
        }
        SourceSelector::Http(url) => {
            #[cfg(feature = "acquisition")]
            crate::acquisition::object::HttpsObjectSpec::validate_cached_locator(url)?;
            #[cfg(not(feature = "acquisition"))]
            {
                let _ = url;
                return Err(Error::Unsupported("HTTP source form requires compiled acquisition contract parser; no request is performed".into()));
            }
        }
        _ => {}
    }
    match selector {
        SourceSelector::Local(weights) => {
            if model.snapshot.is_some() {
                return Err(Error::Malformed(
                    "snapshot is for cached provider/catalog sources".into(),
                ));
            }
            let parent = weights.parent().unwrap_or(Path::new("."));
            let config = model
                .model_config
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| parent.join("config.json"));
            let tokenizer = model
                .tokenizer
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| parent.join("tokenizer.json"));
            let config = bounded(&config, 1024 * 1024)?;
            let geometry = ModelGeometry::from_json(&config)?;
            let model_target = crate::prepare::PreparedQwen::routing_target(&geometry)?;
            let weights = Artifact::open(&weights)?;
            let tokenizer = std::fs::File::open(tokenizer)?;
            let stamp = tokenizer.metadata()?;
            let tokenizer_bytes = stamp.len();
            if !stamp.is_file() || tokenizer_bytes > 64 * 1024 * 1024 {
                return Err(Error::Limit("regular tokenizer64MiB required".into()));
            }
            let planning_bytes = crate::prepare::PreparedQwen::routing_bytes(
                &weights.info,
                &geometry,
                tokenizer_bytes,
                context,
            )?;
            Ok(Selection {
                context,
                resolved: Resolved::Local {
                    weights,
                    config,
                    tokenizer,
                    tokenizer_bytes,
                },
                geometry,
                planning_bytes,
                model_target,
                source_label:
                    "local original GGUF; same inspected descriptor retained through preparation"
                        .into(),
                raw_ticket_peak: 0,
            })
        }
        other => cached(model, other, context, cache_root),
    }
}
#[cfg(not(all(feature = "cache", target_os = "linux")))]
fn cached(
    _: &crate::pipeline::ModelInstance,
    _: SourceSelector,
    _: usize,
    _: Option<&Path>,
) -> Result<Selection> {
    Err(Error::Unsupported("cached provider/catalog run requires Linux and the cache feature; acquire an exact source explicitly using models/pull".into()))
}
#[cfg(all(feature = "cache", target_os = "linux"))]
fn cached(
    model: &crate::pipeline::ModelInstance,
    selector: SourceSelector,
    context: usize,
    cache_root: Option<&Path>,
) -> Result<Selection> {
    use crate::cache::{CacheMetadata, FileRole};
    if model.model_config.is_some() || model.tokenizer.is_some() {
        return Err(Error::Malformed(
            "cached run requires role-bound config/tokenizer in the selected immutable snapshot; local sidecar overrides would break provenance".into(),
        ));
    }
    let root = crate::cache::paths::resolve_root(
        cache_root,
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?;
    let metadata = CacheMetadata::open(&root)?;
    let snapshot = metadata.select(&selector, model.snapshot.as_deref())?;
    let unique = |role| -> Result<String> {
        let entries = snapshot
            .manifest
            .entries()
            .iter()
            .filter(|e| e.role() == role)
            .collect::<Vec<_>>();
        if entries.len() != 1 {
            return Err(Error::Malformed(
                "selected snapshot needs exactly one weights/config/tokenizer role; select a complete unambiguous acquired snapshot".into(),
            ));
        }
        Ok(entries[0].path().into())
    };
    let weights = unique(FileRole::Weights)?;
    let config = unique(FileRole::Config)?;
    let tokenizer = unique(FileRole::Tokenizer)?;
    let geometry = ModelGeometry::from_json(&metadata.config(&snapshot, &config)?)?;
    let model_target = crate::prepare::PreparedQwen::routing_target(&geometry)?;
    let info = metadata.inspect_weights(&snapshot, &weights)?;
    let tokenizer_bytes = snapshot
        .manifest
        .entries()
        .iter()
        .find(|e| e.path() == tokenizer)
        .expect("unique role")
        .bytes();
    if tokenizer_bytes > 64 * 1024 * 1024 {
        return Err(Error::Limit("tokenizer64MiB".into()));
    }
    let planning_bytes =
        crate::prepare::PreparedQwen::routing_bytes(&info, &geometry, tokenizer_bytes, context)?;
    let raw_ticket_peak = info
        .file_len
        .checked_add(
            snapshot
                .manifest
                .entries()
                .iter()
                .find(|e| e.path() == config)
                .expect("unique config")
                .bytes(),
        )
        .and_then(|n| n.checked_add(tokenizer_bytes))
        .and_then(|n| n.checked_add(64 * 1024))
        .ok_or_else(|| Error::Limit("raw source ticket planning overflow".into()))?;
    let disk = metadata.disk_limit_bytes()?;
    Ok(Selection {
        context,
        resolved: Resolved::Cached {
            root,
            disk,
            snapshot: snapshot.digest.clone(),
            weights,
            config,
            tokenizer,
            layout: info,
        },
        geometry,
        planning_bytes,
        model_target,
        source_label: format!("immutable cached snapshot {}", snapshot.digest),
        raw_ticket_peak,
    })
}

/// Materialized source plus the original full CPU planning envelope. The source
/// drops before its charge; execution cannot exceed the planned context.
pub struct PreparedCpu {
    pub(crate) prepared: super::PreparedQwen,
    pub(crate) envelope: crate::resource_arbiter::GlobalResourceLease,
    pub(crate) context: usize,
}
impl Selection {
    /// Read-only routing metadata, not a source issuer or execution permission.
    pub fn resolve(
        model: &crate::pipeline::ModelInstance,
        context: usize,
        cache_root: Option<&Path>,
    ) -> Result<Self> {
        let selection = resolve(model, context, cache_root)?;
        if !model.target.is_empty() && model.target != selection.model_target {
            return Err(Error::Malformed(
                "declared model target differs from actual source geometry".into(),
            ));
        }
        if context as u64 > selection.geometry.max_position_embeddings {
            return Err(Error::Limit(
                "requested context exceeds actual model geometry".into(),
            ));
        }
        Ok(selection)
    }
    /// Fresh host availability gates the original process-shared fixed domain;
    /// free memory/headroom never become a replacement operator ceiling.
    pub fn prepare_on_host(
        self,
        headroom: crate::run_policy::Headroom,
        fixed_ceiling: u64,
        available_after_margin: u64,
    ) -> Result<PreparedCpu> {
        use crate::resource_arbiter::{
            PhysicalDeviceKey, ResourceArbiter, ResourceOwner, TransferAdmission,
        };
        let arbiter = ResourceArbiter::process_shared();
        let key = PhysicalDeviceKey::new("host", "system-ram")?;
        let host = arbiter.observe_host_backing()?;
        let (ceiling, available) = host.ceiling.zip(host.available).ok_or_else(|| {
            Error::State("fresh HOST capacity/current availability unavailable".into())
        })?;
        let fixed_ceiling = fixed_ceiling.min(ceiling);
        let backing = arbiter.host_backing()?;
        let margin = headroom.bytes(ceiling, true)?;
        // The old proposal constrains the new observation. Never grow a cap
        // from free memory or add available bytes to logical reservations.
        let usable = backing
            .conservative_available(fixed_ceiling, available, margin)
            .min(available_after_margin.saturating_sub(backing.reserved));
        if self.planning_bytes() > usable {
            return Err(Error::resource(
                "host memory",
                "direct CPU cold fit",
                self.planning_bytes(),
                usable,
            ));
        }
        // Only now establish the scoped operator limit for an actual load.
        if arbiter.snapshot(&key).is_err() {
            arbiter.register(key.clone(), fixed_ceiling)?;
        }
        let budget = arbiter.snapshot(&key)?;
        if budget.fenced {
            return Err(Error::State("original HOST admission is fenced".into()));
        }
        if self.planning_bytes() > budget.remaining {
            return Err(Error::resource(
                "host memory",
                "direct CPU scoped admission",
                self.planning_bytes(),
                budget.remaining,
            ));
        }

        self.prepare(&TransferAdmission {
            arbiter,
            resource: key,
            owner: ResourceOwner::new("fabric-run", "direct-cpu")?,
        })
    }
    pub fn geometry(&self) -> &ModelGeometry {
        &self.geometry
    }
    pub fn planning_bytes(&self) -> u64 {
        self.planning_bytes
    }
    pub fn model_target(&self) -> &str {
        &self.model_target
    }
    pub fn source_label(&self) -> &str {
        &self.source_label
    }
    pub fn prepare(
        self,
        admission: &crate::resource_arbiter::TransferAdmission,
    ) -> Result<PreparedCpu> {
        use crate::resource_arbiter::{AllocationClass, AllocationRequest};
        if self.context as u64 > self.geometry.max_position_embeddings {
            return Err(Error::Limit(
                "requested context exceeds actual model geometry".into(),
            ));
        }
        let envelope = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::ProviderEnvelope,
                bytes: self
                    .planning_bytes
                    .checked_sub(self.raw_ticket_peak)
                    .ok_or_else(|| Error::State("raw tickets exceed planning envelope".into()))?,
            }],
        )?;
        let prepared = match self.resolved {
            Resolved::Local {
                weights,
                config,
                mut tokenizer,
                tokenizer_bytes,
            } => {
                use std::io::Read;
                let mut bytes = Vec::new();
                (&mut tokenizer)
                    .take(tokenizer_bytes + 1)
                    .read_to_end(&mut bytes)?;
                if bytes.len() as u64 != tokenizer_bytes {
                    return Err(Error::State(
                        "held tokenizer changed before preparation".into(),
                    ));
                }
                crate::prepare::PreparedQwen::from_inspected_gguf(weights, &config, &bytes)?
            }
            #[cfg(all(feature = "cache", target_os = "linux"))]
            Resolved::Cached {
                root,
                disk,
                snapshot,
                weights,
                config,
                tokenizer,
                layout,
            } => {
                let store = crate::cache::CacheStore::open_existing(
                    crate::cache::CacheOptions::new(Some(root), disk),
                )?;
                let lease = store.open_snapshot(&snapshot)?;
                crate::prepare::PreparedQwen::from_snapshots_with_layout(
                    &lease,
                    &weights,
                    &lease,
                    &config,
                    &lease,
                    &tokenizer,
                    admission,
                    Some(&layout),
                )?
            }
        };
        Ok(PreparedCpu {
            prepared,
            envelope,
            context: self.context,
        })
    }
}
