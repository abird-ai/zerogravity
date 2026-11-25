//! Selection metadata has no SourceIdentity or execution eligibility. Runtime
//! reopens exact immutable snapshots through the existing verified bridge.
use super::source::{GgufSource, require_geometry};
use crate::artifact::Artifact;
#[cfg(all(feature = "cache", target_os = "linux"))]
use crate::artifact::ArtifactInfo;
use crate::qwen35::config::ModelGeometry;
use crate::{Error, Result};
use std::path::Path;
enum Resolved {
    Local {
        weights: Artifact,
        config: Vec<u8>,
        tokenizer: std::fs::File,
        tokenizer_bytes: u64,
    },
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Cached {
        source: Box<super::source::CachedGguf>,
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
fn resolve(
    model: &crate::pipeline::ModelInstance,
    context: usize,
    cache_root: Option<&Path>,
) -> Result<Selection> {
    if !(1..=32768).contains(&context) {
        return Err(Error::Limit("CPU source planning context1..32768".into()));
    }
    match GgufSource::resolve(model, cache_root)? {
        GgufSource::Local(source) => {
            let (config, geometry) = source.read_config()?;
            let model_target = require_geometry(&geometry, Some(&model.target), context)?;
            let weights = source.open_weights()?.inspect()?;
            let tokenizer = std::fs::File::open(&source.tokenizer)?;
            let stamp = tokenizer.metadata()?;
            let tokenizer_bytes = stamp.len();
            if !stamp.is_file() || tokenizer_bytes > 64 * 1024 * 1024 {
                return Err(Error::Limit("regular tokenizer64MiB required".into()));
            }
            let planning_bytes = super::PreparedQwen::routing_bytes(
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
        #[cfg(all(feature = "cache", target_os = "linux"))]
        GgufSource::Cached(source) => {
            let (geometry, info) = source.inspect()?;
            let model_target = require_geometry(&geometry, Some(&model.target), context)?;
            let planning_bytes = super::PreparedQwen::routing_bytes(
                &info,
                &geometry,
                source.tokenizer.bytes(),
                context,
            )?;
            let raw_ticket_peak = info
                .file_len
                .checked_add(source.config.bytes())
                .and_then(|n| n.checked_add(source.tokenizer.bytes()))
                .and_then(|n| n.checked_add(64 * 1024))
                .ok_or_else(|| Error::Limit("raw source ticket planning overflow".into()))?;
            let source_label = format!("immutable cached snapshot {}", source.digest());
            Ok(Selection {
                context,
                resolved: Resolved::Cached {
                    source,
                    layout: info,
                },
                geometry,
                planning_bytes,
                model_target,
                source_label,
                raw_ticket_peak,
            })
        }
    }
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
        resolve(model, context, cache_root)
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
            Resolved::Cached { source, layout } => {
                let lease = source.open()?;
                crate::prepare::PreparedQwen::from_snapshots_with_layout(
                    &lease,
                    source.weights.path(),
                    &lease,
                    source.config.path(),
                    &lease,
                    source.tokenizer.path(),
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
