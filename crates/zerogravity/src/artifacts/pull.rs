//! Hub selection/orchestration. Original transports, source pins and publisher
//! remain in artifacts; callers own credentials, cancellation and progress IO.
use super::{ArtifactService, acquisition, cache};
use crate::{Error, Result};
use acquisition::{Provider, PullLimits, PullRequest, ResolvedPullPlan, Selection};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy)]
pub enum HubSelection<'a> {
    Files(&'a PullRequest),
    Quant(&'a acquisition::GgufQuantRequest),
    Filters(&'a acquisition::filters::FilterRequest),
}
impl<'a> HubSelection<'a> {
    pub fn provider(self) -> Provider {
        match self {
            Self::Files(r) => r.provider,
            Self::Quant(r) => r.source().provider(),
            Self::Filters(r) => r.source().provider(),
        }
    }
    fn repository(self) -> &'a str {
        match self {
            Self::Files(r) => &r.repository,
            Self::Quant(r) => r.source().repository(),
            Self::Filters(r) => r.source().repository(),
        }
    }
    fn revision(self) -> &'a str {
        match self {
            Self::Files(r) => &r.revision,
            Self::Quant(r) => r.source().revision(),
            Self::Filters(r) => r.source().revision(),
        }
    }
    fn default_reference(self) -> String {
        let provider = if matches!(self, Self::Filters(r) if r.is_bundle()) {
            match self.provider() {
                Provider::HuggingFace => "huggingface-bundle",
                Provider::GitHub => "github-bundle",
            }
        } else {
            self.provider().label()
        };
        format!("{provider}/{}/{}", self.repository(), self.revision())
    }
}

/// Metadata-only Xet resolution cannot acquire native index reconstruction
/// grants, even when supplied options previously enabled them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PullMode {
    Inspect,
    Transfer,
}

pub struct HubPull<'a> {
    runtime: &'a zerogravity_engine::Runtime,
    selection: HubSelection<'a>,
    limits: PullLimits,
    reference: String,
    native: Option<acquisition::xet::XetOptions>,
}
pub struct PreparedHubPull<'a> {
    runtime: &'a zerogravity_engine::Runtime,
    plan: ResolvedPullPlan,
    reference: String,
    mode: PullMode,
}
impl<'a> ArtifactService<'a> {
    /// Cold validation only. No cache, credentials, admission or transport IO.
    pub fn hub(
        &self,
        selection: HubSelection<'a>,
        limits: PullLimits,
        reference: Option<String>,
        native: Option<acquisition::xet::XetOptions>,
    ) -> Result<HubPull<'a>> {
        self.runtime.require_open()?;
        limits.validate()?;
        if let HubSelection::Files(request) = selection {
            request.validate(&limits)?;
        }
        if let Some(native) = &native {
            native.validate()?;
            if matches!(selection, HubSelection::Filters(r) if r.is_bundle()) {
                return Err(Error::Unsupported("native Xet bundle selection not qualified; use original HF bundle or exact native model selection".into()));
            }
            if selection.provider() != Provider::HuggingFace
                || matches!(selection, HubSelection::Files(r) if !matches!(r.selection, Selection::Files { .. }))
                || limits.unknown_length_cap.is_some()
            {
                return Err(Error::Malformed(
                    "native Xet requires bounded HF file selection".into(),
                ));
            }
        }
        let reference = reference.unwrap_or_else(|| selection.default_reference());
        if reference.is_empty() || reference.len() > 4096 || reference.chars().any(char::is_control)
        {
            return Err(Error::Malformed("invalid acquisition reference".into()));
        }
        Ok(HubPull {
            runtime: self.runtime,
            selection,
            limits,
            reference,
            native,
        })
    }
}
impl<'a> HubPull<'a> {
    pub fn reference(&self) -> &str {
        &self.reference
    }
    /// Preserve exact cached-selection validation without transport or admission.
    /// Filter/quant selections require an explicit snapshot digest, not an alias.
    pub fn cached(
        &self,
        store: &cache::CacheStore,
        cancel: &AtomicBool,
    ) -> Result<cache::SnapshotLease> {
        self.runtime.require_open()?;
        check_cancel(cancel)?;
        if self.limits.unknown_length_cap.is_some() {
            return Err(Error::Malformed(
                "unknown-length transfer cap is online-only".into(),
            ));
        }
        let lease = if matches!(
            self.selection,
            HubSelection::Quant(_) | HubSelection::Filters(_)
        ) {
            cache::paths::validate_digest(&self.reference)?;
            store.open_snapshot(&self.reference)?
        } else {
            super::cached(store, &self.reference)?
        };
        let manifest = lease.manifest();
        let bounds = &self.limits.cache;
        if manifest.entries().len() > bounds.max_files
            || manifest.total_bytes() > bounds.max_snapshot_bytes
            || manifest
                .entries()
                .iter()
                .any(|entry| entry.bytes() > bounds.max_file_bytes)
            || manifest.canonical_bytes()?.len() > bounds.max_manifest_bytes
        {
            return Err(Error::Limit(
                "cached selection exceeds caller bounds".into(),
            ));
        }
        match self.selection {
            HubSelection::Filters(request) => acquisition::filters::validate_cached_selection(
                &lease,
                request,
                &self.limits,
                cancel,
            )?,
            HubSelection::Quant(request) => acquisition::quant::validate_cached_selection(
                &lease,
                request,
                &self.limits,
                cancel,
            )?,
            HubSelection::Files(request) if self.native.is_some() => {
                acquisition::xet::validate_cached_selection(&lease, request, &self.limits, cancel)?
            }
            HubSelection::Files(request) => {
                acquisition::validate_cached_selection(&lease, request)?
            }
        }
        check_cancel(cancel)?;
        Ok(lease)
    }

    pub fn resolve(
        &self,
        transport: &impl acquisition::http::HttpTransport,
        admission: &cache::TransferAdmission,
        cancel: &AtomicBool,
        mode: PullMode,
    ) -> Result<PreparedHubPull<'a>> {
        self.runtime.require_open()?;
        check_cancel(cancel)?;
        let native = self.native.clone().map(|options| match mode {
            PullMode::Inspect => options.without_index_reconstruction(),
            PullMode::Transfer => options.with_index_reconstruction(),
        });
        let limits = &self.limits;
        let plan = match (self.selection, native.as_ref()) {
            (HubSelection::Files(r), None) => {
                acquisition::resolve(transport, r, limits, admission, cancel)
            }
            (HubSelection::Quant(r), None) => {
                acquisition::quant::resolve(transport, r, limits, admission, cancel)
            }
            (HubSelection::Filters(r), None) => {
                acquisition::filters::resolve(transport, r, limits, admission, cancel)
            }
            (HubSelection::Files(r), Some(n)) => {
                acquisition::xet::resolve(transport, r, limits, n, admission, cancel)
            }
            (HubSelection::Quant(r), Some(n)) => {
                acquisition::xet::resolve_quant(transport, r, limits, n, admission, cancel)
            }
            (HubSelection::Filters(r), Some(n)) => {
                acquisition::xet::resolve_filtered(transport, r, limits, n, admission, cancel)
            }
        }?;
        Ok(PreparedHubPull {
            runtime: self.runtime,
            plan,
            reference: self.reference.clone(),
            mode,
        })
    }
}
impl PreparedHubPull<'_> {
    pub fn plan(&self) -> &ResolvedPullPlan {
        &self.plan
    }
    pub fn reference(&self) -> &str {
        &self.reference
    }
    /// The original receipt distinguishes durable and durability-unknown states.
    /// Inspection never silently upgrades to transfer authorization.
    pub fn execute(
        &self,
        store: &cache::CacheStore,
        transport: &impl acquisition::http::HttpTransport,
        admission: &cache::TransferAdmission,
        cancel: &Arc<AtomicBool>,
        progress: &mut impl FnMut(&acquisition::PullProgress<'_>),
    ) -> Result<acquisition::PullReceipt> {
        self.runtime.require_open()?;
        if self.mode != PullMode::Transfer {
            return Err(Error::Unsupported(
                "inspection plan cannot transfer source payloads".into(),
            ));
        }
        acquisition::execute_with_progress(
            store,
            &self.plan,
            transport,
            admission,
            cancel,
            Some(&self.reference),
            progress,
        )
    }
}
fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::State("acquisition canceled".into()))
    } else {
        Ok(())
    }
}
