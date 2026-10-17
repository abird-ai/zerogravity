//! Explicit GGUF filename labels, never tensor-format eligibility or fallback.
use super::candidates::{CandidateRequest, MetadataCalls};
use super::http::{HttpTransport, check_cancel};
use super::{ExternalSidecars, Provider, PullLimits, ResolvedPullPlan};
use crate::cache::{FileRole, Manifest, SnapshotLease, TransferAdmission};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufQuantRequest {
    source: CandidateRequest,
    quant: String,
}
impl GgufQuantRequest {
    pub fn new(source: CandidateRequest, tag: &str) -> Result<Self> {
        if tag.is_empty()
            || tag.len() > 32
            || !tag.as_bytes()[0].is_ascii_alphabetic()
            || !tag.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(Error::Malformed("invalid GGUF quant filename tag".into()));
        }
        Ok(Self {
            source,
            quant: tag.to_ascii_uppercase(),
        })
    }
    pub fn source(&self) -> &CandidateRequest {
        &self.source
    }
    pub fn quant(&self) -> &str {
        &self.quant
    }
}
/// Provider-owned initial choice only: no fake Files request or second pin/tree.
#[derive(Clone, Copy)]
pub(super) enum InitialChoice<'a> {
    Files {
        paths: &'a [String],
        subfolder: Option<&'a str>,
    },
    Quant(&'a GgufQuantRequest),
}
fn missing() -> Error {
    Error::Unsupported(
        "requested GGUF quant is absent; inspect candidates or use an explicit file".into(),
    )
}
fn malformed() -> Error {
    Error::Malformed("invalid matching GGUF quant family".into())
}
fn matches_tag(stem: &str, tag: &str) -> bool {
    let leaf = stem.rsplit('/').next().unwrap_or(stem);
    if leaf.len() < tag.len() {
        return false;
    }
    let start = leaf.len() - tag.len();
    let Some(tail) = leaf.get(start..) else {
        return false;
    };
    tail.eq_ignore_ascii_case(tag)
        && (start == 0 || matches!(leaf.as_bytes()[start - 1], b'-' | b'.'))
}
#[derive(Ord, PartialOrd, Eq, PartialEq)]
enum Family {
    Single(String),
    Split(String, usize),
}
/// All regular inventory is checked before subfolder projection. Nonregular
/// selected members can never satisfy a split; source paths retain exact bytes.
pub(super) fn select(
    entries: &BTreeMap<String, bool>,
    request: &GgufQuantRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
) -> Result<BTreeSet<String>> {
    if entries.len() > limits.cache.max_files {
        return Err(Error::Limit("GGUF quant inventory count".into()));
    }
    let mut control = 0usize;
    for (path, regular) in entries {
        check_cancel(cancel)?;
        crate::cache::paths::validate_relative_path(path)?;
        control = control
            .checked_add(path.len() + 64)
            .filter(|n| *n <= limits.cache.max_manifest_bytes)
            .ok_or_else(|| Error::Limit("GGUF quant inventory control".into()))?;
        if *regular {
            for (at, _) in path.match_indices('/') {
                if entries.get(&path[..at]).copied().unwrap_or(false) {
                    return Err(malformed());
                }
            }
        }
    }
    let mut families = BTreeMap::<Family, BTreeSet<String>>::new();
    let mut totals = BTreeMap::<String, usize>::new();
    for (path, regular) in entries {
        check_cancel(cancel)?;
        if !request.source.includes(path) {
            continue;
        }
        let Some(stem) = path.strip_suffix(".gguf") else {
            continue;
        };
        // Split/tag grammar belongs to the filename, never a directory.
        // Family identity and the strict split parser still use full paths.
        let basename = stem.rsplit('/').next().unwrap_or(stem);
        // A terminal tag immediately followed by -of- lacks its ordinal;
        // do not silently drop that malformed matching cohort for another one.
        if basename
            .rsplit_once("-of-")
            .is_some_and(|(left, _)| matches_tag(left, request.quant()))
        {
            return Err(malformed());
        }
        let potential = basename
            .rsplit_once("-of-")
            .and_then(|(left, _)| left.rsplit_once('-'))
            .map(|(prefix, _)| prefix);
        let family = if potential.is_some_and(|prefix| matches_tag(prefix, request.quant())) {
            let (prefix, _, total) =
                super::selection::split_gguf_parts(path)?.ok_or_else(malformed)?;
            if total > limits.cache.max_files {
                return Err(Error::Limit("GGUF quant split count".into()));
            }
            if totals
                .insert(prefix.clone(), total)
                .is_some_and(|old| old != total)
            {
                return Err(malformed());
            }
            Family::Split(prefix, total)
        } else if matches_tag(basename, request.quant()) {
            Family::Single(path.clone())
        } else {
            continue;
        };
        if !regular {
            return Err(Error::Unsupported(
                "matching GGUF quant member is not regular".into(),
            ));
        }
        families.entry(family).or_default().insert(path.clone());
    }
    // Validate every matching cohort BEFORE ambiguity, never discard an
    // incomplete cohort to choose a complete one.
    for (family, selected) in &families {
        if let Family::Split(prefix, total) = family {
            if selected.len() != *total {
                return Err(Error::Unsupported(
                    "incomplete matching GGUF split family".into(),
                ));
            }
            for part in 1..=*total {
                if !selected.contains(&format!("{prefix}-{part:05}-of-{total:05}.gguf")) {
                    return Err(malformed());
                }
            }
        }
    }
    if families.len() > 1 {
        return Err(Error::Unsupported(
            "ambiguous GGUF quant families; narrow the subfolder or choose an explicit file".into(),
        ));
    }
    check_cancel(cancel)?;
    families.into_values().next().ok_or_else(missing)
}
fn limits_for_quant(limits: &PullLimits) -> Result<()> {
    limits.validate()?;
    if limits.unknown_length_cap.is_some() {
        return Err(Error::Unsupported(
            "GGUF quant requires exact model lengths".into(),
        ));
    }
    Ok(())
}
pub(super) fn metadata_admission(
    limits: &PullLimits,
    admission: &TransferAdmission,
) -> Result<zerogravity_core::resource_arbiter::GlobalResourceLease> {
    limits_for_quant(limits)?;
    let bytes = (limits.max_metadata_bytes as u64)
        .checked_mul(8)
        .and_then(|n| n.checked_add(16 * 1024 * 1024))
        .ok_or_else(|| Error::Limit("GGUF quant metadata admission".into()))?;
    admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes,
        }],
    )
}
pub fn resolve(
    transport: &impl HttpTransport,
    request: &GgufQuantRequest,
    limits: &PullLimits,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    check_cancel(cancel)?;
    let _lease = metadata_admission(limits, admission)?;
    let transport = MetadataCalls::new(transport, limits.max_pages);
    match request.source.provider() {
        Provider::HuggingFace => super::huggingface::resolve_quant_from(
            &transport,
            request,
            limits,
            cancel,
            "https://huggingface.co/",
        ),
        Provider::GitHub => super::github::resolve_quant(&transport, request, limits, cancel),
    }
}
/// New strict eligibility over a verified CACHE SUBSET only. Does not prove
/// unique quant family in the remote repository, freshness or native protocol.
/// Selected weights/config/tokenizer must carry their corresponding roles;
/// ordinary legacy Files remains role-agnostic and initial-selection-only.
pub fn validate_cached_selection(
    lease: &SnapshotLease,
    request: &GgufQuantRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
) -> Result<()> {
    limits_for_quant(limits)?;
    check_cancel(cancel)?;
    super::components::validate_cached_file_source(
        lease,
        request.source.provider(),
        request.source.repository(),
        request.source.revision(),
    )?;
    Manifest::new(
        lease.manifest().source().clone(),
        lease.manifest().entries().to_vec(),
        &limits.cache,
    )?;
    super::executor::reject_cached_pointers(lease, cancel)?;
    let entries = lease
        .manifest()
        .entries()
        .iter()
        .map(|e| (e.path().to_owned(), true))
        .collect();
    let mut selected = select(&entries, request, limits, cancel)?;
    // GGUF-only initial choice cannot read a safetensors index. Original shared
    // closure still owns complete splits and required/available sidecars.
    super::selection::close_model_selection(
        &entries,
        &mut selected,
        limits,
        ExternalSidecars::default(),
        &mut |_| {
            Err(Error::Unsupported(
                "GGUF quant cannot acquire a safetensors index".into(),
            ))
        },
    )?;
    for path in selected {
        check_cancel(cancel)?;
        let all = lease.manifest().entries();
        let entry = &all[all
            .binary_search_by(|e| e.path().cmp(&path))
            .map_err(|_| malformed())?];
        let required = if path.ends_with(".gguf") {
            Some(FileRole::Weights)
        } else {
            match path.rsplit('/').next().unwrap_or(&path) {
                "config.json" => Some(FileRole::Config),
                "tokenizer.json" => Some(FileRole::Tokenizer),
                _ => None,
            }
        };
        if required.is_some_and(|role| role != entry.role()) {
            return Err(Error::Malformed(
                "cached quant required role mismatch".into(),
            ));
        }
    }
    check_cancel(cancel)
}
