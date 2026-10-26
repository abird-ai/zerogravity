//! SAME source/compiled-layout inventory consumed before original allocation.
use super::{super::text_model::TextModelBinding, packets::Role};
use std::{collections::BTreeSet, sync::Arc};
use zerogravity_core::{Error, Result};
const MIB: usize = 1024 * 1024;
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error::Limit("text native inventory sum".into()))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Limit("text native inventory product".into()))
}
fn sum(items: impl IntoIterator<Item = usize>) -> Result<usize> {
    items.into_iter().try_fold(0, add)
}
pub(crate) struct TextNativeInventory {
    pub(crate) device_layout: Vec<usize>,
    pub(crate) retained_host_bytes: usize,
    pub(crate) initialization_host_bytes: usize,
    pub(crate) provider_bytes: usize,
    pub(crate) rotary_bytes: usize,
    pub(super) private_layout: Vec<(Role, usize)>,
    pub(super) catalogue_bytes: usize,
}
impl TextNativeInventory {
    pub(crate) fn for_scopes(
        capture: super::TextKernelCapture,
        scopes: &[Arc<TextModelBinding>],
        audit_head: bool,
        cohort_b4: bool,
        head_mode: super::super::text_head::HeadMode,
    ) -> Result<Self> {
        let first = scopes
            .first()
            .ok_or_else(|| Error::State("text inventory missing genuine source scopes".into()))?;
        capture.require_cold_scopes(scopes.iter().map(Arc::as_ref))?;
        let calls = capture.prepared_call_count()?;
        let attention_layers = super::attention::LAYERS.len();
        let catalogue_bytes = super::generated::metadata_bytes(capture)?;
        let rotary_bytes = {
            let _metadata = super::storage::admit_host_cohort(first.source(), catalogue_bytes)?;
            let sites = super::generated::sites(capture)?;
            super::attention::rotary_binding_bytes(&sites)?
        };
        let cohort_metadata_bytes = if cohort_b4 {
            if super::super::cohort_native::CohortNativeLibrary::rotary_binding_bytes(
                capture, first,
            )? != rotary_bytes
            {
                return Err(Error::Unsupported(
                    "N1/B4 immutable rotary binding extents differ".into(),
                ));
            }
            super::super::cohort_native::CohortNativeLibrary::host_bytes(capture)
        } else {
            0
        };
        let private_layout = super::generated::private_layout(capture)?.to_vec();
        let mut device_layout = Vec::with_capacity(sum([
            mul(attention_layers, 4)?,
            mul(calls, 7)?,
            private_layout.len(),
            1,
        ])?);
        device_layout.extend(std::iter::repeat_n(
            super::transport::CACHE_BYTES,
            attention_layers,
        ));
        device_layout.extend(std::iter::repeat_n(512, mul(attention_layers, 2)?));
        device_layout.extend(std::iter::repeat_n(4, attention_layers));
        for &prompt in capture.captured_prefill_rows() {
            for call in 0..capture.captured_call_count() {
                let rows = if call == 0 { prompt } else { 1 };
                device_layout.extend([mul(rows, 8)?, 4, 4, 8, 8, 4, 16]);
            }
        }
        let mut private_seen = BTreeSet::new();
        for &(role, n) in &private_layout {
            if !matches!(
                role,
                Role::Lse | Role::PrivateO | Role::PartialO | Role::PartialLse
            ) || n == 0
                || n > 8 * MIB
                || !private_seen.insert((role, n))
            {
                return Err(Error::State(
                    "text actual private FA catalogue layout".into(),
                ));
            }
            device_layout.push(n);
        }
        let output = first.record().selection().output();
        let producer = first
            .record()
            .plan()
            .source_value_provenance(output)?
            .ok_or_else(|| Error::State("text actual head producer absent".into()))?;
        let head = first
            .record()
            .operator(producer.node_index())?
            .ok_or_else(|| Error::State("text actual head descriptor absent".into()))?;
        let [member] = head.matrices() else {
            return Err(Error::State("text actual tied head member arity".into()));
        };
        let head_shape = crate::resident_leases::TiedWeight::source_layout(first.source(), member)?;
        device_layout.push(if head_mode.resident() {
            cubecl::cuda::Bf16VocabularyHead::resident_manifest(head_shape)
                .map_err(|e| Error::Unsupported(format!("resident head source geometry: {e}")))?;
            head_shape.bytes()
        } else {
            MIB
        });
        // The one resident table replaces the streamed slot; it is never a
        // full retained host shadow. Original uploader keeps one source window.
        let upload_metadata = mul(usize::from(head_mode.resident()), 64 * 1024)?;
        // No sharing/dedup assumption: each of12 <=512B source windows may
        // retain two1MiB chunks; actual scans/source metadata are caller-priced.
        let source_windows = mul(mul(attention_layers, 4)?, MIB + 24)?;
        let transport_metadata = scopes
            .iter()
            .filter(|scope| scope.call().rows().base() == 0)
            .try_fold(0usize, |total, scope| {
                (0..capture.captured_call_count()).try_fold(total, |n, ordinal| {
                    n.checked_add(super::transport::Definitions::metadata_bound(
                        scope.call().rows().len(),
                        ordinal,
                    )?)
                    .ok_or_else(|| Error::Limit("text captured cold transport owner sum".into()))
                })
            })?;
        let retained_host_bytes = sum([
            upload_metadata,
            catalogue_bytes,
            2 * MIB,
            mul(attention_layers, 64 * 1024)?,
            mul(calls, 64 * 1024 + 3 * 4096)?,
            128 * 1024,
            mul(capture.cold_scope_count()?, 64 * 1024)?,
            mul(usize::from(audit_head), MIB)?,
            source_windows,
            // Second retained source window in the bounded retirement ring.
            mul(
                usize::from(!head_mode.resident()),
                super::super::text_head::STREAMED_WINDOW_STAGING_UPPER,
            )?,
        ])?;
        let retained_host_bytes = retained_host_bytes
            .checked_add(transport_metadata)
            .ok_or_else(|| Error::Limit("text retained cold source metadata".into()))?;
        // Cold-head binding budget above covers every captured scope although
        // Storage prepares one head. This peak is six attention-cache uploads
        // plus the head streaming window; current wrappers are caller-priced.
        let initialization_host_bytes = sum([
            mul(mul(attention_layers, 3)?, super::transport::CACHE_BYTES)?,
            64 * 1024,
            2 * (MIB + 24),
        ])?
        .max(catalogue_bytes)
        .max(cohort_metadata_bytes);
        Ok(Self {
            device_layout,
            retained_host_bytes,
            initialization_host_bytes,
            provider_bytes: super::text_native_provider_bytes_for_cohort(capture, cohort_b4),
            rotary_bytes,
            private_layout,
            catalogue_bytes,
        })
    }
}
