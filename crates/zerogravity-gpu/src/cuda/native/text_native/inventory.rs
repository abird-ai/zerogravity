//! SAME source/compiled-layout inventory consumed before original allocation.
use super::{super::text_model::TextModelBinding, packets::Role};
use std::{collections::BTreeSet, sync::Arc};
use zerogravity_core::{Error, Result};
const MIB: usize = 1024 * 1024;
pub(crate) struct TextNativeInventory {
    pub(crate) device_layout: Vec<usize>,
    pub(crate) retained_host_bytes: usize,
    pub(crate) initialization_host_bytes: usize,
    pub(crate) provider_bytes: usize,
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
        let mut rows = BTreeSet::new();
        capture.require_bindings(first.record())?;
        let mut prompts = BTreeSet::new();
        for scope in scopes {
            capture.require_bindings(scope.record())?;
            if scope.native_capture() != capture
                || !scope.source().same_issuer(first.source())
                || !rows.insert(scope.call().rows().len())
            {
                return Err(Error::State(
                    "text inventory source/family/real row geometries".into(),
                ));
            }
            if scope.call().rows().base() == 0 {
                prompts.insert(scope.call().rows().len());
            }
        }
        if scopes.len() != 6
            || rows != BTreeSet::from([1, 47, 57, 62, 67, 72])
            || prompts != BTreeSet::from([47, 57, 62, 67, 72])
        {
            return Err(Error::State(
                "text inventory exactly five prompts plus fresh N1".into(),
            ));
        }
        let catalogue_bytes = super::generated::metadata_bytes(capture)?;
        let private_layout = super::generated::private_layout(capture)?.to_vec();
        let mut device_layout = Vec::with_capacity(6 + 12 + 6 + 40 * 7 + private_layout.len() + 1);
        device_layout.extend([super::transport::CACHE_BYTES; 6]);
        device_layout.extend([512; 12]);
        device_layout.extend([4; 6]);
        for prompt in [47, 57, 62, 67, 72] {
            for call in 0..8 {
                let rows = if call == 0 { prompt } else { 1 };
                device_layout.extend([rows * 8, 4, 4, 8, 8, 4, 16]);
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
        let upload_metadata = usize::from(head_mode.resident()) * 64 * 1024;
        // No sharing/dedup assumption: each of12 <=512B source windows may
        // retain two1MiB chunks; actual scans/source metadata are caller-priced.
        let source_windows = 24 * (MIB + 24);
        let transport_metadata = scopes
            .iter()
            .filter(|scope| scope.call().rows().base() == 0)
            .try_fold(0usize, |total, scope| {
                (0..8).try_fold(total, |n, ordinal| {
                    n.checked_add(super::transport::Definitions::metadata_bound(
                        scope.call().rows().len(),
                        ordinal,
                    )?)
                    .ok_or_else(|| Error::Limit("text forty cold transport owner sum".into()))
                })
            })?;
        let retained_host_bytes = upload_metadata
            + catalogue_bytes
            + 2 * MIB
            + 6 * 64 * 1024
            + 40 * (64 * 1024 + 3 * 4096)
            + 128 * 1024
            + 6 * 64 * 1024
            + usize::from(audit_head) * MIB
            + source_windows
            // Second retained source window in the bounded retirement ring.
            + usize::from(!head_mode.resident())
                * super::super::text_head::STREAMED_WINDOW_STAGING_UPPER;
        let retained_host_bytes = retained_host_bytes
            .checked_add(transport_metadata)
            .ok_or_else(|| Error::Limit("text retained cold source metadata".into()))?;
        // Conservative cold-binding budget permits all six supported shapes;
        // Storage actually prepares ONE cold Head. Fresh current wrappers are caller-priced.
        // Include the Head streaming source window peak with init/transient upper.
        let initialization_host_bytes =
            6 * 3 * super::transport::CACHE_BYTES + 64 * 1024 + 2 * (MIB + 24);
        Ok(Self {
            device_layout,
            retained_host_bytes,
            initialization_host_bytes,
            provider_bytes: super::text_native_provider_bytes_for_cohort(capture, cohort_b4),
            private_layout,
            catalogue_bytes,
        })
    }
}
