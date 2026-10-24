//! SAME source/compiled-layout inventory consumed before original allocation.
use super::{super::short_hf_model::ShortHfModelBinding, packets::Role};
use std::{collections::BTreeSet, sync::Arc};
use zerogravity_core::{Error, Result};
const MIB: usize = 1024 * 1024;
pub(crate) struct ShortHfNativeInventory {
    pub(crate) device_layout: Vec<usize>,
    pub(crate) retained_host_bytes: usize,
    pub(crate) initialization_host_bytes: usize,
    pub(crate) provider_bytes: usize,
    pub(super) private_layout: Vec<(Role, usize)>,
    pub(super) catalogue_bytes: usize,
}
impl ShortHfNativeInventory {
    pub(crate) fn for_scopes(
        scopes: &[Arc<ShortHfModelBinding>],
        audit_head: bool,
        cohort_b4: bool,
        head_mode: super::super::short_hf_head::HeadMode,
    ) -> Result<Self> {
        let first = scopes.first().ok_or_else(|| {
            Error::State("short-HF inventory missing genuine source scopes".into())
        })?;
        let profile = first.call().model_profile();
        let mut rows = BTreeSet::new();
        let mut prompts = BTreeSet::new();
        for scope in scopes {
            if scope.call().model_profile() != profile
                || !Arc::ptr_eq(scope.source().source(), first.source().source())
                || !rows.insert(scope.call().span().len())
            {
                return Err(Error::State(
                    "short-HF inventory source/family/real row geometries".into(),
                ));
            }
            if scope.call().span().base() == 0 {
                prompts.insert(scope.call().span().workload().prompt_rows());
            }
        }
        if scopes.len() != 6
            || rows != BTreeSet::from([1, 47, 57, 62, 67, 72])
            || prompts != BTreeSet::from([47, 57, 62, 67, 72])
        {
            return Err(Error::State(
                "short-HF inventory exactly five prompts plus fresh N1".into(),
            ));
        }
        let catalogue_bytes = super::generated::metadata_bytes(profile)?;
        let private_layout = super::generated::private_layout(profile)?.to_vec();
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
                    "short-HF actual private FA catalogue layout".into(),
                ));
            }
            device_layout.push(n);
        }
        let head_shape =
            crate::resident_leases::TiedWeight::source_layout(first.source().source())?;
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
            .filter(|scope| scope.call().span().base() == 0)
            .try_fold(0usize, |total, scope| {
                (0..8).try_fold(total, |n, ordinal| {
                    n.checked_add(super::transport::Definitions::metadata_bound(
                        scope.call().span().workload(),
                        ordinal,
                    )?)
                    .ok_or_else(|| Error::Limit("short-HF forty cold transport owner sum".into()))
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
                * super::super::short_hf_head::STREAMED_WINDOW_STAGING_UPPER;
        let retained_host_bytes = retained_host_bytes
            .checked_add(transport_metadata)
            .ok_or_else(|| Error::Limit("short-HF retained cold source metadata".into()))?;
        // Conservative cold-binding budget permits all six supported shapes;
        // Storage actually prepares ONE cold Head. Fresh current wrappers are caller-priced.
        // Include the Head streaming source window peak with init/transient upper.
        let initialization_host_bytes =
            6 * 3 * super::transport::CACHE_BYTES + 64 * 1024 + 2 * (MIB + 24);
        Ok(Self {
            device_layout,
            retained_host_bytes,
            initialization_host_bytes,
            provider_bytes: super::short_hf_native_provider_bytes_for_cohort(profile, cohort_b4),
            private_layout,
            catalogue_bytes,
        })
    }
}
