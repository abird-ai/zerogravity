//! Family-typed B4 state transport: ONE prepared launch across actual rows.
//!
//! Reuses the module's checked-span/service/disjointness validation and the
//! existing prepare->bind->launch shape. The per-row equations live in the two
//! grouped `.cu` sources next to this file; the original per-row kernels stay
//! the independent bit oracle and admission fallback keys. No device pointer
//! table (direct checked per-row pointer arguments), no model-root or
//! publication ownership enters this primitive. B4 only: rows==4; no B8
//! inference (a naive eight-row KV export ABI would need 58 bindings).
use super::checked_span;
mod abi;
use crate::cuda::kernel::PinnedKernel;
use cubecl::{client::Client, server::Handle};
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::Arc;
use zerogravity_core::{Error, Result};

const STATE_SOURCE: &str = include_str!("typed/state_copy_grouped.cu");
const KV_SOURCE: &str = include_str!("typed/model_kv_copy_grouped.cu");

/// Single definitions of the exact family extents (bytes).
pub(crate) const CONV_ELEMENTS: usize = 6144 * 3;
pub(crate) const RECURRENT_ELEMENTS: usize = 16 * 128 * 128;
pub(crate) const CONV_WIDENED_BYTES: usize = CONV_ELEMENTS * 4;
pub(crate) const CONV_NATIVE_BYTES: usize = CONV_ELEMENTS * 2;
pub(crate) const RECURRENT_BYTES: usize = RECURRENT_ELEMENTS * 4;
pub(crate) const KV_CANONICAL_BYTES: usize = 512 * 512 * 4;
pub(crate) const KV_CACHE_BYTES: usize = 17 * 2 * 544 * 512 * 2;
const ROWS: usize = 4;
const GROUPED_THREADS: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TypedLayout {
    /// Canonical `[channel,frame]` widened-BF16 U32 <-> native `[frame,channel]`.
    Conv,
    /// Exact U32 `[head,K,V]` <-> `[head,V,K]` transpose; no FP math.
    RecurrentH,
    /// Paired K/V native page/stride layout with per-row base/end/block views.
    Kv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TypedDirection {
    Import,
    Export,
}

/// Validated row descriptor. Exactly the fields a (layout, direction) consumes
/// may be present; any other combination is refused before any launch.
pub(crate) struct TypedRow<'a> {
    /// Canonical prior plane: conv/H import source; KV prior K (both directions).
    pub prior: Option<&'a Handle>,
    /// Canonical candidate plane: conv/H export destination; KV candidate K.
    pub candidate: Option<&'a Handle>,
    /// Native cache view (conv/H import destination, export source; KV cache).
    pub native: &'a Handle,
    /// Paired V plane: KV import prior V / export candidate V.
    pub second: Option<&'a Handle>,
    /// KV export only: prior V for that row's unchanged future padding.
    pub second_prior: Option<&'a Handle>,
    /// Per-row base/end/block views (KV only).
    pub controls: Option<[&'a Handle; 3]>,
}

/// Checked handle clones, never a new allocation or lease capability.
pub(crate) struct PreparedTypedTransport {
    launch: crate::cuda::kernel::PreparedPinnedLaunch,
}

/// Exact field presence per family: every consumed field present, every other
/// field absent. Refused before any device command.
fn require_row_fields(
    layout: TypedLayout,
    direction: TypedDirection,
    prior: bool,
    candidate: bool,
    second: bool,
    second_prior: bool,
    controls: bool,
) -> Result<()> {
    let kv = layout == TypedLayout::Kv;
    let import = direction == TypedDirection::Import;
    let expected = [
        ("prior", prior),
        ("candidate", candidate),
        ("second", second),
        ("second_prior", second_prior),
        ("controls", controls),
    ];
    let want = |name: &str| match (kv, import, name) {
        (false, i, "prior") => i,   // GDN import reads the canonical prior plane.
        (true, _, "prior") => true, // KV always reads its prior K plane.
        (_, i, "candidate") => !i,  // Both families: export destination.
        (true, _, "second") => true,
        (true, i, "second_prior") => !i,
        (true, _, "controls") => true,
        _ => false,
    };
    for (name, present) in expected {
        if present != want(name) {
            return Err(Error::Malformed(
                "typed transport row field combination".into(),
            ));
        }
    }
    Ok(())
}

impl TypedRow<'_> {
    fn require_combination(&self, layout: TypedLayout, direction: TypedDirection) -> Result<()> {
        require_row_fields(
            layout,
            direction,
            self.prior.is_some(),
            self.candidate.is_some(),
            self.second.is_some(),
            self.second_prior.is_some(),
            self.controls.is_some(),
        )
    }
}

/// Exactly one physical KV cache and stream may back the four logical rows.
/// Compare the verified allocation identity AND its complete byte view; an
/// overlapping or shorter slice is not the same binding.
fn require_shared_kv_cache(rows: &[TypedRow<'_>]) -> Result<()> {
    let first = rows
        .first()
        .ok_or_else(|| Error::Malformed("grouped KV requires rows".into()))?
        .native;
    let identity = crate::resident_program::BufferRange::new(first)?;
    for row in &rows[1..] {
        if row.native.stream != first.stream
            || crate::resident_program::BufferRange::new(row.native)? != identity
        {
            return Err(Error::State(
                "grouped KV rows have different native cache views".into(),
            ));
        }
    }
    Ok(())
}

/// One grouped GDN state-copy definition per kind (byte-identical per-row
/// equations; row selected by blockIdx.y). Import kinds: 0 conv, 2 H.
pub(crate) fn gdn_definition(kind: usize) -> Result<PinnedKernel> {
    let (elements, input_type, output_type) = match kind {
        0 => (CONV_ELEMENTS, "unsigned int", "unsigned short"),
        1 => (CONV_ELEMENTS, "unsigned short", "unsigned int"),
        2 | 3 => (RECURRENT_ELEMENTS, "unsigned int", "unsigned int"),
        _ => return Err(Error::Unsupported("typed transport GDN kind".into())),
    };
    let abi = abi::gdn(kind)?;
    let entry = abi.entry;
    let text: Arc<str> = format!(
        "// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_ENTRY {entry}\n#define ZG_COPY {kind}\n#define ZG_ELEMENTS {elements}u\n#define ZG_INPUT {input_type}\n#define ZG_OUTPUT {output_type}\n{STATE_SOURCE}",
    )
    .into();
    PinnedKernel::new(entry, text, GROUPED_THREADS, abi::metadata(&abi))
}

/// One grouped KV transport definition per direction, built against the real
/// generated ABI: import = 22 bindings (4 prior K + 4 prior V + cache + 12
/// per-row control views + status); export = 30 (adds 4 candidate K + 4
/// candidate V). Both hold under the existing 32-binding PinnedKernel guard.
pub(crate) fn kv_definition(incoming: bool) -> Result<PinnedKernel> {
    let abi = abi::kv(incoming)?;
    let entry = abi.entry;
    let text: Arc<str> = format!(
        "// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_ENTRY {entry}\n#define ZG_IN {}\n{KV_SOURCE}",
        usize::from(incoming),
    )
    .into();
    PinnedKernel::new(entry, text, GROUPED_THREADS, abi::metadata(&abi))
}

/// Per-family extents for the kernel's per-row (source, destination) pairs.
fn gdn_extents(layout: TypedLayout, direction: TypedDirection) -> Result<(usize, usize)> {
    match (layout, direction) {
        (TypedLayout::Conv, TypedDirection::Import) => Ok((CONV_WIDENED_BYTES, CONV_NATIVE_BYTES)),
        (TypedLayout::Conv, TypedDirection::Export) => Ok((CONV_NATIVE_BYTES, CONV_WIDENED_BYTES)),
        (TypedLayout::RecurrentH, _) => Ok((RECURRENT_BYTES, RECURRENT_BYTES)),
        _ => Err(Error::Unsupported("typed transport GDN family".into())),
    }
}

impl PreparedTypedTransport {
    /// Validates every row (foreign service, exact extents, alignment,
    /// pairwise disjointness across all rows) before binding ONE grouped
    /// launch. All rows are validated before the first command; no
    /// partial-row fallback exists after this returns.
    pub(crate) fn prepare(
        client: &Client,
        kernel: &PinnedKernel,
        rows: &[TypedRow<'_>],
        status: &Handle,
        layout: TypedLayout,
        direction: TypedDirection,
        grid: [u32; 3],
    ) -> Result<Self> {
        if rows.len() != ROWS || grid[1] != ROWS as u32 || grid.contains(&0) {
            return Err(Error::Malformed(
                "typed transport actual B4 rows/grouped grid only".into(),
            ));
        }
        // One typed ABI determines both the PinnedKernel metadata and the
        // ordered checked original handles; no second role/width loop.
        for row in rows {
            row.require_combination(layout, direction)?;
        }
        let args = if layout == TypedLayout::Kv {
            require_shared_kv_cache(rows)?;
            abi::kv(direction == TypedDirection::Import)?
        } else {
            let kind = match (layout, direction) {
                (TypedLayout::Conv, TypedDirection::Import) => 0,
                (TypedLayout::Conv, TypedDirection::Export) => 1,
                (TypedLayout::RecurrentH, TypedDirection::Import) => 2,
                (TypedLayout::RecurrentH, TypedDirection::Export) => 3,
                _ => return Err(Error::Unsupported("typed transport family".into())),
            };
            abi::gdn(kind)?
        };
        // The source entry, complete argument metadata and actual bound values
        // must describe the *same* native variant, before any GPU launch.
        kernel.require_typed_abi(args.entry, args.iter().map(|arg| (arg.bytes, arg.access)))?;
        let mut handles = Vec::with_capacity(args.len());
        let mut ranges = Vec::with_capacity(args.len());
        for arg in args.iter() {
            let handle = arg.handle(rows, status)?;
            if handle.service != client.service_id() {
                return Err(Error::State(
                    "typed transport foreign physical service".into(),
                ));
            }
            checked_span(
                handle.size(),
                handle.offset_start.unwrap_or(0),
                handle.offset_end.unwrap_or(0),
                arg.bytes / 4,
            )?;
            let range = crate::resident_program::BufferRange::new(handle)?;
            if ranges
                .iter()
                .any(|prior: &crate::resident_program::BufferRange| range.overlaps(prior))
            {
                return Err(Error::State(
                    "typed transport aliased independent row planes".into(),
                ));
            }
            ranges.push(range);
            handles.push(handle.clone());
        }
        // The cold kernel's ABI must match this family's binding order; the
        // `PinnedKernel::prepare` validation re-checks service/alignment/
        // extent against the real generated ABI and the <=32-binding guard.
        let launch = kernel.prepare(client, grid, handles)?;
        Ok(Self { launch })
    }
    /// ONE original-client launch for all rows. The enclosing physical
    /// submission already retains every original lease and the status epoch.
    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
        self.launch.launch_on(client)?;
        Ok(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_runtime::kernel::{CubeKernel, KernelMetadata};

    #[test]
    fn grouped_definitions_hold_the_real_abi_under_the_32_binding_guard() -> Result<()> {
        for kind in 0..4 {
            let kernel = gdn_definition(kind)?;
            // 4 source + 4 destination + status = 9 bindings per grouped launch.
            assert_eq!(kernel.binding_count(), 9);
            assert!(kernel.binding_count() <= 32);
            assert_eq!(kernel.id().mode, cubecl::prelude::ExecutionMode::Unchecked);
            assert_eq!(kernel.id().address_type, cubecl::ir::AddressType::U64);
        }
        let incoming = kv_definition(true)?;
        let outgoing = kv_definition(false)?;
        // Real generated ABI (the design's 21 was an estimate): import with
        // per-row control views is 22; export adds 8 candidate lanes = 30.
        assert_eq!(incoming.binding_count(), 22);
        assert_eq!(outgoing.binding_count(), 30);
        // The CUDA source declares base/end/block *per row*. Guard against
        // accidentally grouping fields by width and swapping pointer types.
        let controls: Vec<usize> = (0..ROWS).flat_map(|_| [8, 4, 4]).collect();
        let mut import_sizes = vec![KV_CANONICAL_BYTES; 8];
        import_sizes.push(KV_CACHE_BYTES);
        import_sizes.extend(&controls);
        import_sizes.push(4);
        assert_eq!(incoming.binding_sizes(), import_sizes);
        let mut export_sizes = vec![KV_CANONICAL_BYTES; 8];
        export_sizes.push(KV_CACHE_BYTES);
        export_sizes.extend(vec![KV_CANONICAL_BYTES; 8]);
        export_sizes.extend(&controls);
        export_sizes.push(4);
        assert_eq!(outgoing.binding_sizes(), export_sizes);
        assert_ne!(incoming.id(), outgoing.id());
        // A naive eight-row KV export would need 2*8 prior + cache + 2*8
        // candidate + 3*8 control + status = 74 bindings; `prepare` refuses
        // any row population other than the admitted four.
        Ok(())
    }

    #[test]
    fn wrong_native_variant_refuses_even_when_byte_extents_match() -> Result<()> {
        let h_import = gdn_definition(2)?;
        let h_export = abi::gdn(3)?;
        assert!(
            h_import
                .require_typed_abi(
                    h_export.entry,
                    h_export.iter().map(|arg| (arg.bytes, arg.access))
                )
                .is_err()
        );
        let true_import = abi::gdn(2)?;
        h_import.require_typed_abi(
            true_import.entry,
            true_import.iter().map(|arg| (arg.bytes, arg.access)),
        )?;
        let kv_in = kv_definition(true)?;
        let kv_export = abi::kv(false)?;
        assert!(
            kv_in
                .require_typed_abi(
                    kv_export.entry,
                    kv_export.iter().map(|arg| (arg.bytes, arg.access))
                )
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn grouped_definitions_are_stable_and_reuse_exact_per_row_equations() -> Result<()> {
        for kind in 0..4 {
            let first = gdn_definition(kind)?;
            assert_eq!(first.id(), gdn_definition(kind)?.id());
            assert_eq!(
                first.id().stable_hash(),
                gdn_definition(kind)?.id().stable_hash()
            );
            assert_ne!(first.id(), gdn_definition((kind + 1) % 4)?.id());
        }
        let conv_import = gdn_definition(0)?.source().expect("grouped source").source;
        assert!(conv_import.contains("blockIdx.y"));
        assert!(conv_import.contains("(bits & 0xffffu) != 0u"));
        assert!(conv_import.contains("atomicOr(status, 1u)"));
        let h_in = gdn_definition(2)?.source().expect("grouped source").source;
        assert!(h_in.contains("1u << ZG_COPY"));
        let export = kv_definition(false)?
            .source()
            .expect("grouped source")
            .source;
        assert!(export.contains("k = prior_k[index]; v = prior_v[index];"));
        assert!(export.contains("atomicOr(status, 2u)"));
        let import = kv_definition(true)?
            .source()
            .expect("grouped source")
            .source;
        assert!(import.contains("block * 557056ull"));
        assert!(import.contains("atomicOr(status, 1u);"));
        Ok(())
    }

    #[test]
    fn family_extents_stay_exact_and_grids_follow_the_original_equations() {
        assert_eq!(CONV_ELEMENTS, 18_432);
        assert_eq!(RECURRENT_ELEMENTS, 262_144);
        assert_eq!(CONV_WIDENED_BYTES, 73_728);
        assert_eq!(CONV_NATIVE_BYTES, 36_864);
        assert_eq!(RECURRENT_BYTES, 1_048_576);
        assert_eq!(KV_CANONICAL_BYTES, 1_048_576);
        assert_eq!(KV_CACHE_BYTES, 18_939_904);
        // GDN grouped grid: the original per-row blocks, rows on blockIdx.y.
        assert_eq!((CONV_ELEMENTS as u32).div_ceil(128), 144);
        assert_eq!((RECURRENT_ELEMENTS as u32).div_ceil(128), 2048);
        // KV export covers the complete canonical plane; import covers the
        // live prefix of the largest admitted base (53).
        assert_eq!(512 * 512, 262_144);
        assert_eq!((53 * 512) as u32, 27_136);
    }

    #[test]
    fn grouped_kv_refuses_unshared_native_cache_even_with_equal_extents() -> Result<()> {
        use cubecl_common::device::{DeviceId, ServiceId};
        use cubecl_core::environment::stream::StreamId;
        let service = ServiceId::of::<()>(DeviceId::new(0, 0));
        let stream = StreamId::current();
        let native = Handle::new(service, stream, KV_CACHE_BYTES as u64);
        let mut handles = vec![native.clone(); ROWS];
        fn rows(handles: &[Handle]) -> Vec<TypedRow<'_>> {
            handles
                .iter()
                .map(|native| TypedRow {
                    prior: None,
                    candidate: None,
                    native,
                    second: None,
                    second_prior: None,
                    controls: None,
                })
                .collect()
        }
        require_shared_kv_cache(&rows(&handles))?;
        handles[3] = Handle::new(service, stream, KV_CACHE_BYTES as u64);
        assert!(require_shared_kv_cache(&rows(&handles)).is_err());
        handles[3] = native.clone().offset_end(4);
        assert!(require_shared_kv_cache(&rows(&handles)).is_err());
        let mut different_stream = native.clone();
        different_stream.stream = StreamId::allocate();
        handles[3] = different_stream;
        assert!(require_shared_kv_cache(&rows(&handles)).is_err());
        Ok(())
    }

    #[test]
    fn row_field_combinations_refuse_every_unexpected_population() -> Result<()> {
        let gdn = |direction: TypedDirection, candidate: bool, prior: bool| {
            require_row_fields(
                TypedLayout::Conv,
                direction,
                prior,
                candidate,
                false,
                false,
                false,
            )
        };
        gdn(TypedDirection::Import, false, true).unwrap();
        assert!(gdn(TypedDirection::Import, true, true).is_err());
        assert!(gdn(TypedDirection::Import, false, false).is_err());
        gdn(TypedDirection::Export, true, false).unwrap();
        assert!(gdn(TypedDirection::Export, false, false).is_err());
        assert!(gdn(TypedDirection::Export, true, true).is_err());
        let h = |direction: TypedDirection, second: bool| {
            require_row_fields(
                TypedLayout::RecurrentH,
                direction,
                direction == TypedDirection::Import,
                direction == TypedDirection::Export,
                second,
                false,
                false,
            )
        };
        h(TypedDirection::Import, false).unwrap();
        assert!(h(TypedDirection::Import, true).is_err());
        h(TypedDirection::Export, false).unwrap();
        let kv = |direction: TypedDirection, second_prior: bool| {
            require_row_fields(
                TypedLayout::Kv,
                direction,
                true,
                direction == TypedDirection::Export,
                true,
                second_prior,
                true,
            )
        };
        kv(TypedDirection::Import, false).unwrap();
        assert!(kv(TypedDirection::Import, true).is_err());
        kv(TypedDirection::Export, true).unwrap();
        assert!(kv(TypedDirection::Export, false).is_err());
        // Controls never belong to a GDN family row.
        assert!(
            require_row_fields(
                TypedLayout::Conv,
                TypedDirection::Import,
                true,
                false,
                false,
                false,
                true
            )
            .is_err()
        );
        Ok(())
    }
}
