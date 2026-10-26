//! Actual original layer3 BF16 members; observed W bytes are never inputs here.
use super::super::projection_cubin::not_cancelled;
use super::manifests::AttentionRole;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::source_weights::{
    ParameterWindow, TensorWindow, VerifiedTensorSource, WindowWorkspace,
};
use zerogravity_core::{
    Error, Result, accelerator::program::MatrixAxes, device_events::BufferRole, quant::QuantKind,
};

const WINDOW_BYTES: usize = 1024 * 1024;
const PACKED_BYTES: usize = 10_485_760 + 4_194_304 + 512 + 512;
struct WeightPayload {
    _source: Arc<VerifiedTensorSource>,
    _matrices: Vec<Arc<TensorWindow>>,
    _parameters: Vec<Arc<ParameterWindow>>,
    values: BTreeMap<AttentionRole, Vec<u8>>,
    // Original packed/upload-copy charge outlives every actual payload Vec.
    _host: WindowWorkspace,
}
pub(super) struct AttentionWeights {
    leases: BTreeMap<AttentionRole, ResidentLease>,
    payload: Arc<WeightPayload>,
}
impl AttentionWeights {
    /// All other managed leaf/cache/value buffers are allocated first. This
    /// factory reserves actual original source windows and both payload/copy
    /// capacities before Vec allocation, then uploads on the original init.
    pub(super) fn prepare(
        source: Arc<VerifiedTensorSource>,
        service: &Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        service.ensure_usable()?;
        init.require_service(service)?;
        if source.source_identity() != service.source_identity() {
            return Err(Error::State("attention original W source/service".into()));
        }
        let host = source.reserve_operator_buffers(PACKED_BYTES, PACKED_BYTES)?;
        let mut windows = Vec::new();
        let mut parameters = Vec::new();
        let mut values = BTreeMap::new();
        for (role, members, expected) in [
            (
                AttentionRole::QkvW,
                &[
                    ("q_proj.weight", 4096usize, 1024usize),
                    ("k_proj.weight", 512, 1024),
                    ("v_proj.weight", 512, 1024),
                ][..],
                10_485_760usize,
            ),
            (
                AttentionRole::OutW,
                &[("o_proj.weight", 1024usize, 2048usize)][..],
                4_194_304usize,
            ),
        ] {
            let mut packed = Vec::with_capacity(expected);
            for &(suffix, rows, columns) in members {
                not_cancelled(cancelled)?;
                let name = format!("model.language_model.layers.3.self_attn.{suffix}");
                let matrix = source.matrix(&name, MatrixAxes::OutputInput)?;
                if matrix.tensor().kind != QuantKind::Bf16
                    || matrix.rows() != rows
                    || matrix.columns() != columns
                {
                    return Err(Error::Unsupported(
                        "attention closed original BF16 W member geometry".into(),
                    ));
                }
                let step = WINDOW_BYTES / (columns * 2);
                for first in (0..rows).step_by(step) {
                    not_cancelled(cancelled)?;
                    let end = (first + step).min(rows);
                    let window = Arc::new(match cancelled {
                        Some(flag) => source.read_rows_cancellable(&matrix, first..end, flag)?,
                        None => source.read_rows(&matrix, first..end)?,
                    });
                    if window.as_slice().len() != (end - first) * columns * 2
                        || !super::super::finite_bf16(window.as_slice())
                    {
                        return Err(Error::Malformed(
                            "attention finite complete original W window".into(),
                        ));
                    }
                    packed.extend_from_slice(window.as_slice());
                    windows.push(window);
                }
            }
            if packed.len() != expected || packed.capacity() != expected {
                return Err(Error::State("attention exact admitted W packing".into()));
            }
            values.insert(role, packed);
        }
        for (role, suffix) in [
            (AttentionRole::QGamma, "q_norm.weight"),
            (AttentionRole::KGamma, "k_norm.weight"),
        ] {
            not_cancelled(cancelled)?;
            let name = format!("model.language_model.layers.3.self_attn.{suffix}");
            let parameter = source.parameter(&name)?;
            if parameter.tensor().kind != QuantKind::Bf16
                || parameter.tensor().shape != [256]
                || parameter.elements() != 256
            {
                return Err(Error::Unsupported(
                    "attention actual original BF16 QK gamma256".into(),
                ));
            }
            let window = Arc::new(source.read_parameter(&parameter, 0..256)?);
            if !window.is_complete()
                || window.as_slice().len() != 512
                || !super::super::finite_bf16(window.as_slice())
            {
                return Err(Error::Malformed(
                    "attention complete finite source QK gamma".into(),
                ));
            }
            values.insert(role, window.as_slice().to_vec());
            parameters.push(window);
        }
        let payload = Arc::new(WeightPayload {
            _source: source,
            _matrices: windows,
            _parameters: parameters,
            values,
            _host: host,
        });
        init.retain_completion_owner(payload.clone())?;
        let mut leases = BTreeMap::new();
        for (&role, bytes) in &payload.values {
            not_cancelled(cancelled)?;
            leases.insert(
                role,
                init.upload_bytes(bytes, BufferRole::ImmutableWeights)?,
            );
        }
        let result = Arc::new(Self { leases, payload });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(super) fn host_bytes(&self) -> usize {
        self.payload._host.bytes()
    }
    pub(super) fn leases(&self) -> &BTreeMap<AttentionRole, ResidentLease> {
        &self.leases
    }
    /// Final real source readback/unload guard uses these original payloads.
    pub(super) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.leases
            .iter()
            .map(|(&role, lease)| (lease.clone(), self.payload.values[&role].as_slice()))
            .collect()
    }
}
