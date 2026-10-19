//! Owned verified original-index windows; deliberately not a TensorSource.
//! No model permit, borrowed cache slice, GPU readiness or StateRoot publisher.
use super::SourceWeights;
use crate::artifact::{TensorInfo, parse_safetensors};
use crate::cache::TransferAdmission;
use crate::sharded::SafetensorsIndex;
use crate::ssd_extent::{ExtentSource, StagedSlice, source_not_cancelled};
use crate::verified_shards::{ExpectedSourceShard, source_bundle_sha256};
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{MatrixAxes, SourceParameter},
};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};

const METADATA_BASE: u64 = 8 * 1024 * 1024;
const MAX_METADATA: u64 = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct WindowOpenLimits {
    /// Source file bytes, not a promise to materialize them in RAM.
    pub max_source_bytes: u64,
    /// Conservative owned parser/index/extent metadata envelope, retained.
    pub max_metadata_bytes: u64,
    pub max_shards: usize,
    pub max_tensors: usize,
    pub scan_chunk_bytes: usize,
}
impl Default for WindowOpenLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 4 * 1024 * 1024 * 1024,
            max_metadata_bytes: MAX_METADATA,
            max_shards: 64,
            max_tensors: 200_000,
            scan_chunk_bytes: 4 * 1024 * 1024,
        }
    }
}

#[must_use]
pub struct VerifiedWindowPlan {
    index: SafetensorsIndex,
    index_sha256: String,
    bundle_sha256: String,
    shards: Vec<ExpectedSourceShard>,
    limits: WindowOpenLimits,
    metadata_baseline: u64,
    source_bytes: u64,
    admission: TransferAdmission,
    // Last field: owned index/descriptors drop before their metadata charge.
    metadata: GlobalResourceLease,
}

struct WindowShard {
    basename: String,
    sha256: String,
    source: ExtentSource,
}
struct TensorLocation {
    info: TensorInfo,
    shard: usize,
}
struct SourceOwner {
    identity: SourceIdentity,
    index_sha256: String,
    bundle_sha256: String,
    source_bytes: u64,
    tensors: BTreeMap<String, Arc<TensorLocation>>,
    shards: Vec<WindowShard>,
    admission: TransferAdmission,
    metadata_bytes: u64,
    _metadata: GlobalResourceLease,
}

/// Sealed original source owner. Backing is metadata plus opened descriptors;
/// complete borrowed tensor bytes are intentionally unavailable.
#[derive(Clone)]
pub struct VerifiedTensorSource(Arc<SourceOwner>);

/// Exact matrix geometry from authenticated original Safetensors headers.
/// A descriptor cannot be rebound to another source, name, codec or axes.
#[derive(Clone)]
pub struct VerifiedMatrix {
    location: Arc<TensorLocation>,
    owner: Arc<SourceOwner>,
    axes: MatrixAxes,
    columns: usize,
    rows: usize,
    row_bytes: usize,
}

/// Owned immutable bytes plus the original source and real host staging lease.
pub struct TensorWindow {
    staged: StagedSlice,
    rows: Range<usize>,
    matrix: VerifiedMatrix,
}

/// Sealed full original rank1 parameter; no matrix axes or adjusted values.
#[derive(Clone)]
pub struct VerifiedParameter {
    location: Arc<TensorLocation>,
    owner: Arc<SourceOwner>,
    elements: usize,
    element_bytes: usize,
    bytes: usize,
}

/// Owned contiguous raw element range, retaining its full original descriptor.
/// A partial range is never presented as a complete original tensor.
pub struct ParameterWindow {
    // Staged payload/offsets and their charge retire before source metadata.
    staged: StagedSlice,
    elements: Range<usize>,
    parameter: VerifiedParameter,
}

impl SourceWeights {
    /// Authenticate the original index and reserve bounded host metadata.
    /// This alternative leaves complete owned-source and 4GiB APIs unchanged.
    pub fn prepare_verified_windows(
        index_json: &[u8],
        expected_index_sha256: &str,
        shards: Vec<ExpectedSourceShard>,
        limits: WindowOpenLimits,
        admission: &TransferAdmission,
    ) -> Result<VerifiedWindowPlan> {
        if admission.resource.backend != "host"
            || [
                &admission.resource.backend,
                &admission.resource.device,
                &admission.owner.model,
                &admission.owner.session,
            ]
            .iter()
            .any(|label| label.is_empty() || label.len() > 256)
        {
            return Err(Error::State(
                "tensor windows require an existing host resource domain/owner".into(),
            ));
        }
        if limits.max_source_bytes == 0
            || limits.max_metadata_bytes == 0
            || limits.max_metadata_bytes > MAX_METADATA
            || limits.max_shards == 0
            || limits.max_shards > 64
            || limits.max_tensors == 0
            || limits.max_tensors > 200_000
            || shards.is_empty()
            || shards.len() > limits.max_shards
            || index_json.len() > 8 * 1024 * 1024
        {
            return Err(Error::Limit("verified window source limits".into()));
        }
        crate::cache::paths::validate_digest(expected_index_sha256)?;
        if format!("{:x}", Sha256::digest(index_json)) != expected_index_sha256 {
            return Err(Error::Malformed(
                "original window index SHA256 mismatch".into(),
            ));
        }
        let mut source_bytes = 0u64;
        let mut carried = 0u64;
        let mut chunk_count = 0u64;
        for shard in &shards {
            let (_, bytes, _) = shard.binding();
            source_bytes = source_bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::Limit("window source byte sum overflow".into()))?;
            chunk_count = chunk_count
                .checked_add(ExtentSource::scan_chunk_count(bytes, limits.scan_chunk_bytes)? as u64)
                .ok_or_else(|| Error::Limit("window source chunk sum overflow".into()))?;
            carried = carried
                .checked_add(shard.carried_metadata_bytes()?)
                .ok_or_else(|| Error::Limit("window carried metadata overflow".into()))?;
        }
        let shard_vector_bytes = (shards.capacity() as u64)
            .checked_mul(std::mem::size_of::<ExpectedSourceShard>() as u64)
            .ok_or_else(|| Error::Limit("owned shard vector capacity overflow".into()))?;
        // The retained admission clone and its ledger owner/key labels are
        // bounded metadata; borrowed admission inputs remain caller-owned.
        let admission_labels = [
            &admission.resource.backend,
            &admission.resource.device,
            &admission.owner.model,
            &admission.owner.session,
        ]
        .into_iter()
        .map(|label| label.len() as u64)
        .sum::<u64>()
            * 4;
        let metadata_baseline = METADATA_BASE
            .checked_add(index_json.len() as u64 * 16)
            .and_then(|n| n.checked_add(chunk_count * 512))
            .and_then(|n| n.checked_add(carried))
            .and_then(|n| n.checked_add(shard_vector_bytes))
            .and_then(|n| n.checked_add(admission_labels))
            .ok_or_else(|| Error::Limit("window metadata envelope overflow".into()))?;
        if source_bytes > limits.max_source_bytes || metadata_baseline > limits.max_metadata_bytes {
            return Err(Error::Limit(
                "source or retained metadata exceeds window budget".into(),
            ));
        }
        // The caller registers/attests this host pool. No inferred free-RAM claim.
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: limits.max_metadata_bytes,
            }],
        )?;
        let index = SafetensorsIndex::from_json(index_json)?;
        if index.tensors().len() > limits.max_tensors {
            return Err(Error::Limit("window source tensor count".into()));
        }
        let mut basenames = BTreeSet::new();
        for shard in &shards {
            let (basename, _, _) = shard.binding();
            if !index.shards().contains(basename) || !basenames.insert(basename) {
                return Err(Error::Malformed(
                    "extra or duplicate original window shard".into(),
                ));
            }
        }
        if basenames != index.shards() {
            return Err(Error::Malformed(
                "incomplete original window shard set".into(),
            ));
        }
        let bundle_sha256 = source_bundle_sha256(expected_index_sha256, &shards)?;
        Ok(VerifiedWindowPlan {
            index,
            index_sha256: expected_index_sha256.into(),
            bundle_sha256,
            shards,
            limits,
            metadata_baseline,
            source_bytes,
            admission: admission.clone(),
            metadata,
        })
    }
}

impl VerifiedWindowPlan {
    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
    pub fn metadata_reserved_bytes(&self) -> u64 {
        self.limits.max_metadata_bytes
    }
    pub fn open(self) -> Result<VerifiedTensorSource> {
        self.open_inner(None)
    }
    pub fn open_cancellable(self, cancelled: &AtomicBool) -> Result<VerifiedTensorSource> {
        self.open_inner(Some(cancelled))
    }
    fn open_inner(self, cancelled: Option<&AtomicBool>) -> Result<VerifiedTensorSource> {
        source_not_cancelled(cancelled)?;
        let mut tensors = BTreeMap::new();
        let mut shards = Vec::with_capacity(self.shards.len());
        let mut payload = 0u64;
        let mut metadata_used = self.metadata_baseline;
        for expected in self.shards {
            let (basename, opened, bytes, sha256) = expected.into_parts();
            let (untrusted, file) = opened.into_open_parts();
            drop(untrusted);
            let source = ExtentSource::scan_verified(
                file,
                bytes,
                &sha256,
                self.limits.scan_chunk_bytes,
                &self.admission,
                cancelled,
            )?;
            source_not_cancelled(cancelled)?;
            // Guard the common parser's allocation before it sees the header.
            let prefix = source.read_window(0, 8, &self.admission)?;
            let header_bytes =
                u64::from_le_bytes(prefix.as_slice().try_into().expect("eight header bytes"));
            metadata_used = header_bytes
                .checked_mul(16)
                .and_then(|n| metadata_used.checked_add(n))
                .ok_or_else(|| Error::Limit("verified window header envelope overflow".into()))?;
            if metadata_used > self.limits.max_metadata_bytes {
                return Err(Error::Limit(
                    "verified headers exceed retained metadata envelope".into(),
                ));
            }
            drop(prefix);
            let mut reader = VerifiedReader {
                source: &source,
                admission: &self.admission,
                position: 0,
                chunk_bytes: self.limits.scan_chunk_bytes,
                cancelled,
            };
            // Exact geometry comes ONLY from the shared parser reading chunks
            // authenticated by the successful whole-file scan.
            let info = parse_safetensors(&mut reader, bytes)?;
            let shard = shards.len();
            for info in info.tensors {
                if self.index.tensors().get(&info.name).map(String::as_str)
                    != Some(basename.as_str())
                    || tensors.contains_key(&info.name)
                    || tensors.len() >= self.limits.max_tensors
                {
                    return Err(Error::Malformed(
                        "verified original window index/header mismatch".into(),
                    ));
                }
                payload = payload
                    .checked_add(info.byte_len)
                    .ok_or_else(|| Error::Limit("verified window payload sum overflow".into()))?;
                tensors.insert(info.name.clone(), Arc::new(TensorLocation { info, shard }));
            }
            shards.push(WindowShard {
                basename,
                sha256,
                source,
            });
        }
        if tensors.len() != self.index.tensors().len()
            || payload != self.index.declared_total_size()
        {
            return Err(Error::Malformed(
                "verified window tensors do not close original index".into(),
            ));
        }
        source_not_cancelled(cancelled)?;
        let identity = SourceIdentity::for_new_allocation()?;
        Ok(VerifiedTensorSource(Arc::new(SourceOwner {
            identity,
            index_sha256: self.index_sha256,
            bundle_sha256: self.bundle_sha256,
            source_bytes: self.source_bytes,
            tensors,
            shards,
            admission: self.admission,
            metadata_bytes: self.limits.max_metadata_bytes,
            _metadata: self.metadata,
        })))
    }
}

/// Finite same-source host workspace charge; no execution or free-RAM permit.
/// Payload owners must retain this until their actual allocations are dropped.
pub struct WindowWorkspace {
    _lease: GlobalResourceLease,
    source: VerifiedTensorSource,
    bytes: usize,
}
impl WindowWorkspace {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
}

impl VerifiedTensorSource {
    pub fn source_identity(&self) -> SourceIdentity {
        self.0.identity
    }
    /// Immutable authenticated headers only; never a partial TensorSource.
    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        self.0
            .tensors
            .get(name)
            .map(|location| &location.info)
            .ok_or_else(|| Error::Malformed("verified window tensor header missing".into()))
    }
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.0.tensors.keys().map(String::as_str)
    }
    pub fn same_issuer(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Reserve before allocating private operator buffers. Uses the original
    /// admitted host domain/owner; no alternate pool, GPU event or model permit.
    pub fn reserve_workspace(&self, bytes: usize) -> Result<WindowWorkspace> {
        if bytes == 0 || bytes > 32 * 1024 * 1024 {
            return Err(Error::Limit("verified source workspace bound".into()));
        }
        self.admit_workspace(
            bytes,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: bytes as u64,
            }],
        )
    }
    /// Conservative private output + scratch envelope, under this same owner.
    pub fn reserve_operator_buffers(
        &self,
        output_bytes: usize,
        scratch_bytes: usize,
    ) -> Result<WindowWorkspace> {
        let bytes = output_bytes
            .checked_add(scratch_bytes)
            .filter(|n| *n > 0 && *n <= 32 * 1024 * 1024)
            .ok_or_else(|| Error::Limit("verified source operator buffer bound".into()))?;
        if output_bytes == 0 || scratch_bytes == 0 {
            return Err(Error::Limit(
                "operator output/scratch must both be admitted".into(),
            ));
        }
        self.admit_workspace(
            bytes,
            &[
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: output_bytes as u64,
                },
                AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: scratch_bytes as u64,
                },
            ],
        )
    }
    fn admit_workspace(
        &self,
        bytes: usize,
        requests: &[AllocationRequest],
    ) -> Result<WindowWorkspace> {
        let lease = self.0.admission.arbiter.admit(
            &self.0.admission.resource,
            self.0.admission.owner.clone(),
            requests,
        )?;
        Ok(WindowWorkspace {
            _lease: lease,
            source: self.clone(),
            bytes,
        })
    }
    pub fn source_bytes(&self) -> u64 {
        self.0.source_bytes
    }
    pub fn verified_shard_index_sha256(&self) -> &str {
        &self.0.index_sha256
    }
    pub fn verified_bundle_sha256(&self) -> &str {
        &self.0.bundle_sha256
    }
    pub fn metadata_reserved_bytes(&self) -> u64 {
        self.0.metadata_bytes
    }
    pub fn parameter(&self, name: &str) -> Result<VerifiedParameter> {
        let location =
            self.0.tensors.get(name).ok_or_else(|| {
                Error::Malformed("original parameter window tensor missing".into())
            })?;
        let [elements] = location.info.shape.as_slice() else {
            return Err(Error::Malformed(
                "original parameter window requires raw rank1".into(),
            ));
        };
        // No block codec interpretation or numerical execution eligibility.
        if !matches!(
            location.info.kind,
            QuantKind::F32 | QuantKind::F16 | QuantKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "original parameter window raw dtype".into(),
            ));
        }
        let elements = usize::try_from(*elements)
            .map_err(|_| Error::Limit("parameter window element count overflow".into()))?;
        let element_bytes = usize::try_from(location.info.kind.byte_len(1)?)
            .map_err(|_| Error::Limit("parameter window element bytes overflow".into()))?;
        let bytes = usize::try_from(location.info.byte_len)
            .map_err(|_| Error::Limit("parameter window source length overflow".into()))?;
        if elements == 0 || elements.checked_mul(element_bytes) != Some(bytes) {
            return Err(Error::Malformed(
                "original parameter window geometry".into(),
            ));
        }
        Ok(VerifiedParameter {
            location: location.clone(),
            owner: self.0.clone(),
            elements,
            element_bytes,
            bytes,
        })
    }
    /// Original Safetensors rank1 is contiguous; arbitrary strides have no API.
    pub fn read_parameter(
        &self,
        parameter: &VerifiedParameter,
        elements: Range<usize>,
    ) -> Result<ParameterWindow> {
        self.read_parameter_inner(parameter, elements, None)
    }
    pub fn read_parameter_cancellable(
        &self,
        parameter: &VerifiedParameter,
        elements: Range<usize>,
        cancelled: &AtomicBool,
    ) -> Result<ParameterWindow> {
        self.read_parameter_inner(parameter, elements, Some(cancelled))
    }
    fn read_parameter_inner(
        &self,
        parameter: &VerifiedParameter,
        elements: Range<usize>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ParameterWindow> {
        if !Arc::ptr_eq(&self.0, &parameter.owner)
            || parameter.source_identity() != self.source_identity()
            || !self
                .0
                .tensors
                .get(&parameter.location.info.name)
                .is_some_and(|location| Arc::ptr_eq(location, &parameter.location))
        {
            return Err(Error::State(
                "foreign original parameter window owner".into(),
            ));
        }
        if elements.start >= elements.end || elements.end > parameter.elements {
            return Err(Error::Malformed(
                "original parameter window element range".into(),
            ));
        }
        source_not_cancelled(cancelled)?;
        let location = &parameter.location;
        let start = (elements.start as u64)
            .checked_mul(parameter.element_bytes as u64)
            .and_then(|n| location.info.byte_offset.checked_add(n))
            .ok_or_else(|| Error::Limit("parameter window source offset overflow".into()))?;
        let bytes = (elements.end - elements.start)
            .checked_mul(parameter.element_bytes)
            .ok_or_else(|| Error::Limit("parameter window byte length overflow".into()))?;
        let staged =
            self.0.shards[location.shard]
                .source
                .read_window(start, bytes, &self.0.admission)?;
        // Blocking authenticated IO has retired before cancelled staging drops.
        source_not_cancelled(cancelled)?;
        Ok(ParameterWindow {
            staged,
            elements,
            parameter: parameter.clone(),
        })
    }
    pub fn matrix(&self, name: &str, axes: MatrixAxes) -> Result<VerifiedMatrix> {
        // This first factory accepts Safetensors only: its contiguous output
        // rows are the original [output,input] axes; never reinterpret GGUF.
        if axes != MatrixAxes::OutputInput {
            return Err(Error::Unsupported(
                "original Safetensors window requires OutputInput axes".into(),
            ));
        }
        let location = self
            .0
            .tensors
            .get(name)
            .ok_or_else(|| Error::Malformed("original window tensor missing".into()))?;
        let [rows, columns] = location.info.shape.as_slice() else {
            return Err(Error::Malformed(
                "original window tensor is not a matrix".into(),
            ));
        };
        let rows = usize::try_from(*rows)
            .map_err(|_| Error::Limit("window matrix rows overflow".into()))?;
        let columns = usize::try_from(*columns)
            .map_err(|_| Error::Limit("window matrix columns overflow".into()))?;
        let row_bytes = usize::try_from(location.info.kind.byte_len(columns as u64)?)
            .map_err(|_| Error::Limit("window matrix row bytes overflow".into()))?;
        if rows == 0
            || columns == 0
            || (row_bytes as u64).checked_mul(rows as u64) != Some(location.info.byte_len)
        {
            return Err(Error::Malformed(
                "verified window matrix codec/geometry mismatch".into(),
            ));
        }
        Ok(VerifiedMatrix {
            owner: self.0.clone(),
            location: location.clone(),
            axes,
            columns,
            rows,
            row_bytes,
        })
    }
    pub fn read_rows(&self, matrix: &VerifiedMatrix, rows: Range<usize>) -> Result<TensorWindow> {
        self.read_rows_inner(matrix, rows, None)
    }
    pub fn read_rows_cancellable(
        &self,
        matrix: &VerifiedMatrix,
        rows: Range<usize>,
        cancelled: &AtomicBool,
    ) -> Result<TensorWindow> {
        self.read_rows_inner(matrix, rows, Some(cancelled))
    }
    fn read_rows_inner(
        &self,
        matrix: &VerifiedMatrix,
        rows: Range<usize>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<TensorWindow> {
        if !Arc::ptr_eq(&self.0, &matrix.owner)
            || matrix.source_identity() != self.source_identity()
        {
            return Err(Error::State("foreign original tensor window owner".into()));
        }
        if rows.start >= rows.end || rows.end > matrix.rows {
            return Err(Error::Malformed("original tensor window row range".into()));
        }
        source_not_cancelled(cancelled)?;
        let location = &matrix.location;
        let start = (rows.start as u64)
            .checked_mul(matrix.row_bytes as u64)
            .and_then(|n| location.info.byte_offset.checked_add(n))
            .ok_or_else(|| Error::Limit("original tensor window byte offset overflow".into()))?;
        let bytes = (rows.end - rows.start)
            .checked_mul(matrix.row_bytes)
            .ok_or_else(|| Error::Limit("original tensor window length overflow".into()))?;
        let staged =
            self.0.shards[location.shard]
                .source
                .read_window(start, bytes, &self.0.admission)?;
        // Blocking IO has retired before cancellation may release its staging.
        source_not_cancelled(cancelled)?;
        Ok(TensorWindow {
            matrix: matrix.clone(),
            rows,
            staged,
        })
    }
}
impl VerifiedParameter {
    pub fn source_identity(&self) -> SourceIdentity {
        self.owner.identity
    }
    pub fn tensor(&self) -> &TensorInfo {
        &self.location.info
    }
    pub fn elements(&self) -> usize {
        self.elements
    }
    /// Full original metadata; this caller-owned copy grants no byte access.
    pub fn source_parameter(&self) -> SourceParameter {
        SourceParameter {
            source: self.source_identity(),
            name: self.location.info.name.clone(),
            kind: self.location.info.kind,
            shape: self.location.info.shape.clone(),
            bytes: self.bytes,
        }
    }
    /// Bind typed metadata to this same sealed original issuer/name/geometry.
    pub fn validate_source_parameter(&self, descriptor: &SourceParameter) -> Result<()> {
        let actual = &self.location.info;
        if descriptor.source != self.source_identity()
            || descriptor.name != actual.name
            || descriptor.kind != actual.kind
            || descriptor.shape != actual.shape
            || descriptor.bytes != self.bytes
        {
            return Err(Error::State(
                "typed parameter differs from original sealed source".into(),
            ));
        }
        Ok(())
    }
}
impl ParameterWindow {
    pub fn parameter(&self) -> &VerifiedParameter {
        &self.parameter
    }
    pub fn element_range(&self) -> Range<usize> {
        self.elements.clone()
    }
    pub fn is_complete(&self) -> bool {
        self.elements == (0..self.parameter.elements)
    }
    pub fn as_slice(&self) -> &[u8] {
        self.staged.as_slice()
    }
    pub fn source_offset(&self) -> u64 {
        self.staged.source_offset()
    }
    pub fn charged_staging_bytes(&self) -> usize {
        self.staged.charged_bytes()
    }
    pub fn verified_bundle_sha256(&self) -> &str {
        &self.parameter.owner.bundle_sha256
    }
    pub fn shard_basename(&self) -> &str {
        &self.shard().basename
    }
    pub fn shard_sha256(&self) -> &str {
        &self.shard().sha256
    }
    pub fn extent_index_sha256(&self) -> &str {
        self.shard().source.index_sha256()
    }
    fn shard(&self) -> &WindowShard {
        &self.parameter.owner.shards[self.parameter.location.shard]
    }
}
impl VerifiedMatrix {
    pub fn source_identity(&self) -> SourceIdentity {
        self.owner.identity
    }
    pub fn tensor(&self) -> &TensorInfo {
        &self.location.info
    }
    pub fn axes(&self) -> MatrixAxes {
        self.axes
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
}
impl TensorWindow {
    pub fn matrix(&self) -> &VerifiedMatrix {
        &self.matrix
    }
    pub fn row_range(&self) -> Range<usize> {
        self.rows.clone()
    }
    pub fn as_slice(&self) -> &[u8] {
        self.staged.as_slice()
    }
    pub fn source_offset(&self) -> u64 {
        self.staged.source_offset()
    }
    pub fn charged_staging_bytes(&self) -> usize {
        self.staged.charged_bytes()
    }
    pub fn verified_bundle_sha256(&self) -> &str {
        &self.matrix.owner.bundle_sha256
    }
    pub fn shard_basename(&self) -> &str {
        &self.shard().basename
    }
    pub fn shard_sha256(&self) -> &str {
        &self.shard().sha256
    }
    pub fn extent_index_sha256(&self) -> &str {
        self.shard().source.index_sha256()
    }
    fn shard(&self) -> &WindowShard {
        &self.matrix.owner.shards[self.matrix.location.shard]
    }
}

struct VerifiedReader<'a> {
    source: &'a ExtentSource,
    admission: &'a TransferAdmission,
    position: u64,
    chunk_bytes: usize,
    cancelled: Option<&'a AtomicBool>,
}
impl Read for VerifiedReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        source_not_cancelled(self.cancelled).map_err(std::io::Error::other)?;
        let bytes = output
            .len()
            .min(self.chunk_bytes * 16)
            .min(16 * 1024 * 1024)
            .min((self.source.file_bytes() - self.position).min(usize::MAX as u64) as usize);
        if bytes == 0 {
            return Ok(0);
        }
        let staged = self
            .source
            .read_window(self.position, bytes, self.admission)
            .map_err(std::io::Error::other)?;
        output[..bytes].copy_from_slice(staged.as_slice());
        self.position += bytes as u64;
        Ok(bytes)
    }
}
impl Seek for VerifiedReader<'_> {
    fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
        let position = match from {
            SeekFrom::Start(n) => Some(n),
            SeekFrom::Current(n) => self.position.checked_add_signed(n),
            SeekFrom::End(n) => self.source.file_bytes().checked_add_signed(n),
        };
        self.position = position
            .filter(|n| *n <= self.source.file_bytes())
            .ok_or_else(|| std::io::Error::other("verified source seek out of range"))?;
        Ok(self.position)
    }
}
