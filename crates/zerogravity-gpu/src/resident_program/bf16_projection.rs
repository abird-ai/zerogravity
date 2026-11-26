//! Prepared exact source operand; shared full-K64lane dots and original flight.
use super::*;
use crate::resident_leases::resident_matrix::{
    Bf16MatrixSource, PendingResidentMatrix, ResidentMatrix,
};
use crate::resident_leases::{
    ResidentInitialization, ResidentLease, ResidentService, ResidentSubmission as Flight,
    source_page,
};
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::{VerifiedMatrix, VerifiedTensorSource};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bf16ProjectionStorage {
    Streamed,
    Resident,
}
/// Source/geometry/storage preparation carries no invocation/root authority.
pub(crate) struct PreparedBf16Projection {
    source: VerifiedTensorSource,
    matrix: VerifiedMatrix,
    storage: Bf16ProjectionStorage,
    rows_per_launch: usize,
    launch_row_cap: Option<usize>,
    hoist_packed_indices: bool,
    resident: OnceLock<Arc<ResidentMatrix>>,
}
pub(crate) struct Bf16ProjectionBindings<'a> {
    pub input: &'a Handle,
    pub input_lease: &'a ResidentLease,
    pub output: &'a Handle,
    pub output_lease: &'a ResidentLease,
    pub packed: &'a ResidentLease,
    pub accumulation: &'a ResidentLease,
    pub flags: &'a ResidentLease,
    pub status: &'a ResidentLease,
    pub zero: &'a ResidentLease,
}
impl PreparedBf16Projection {
    #[cfg(any(feature = "hip-source", test))]
    pub(crate) fn new(
        source: VerifiedTensorSource,
        matrix: VerifiedMatrix,
        storage: Bf16ProjectionStorage,
    ) -> Result<Arc<Self>> {
        Self::new_with_launch_row_cap(source, matrix, storage, None)
    }
    /// An immutable selected launch-row bound. Vulkan chooses at most 8192 rows
    /// per dispatch because the grid is a device limit, not an allocation
    /// limit; the head 262144-row embedding is therefore 32 exact full-K tiles.
    pub(crate) fn new_with_launch_row_cap(
        source: VerifiedTensorSource,
        matrix: VerifiedMatrix,
        storage: Bf16ProjectionStorage,
        launch_row_cap: Option<usize>,
    ) -> Result<Arc<Self>> {
        Self::new_with_traversal(source, matrix, storage, launch_row_cap, false)
    }
    /// Resident full-K64 projection with hoisted packed indices.
    #[cfg(feature = "hip-source")]
    pub(crate) fn new_resident_hoisted(
        source: VerifiedTensorSource,
        matrix: VerifiedMatrix,
    ) -> Result<Arc<Self>> {
        Self::new_with_traversal(source, matrix, Bf16ProjectionStorage::Resident, None, true)
    }
    fn new_with_traversal(
        source: VerifiedTensorSource,
        matrix: VerifiedMatrix,
        storage: Bf16ProjectionStorage,
        launch_row_cap: Option<usize>,
        hoist_packed_indices: bool,
    ) -> Result<Arc<Self>> {
        if launch_row_cap.is_some_and(|n| n == 0 || n > 8192) {
            return Err(Error::Limit(
                "BF16 selected launch-row bound exceeds 8192".into(),
            ));
        }
        let authenticated = Bf16MatrixSource::new(source.clone(), matrix)?;
        let matrix = authenticated.matrix().clone();
        let rows = matrix.rows();
        let columns = matrix.columns();
        if rows == 0
            || rows > i32::MAX as usize
            || columns == 0
            || !columns.is_multiple_of(64)
            || rows
                .checked_mul(columns)
                .is_none_or(|n| n / 2 > i32::MAX as usize)
        {
            return Err(Error::Limit(
                "BF16 CPU64 full-K grid/address contract".into(),
            ));
        }
        let row_bytes = columns
            .checked_mul(2)
            .filter(|n| *n != 0 && *n <= source_page::PAGE_BYTES)
            .ok_or_else(|| {
                Error::Limit("BF16 prepared full-K row exceeds transport bound".into())
            })?;
        let rows_per_launch = match storage {
            Bf16ProjectionStorage::Streamed => source_page::PAGE_BYTES / row_bytes,
            Bf16ProjectionStorage::Resident => rows,
        };
        let rows_per_launch =
            launch_row_cap.map_or(rows_per_launch, |cap| rows_per_launch.min(cap).min(rows));
        Ok(Arc::new(Self {
            source,
            matrix,
            storage,
            rows_per_launch,
            launch_row_cap,
            hoist_packed_indices,
            resident: OnceLock::new(),
        }))
    }
    #[cfg(any(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification"
    ))]
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn matrix(&self) -> &VerifiedMatrix {
        &self.matrix
    }
    #[cfg(any(
        feature = "hip-source",
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification"
    ))]
    pub(crate) fn storage(&self) -> Bf16ProjectionStorage {
        self.storage
    }
    /// The cold catalogue and execution use the same immutable traversal.
    pub(crate) fn uses_hoisted_indices(&self) -> bool {
        self.hoist_packed_indices
    }
    pub(crate) fn launch_rows(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.matrix.rows())
            .step_by(self.rows_per_launch)
            .map(|first| self.rows_per_launch.min(self.matrix.rows() - first))
    }
    pub(crate) fn max_launch_rows(&self) -> usize {
        self.rows_per_launch.min(self.matrix.rows())
    }
    pub(crate) fn resident(&self) -> Option<&Arc<ResidentMatrix>> {
        self.resident.get()
    }
    pub(crate) fn require_capability(&self, service: &Arc<ResidentService>) -> Result<()> {
        service.ensure_usable()?;
        // Every Vulkan view starts at a row multiple of the source row extent,
        // so a non-multiple row cannot use the effective offset alignment.
        let row_bytes = self
            .matrix
            .columns()
            .checked_mul(2)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::Limit("BF16 prepared launch-row byte extent".into()))?;
        let launch_rows = u64::try_from(self.max_launch_rows())
            .map_err(|_| Error::Limit("BF16 prepared launch-row width".into()))?;
        let launch_bytes = launch_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::Limit("BF16 prepared launch-row binding extent".into()))?;
        let alignment = service.client().properties().memory.alignment;
        if self.launch_row_cap.is_some() && (alignment == 0 || !row_bytes.is_multiple_of(alignment))
        {
            return Err(Error::Unsupported(
                "BF16 selected Vulkan launch-row binding alignment".into(),
            ));
        }
        if service.source_identity() != self.source.source_identity()
            || self.max_launch_rows()
                > service.client().properties().hardware.max_cube_count.0 as usize
            || self.storage == Bf16ProjectionStorage::Resident
                && launch_bytes > service.client().properties().memory.max_page_size
        {
            return Err(Error::Unsupported(
                "BF16 prepared source/full-row grid/binding capability".into(),
            ));
        }
        Ok(())
    }
    #[cfg_attr(
        not(any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification"
        )),
        allow(dead_code)
    )]
    pub(crate) async fn upload(
        &self,
        init: &mut ResidentInitialization<'_>,
        flag: Option<&AtomicBool>,
    ) -> Result<Option<PendingResidentMatrix>> {
        self.upload_with_finiteness(init, flag, false).await
    }
    pub(crate) async fn upload_with_finiteness(
        &self,
        init: &mut ResidentInitialization<'_>,
        flag: Option<&AtomicBool>,
        finite: bool,
    ) -> Result<Option<PendingResidentMatrix>> {
        match self.storage {
            Bf16ProjectionStorage::Streamed => {
                if finite {
                    return Err(Error::Unsupported(
                        "native GEMM requires resident finite weights".into(),
                    ));
                }
                Ok(None)
            }
            Bf16ProjectionStorage::Resident => {
                let pending = if finite {
                    init.upload_bf16_matrix_with_finiteness(
                        self.source.clone(),
                        self.matrix.clone(),
                        flag,
                        true,
                    )
                    .await?
                } else {
                    init.upload_bf16_matrix(self.source.clone(), self.matrix.clone(), flag)
                        .await?
                };
                let range = self.matrix.byte_range()?;
                if pending.lease().bytes() as u64 != range.end - range.start
                    || pending.lease().source_identity() != self.source.source_identity()
                {
                    return Err(Error::State(
                        "BF16 prepared upload actual source/extent".into(),
                    ));
                }
                Ok(Some(pending))
            }
        }
    }
    pub(crate) fn bind_initialized(&self, pending: Option<PendingResidentMatrix>) -> Result<()> {
        match (self.storage, pending) {
            (Bf16ProjectionStorage::Streamed, None) => Ok(()),
            (Bf16ProjectionStorage::Resident, Some(pending)) => {
                let matrix = pending.bind_initialized()?;
                matrix.require_matrix(&self.source, matrix.lease().service(), &self.matrix)?;
                self.resident
                    .set(matrix)
                    .map_err(|_| Error::State("BF16 prepared operand rebound".into()))
            }
            _ => Err(Error::State(
                "BF16 prepared storage/initialization differs".into(),
            )),
        }
    }
    #[cfg(any(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification"
    ))]
    pub(crate) async fn enqueue(
        &self,
        flight: &mut Flight<'_>,
        page: &Arc<source_page::ResidentSourcePage>,
        b: Bf16ProjectionBindings<'_>,
        flag: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.enqueue_rows(flight, page, b, 1, flag).await
    }
    /// One text invocation's token-major rows, retaining the same weight window and
    /// original flight. Projection rows do not publish independent model steps.
    pub(crate) async fn enqueue_rows(
        &self,
        flight: &mut Flight<'_>,
        page: &Arc<source_page::ResidentSourcePage>,
        b: Bf16ProjectionBindings<'_>,
        input_rows: usize,
        flag: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.enqueue_rows_with_options(flight, page, b, input_rows, flag, true, false, false)
            .await
    }

    /// Native BF16/FP32 GEMM. Only the distinct native numerical profile calls
    /// this path; exact CPU64 arithmetic remains the original consumer.
    #[cfg(feature = "hip-source")]
    pub(crate) fn enqueue_native_rows(
        &self,
        flight: &mut Flight<'_>,
        b: Bf16ProjectionBindings<'_>,
        input_rows: usize,
        native: &crate::hip::native::PreparedSourceNative,
        workspace: &ResidentLease,
    ) -> Result<usize> {
        let columns = self.matrix.columns();
        let rows = self.matrix.rows();
        let (inputs, outputs) = row_extents(input_rows, columns, rows)?;
        if !rows.is_multiple_of(2) || !columns.is_multiple_of(2) {
            return Err(Error::Unsupported(
                "packed BF16 native GEMM geometry".into(),
            ));
        }
        let service = b.output_lease.service();
        let resident = self
            .resident()
            .ok_or_else(|| Error::State("native GEMM resident operand absent".into()))?;
        resident.require_matrix(&self.source, service, &self.matrix)?;
        resident.require_finite()?;
        flight.require_accesses([
            (resident.lease(), DeviceAccess::Read),
            (b.input_lease, DeviceAccess::Read),
            (b.output_lease, DeviceAccess::Write),
            (b.packed, DeviceAccess::Write),
            (b.accumulation, DeviceAccess::Write),
            (b.status, DeviceAccess::Write),
            (workspace, DeviceAccess::Write),
        ])?;
        require_view(b.input_lease, b.input, inputs * 4)?;
        require_view(b.output_lease, b.output, outputs * 4)?;
        if b.input_lease.role() != BufferRole::Activation
            || b.output_lease.role() != BufferRole::Activation
            || b.packed.role() != BufferRole::Scratch
            || b.accumulation.role() != BufferRole::Scratch
            || b.status.role() != BufferRole::Scratch
            || workspace.role() != BufferRole::Scratch
        {
            return Err(Error::State("native GEMM original lease roles".into()));
        }
        let owners = [
            resident.lease(),
            b.input_lease,
            b.output_lease,
            b.packed,
            b.accumulation,
            b.status,
            workspace,
        ];
        for (i, lease) in owners.iter().enumerate() {
            let range = BufferRange::new(&lease.handle())?;
            for peer in &owners[..i] {
                if lease.id() == peer.id() || range.overlaps(&BufferRange::new(&peer.handle())?) {
                    return Err(Error::State("native GEMM original leases alias".into()));
                }
            }
        }
        let packed = view(b.packed, 0, inputs * 2)?;
        let rounded = view(b.accumulation, 0, outputs * 2)?;
        let status = view(b.status, 0, 4)?;
        let geometry = crate::dense::Gemm::bf16_projection(input_rows, rows, columns);
        let dense = crate::hip::dense::Dense::new(native, service, workspace, flight)?;
        unsafe {
            pack::launch_unchecked(
                service.client(),
                grid(inputs / 2),
                threads(),
                arg(b.input, inputs),
                arg(&packed, inputs / 2),
                arg(&status, 1),
            );
        }
        resident.with_rows(flight, &self.source, &self.matrix, 0..rows, &mut |actual| {
            // SAFETY: original flight retains all checked non-overlapping views.
            unsafe {
                dense.enqueue(crate::dense::Command::Gemm {
                    geometry,
                    left: &packed,
                    right: actual.handle(),
                    output: &rounded,
                })
            }?;
            Ok(1)
        })?;
        unsafe {
            widen::launch_unchecked(
                service.client(),
                grid(outputs),
                threads(),
                arg(&rounded, outputs / 2),
                arg(b.output, outputs),
                arg(&status, 1),
                crate::program_kernels::NONFINITE_VALUE,
            );
        }
        Ok(3)
    }

    /// Retained alternate: reuse the exact packed input produced by an earlier
    /// projection on this same ordered flight. Current Qwen keeps per-matrix packing:
    /// streamed weights overlap pack with source transport, while the resident
    /// singleton A/B still loses wall time despite fewer launches.
    #[allow(dead_code)]
    pub(crate) async fn enqueue_rows_prepacked(
        &self,
        flight: &mut Flight<'_>,
        page: &Arc<source_page::ResidentSourcePage>,
        b: Bf16ProjectionBindings<'_>,
        input_rows: usize,
        flag: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.enqueue_rows_with_options(flight, page, b, input_rows, flag, false, false, false)
            .await
    }

    /// Retained alternate for devices/models where one status reduction after all
    /// singleton weight tiles wins. Current Qwen HIP keeps the per-tile contract.
    #[allow(dead_code)]
    pub(crate) async fn enqueue_rows_coalesced_status(
        &self,
        flight: &mut Flight<'_>,
        page: &Arc<source_page::ResidentSourcePage>,
        b: Bf16ProjectionBindings<'_>,
        flag: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.enqueue_rows_with_options(flight, page, b, 1, flag, true, true, false)
            .await
    }

    /// Retained alternate: keep exact projection arithmetic while certifying
    /// validity directly into the shared flight status. Current Qwen HIP does
    /// not select it because the lower launch count loses the decode-heavy A/B.
    #[allow(dead_code)]
    pub(crate) async fn enqueue_rows_certified(
        &self,
        flight: &mut Flight<'_>,
        page: &Arc<source_page::ResidentSourcePage>,
        b: Bf16ProjectionBindings<'_>,
        input_rows: usize,
        flag: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.enqueue_rows_with_options(flight, page, b, input_rows, flag, true, false, true)
            .await
    }

    #[allow(clippy::too_many_arguments)] // Explicit flight, bindings and retained projection modes.
    async fn enqueue_rows_with_options(
        &self,
        flight: &mut Flight<'_>,
        page: &Arc<source_page::ResidentSourcePage>,
        b: Bf16ProjectionBindings<'_>,
        input_rows: usize,
        flag: Option<&AtomicBool>,
        pack_input: bool,
        coalesce_singleton_status: bool,
        certified_status: bool,
    ) -> Result<usize> {
        if coalesce_singleton_status && certified_status {
            return Err(Error::State(
                "BF16 projection status modes are mutually exclusive".into(),
            ));
        }
        let (input_elements, output_elements) =
            row_extents(input_rows, self.matrix.columns(), self.matrix.rows())?;
        cancelled(flag)?;
        let service = b.output_lease.service();
        self.require_capability(service)?;
        flight.require_accesses([
            (b.input_lease, DeviceAccess::Read),
            (b.output_lease, DeviceAccess::Write),
            (b.packed, DeviceAccess::Write),
            (b.accumulation, DeviceAccess::Write),
            (b.flags, DeviceAccess::Write),
            (b.status, DeviceAccess::Write),
            (b.zero, DeviceAccess::Read),
        ])?;
        let columns = self.matrix.columns();
        let total_rows = self.matrix.rows();
        require_view(b.input_lease, b.input, input_elements * 4)?;
        require_view(b.output_lease, b.output, output_elements * 4)?;
        if b.output_lease.role() != BufferRole::Activation
            || b.input_lease.id() == b.output_lease.id()
            || b.zero.role() != BufferRole::ImmutableWeights
            || b.status.bytes() < 4
        {
            return Err(Error::State(
                "BF16 original projection logical roles/alias".into(),
            ));
        }
        if b.input_lease.role() != BufferRole::Activation
            || b.packed.role() != BufferRole::Scratch
            || b.accumulation.role() != BufferRole::Scratch
            || b.flags.role() != BufferRole::Scratch
            || b.status.role() != BufferRole::Scratch
            || !b.zero.is_initialized()
        {
            return Err(Error::State(
                "BF16 prepared original scratch/input/status/zero roles".into(),
            ));
        }
        let (batch, scratch_words) = row_tile(input_rows, total_rows, self.max_launch_rows())?;
        if scratch_words > service.client().properties().hardware.max_cube_count.0 as usize {
            return Err(Error::Unsupported(
                "BF16 projection token/weight row grid".into(),
            ));
        }
        view(b.accumulation, 0, scratch_words * 4)?;
        view(b.flags, 0, scratch_words * 4)?;
        if self.storage == Bf16ProjectionStorage::Resident {
            let resident = self
                .resident
                .get()
                .ok_or_else(|| Error::State("BF16 resident operand not initialized".into()))?;
            resident.require_matrix(&self.source, service, &self.matrix)?;
            flight.require_accesses([(resident.lease(), DeviceAccess::Read)])?;
        }
        if self.storage == Bf16ProjectionStorage::Streamed {
            page.require_operand(&self.source, service, &self.matrix)?;
            flight.require_accesses([(page.lease(), DeviceAccess::Write)])?;
        }
        let storage_lease = match self.storage {
            Bf16ProjectionStorage::Resident => self
                .resident
                .get()
                .ok_or_else(|| Error::State("BF16 resident operand not initialized".into()))?
                .lease(),
            Bf16ProjectionStorage::Streamed => page.lease(),
        };
        let all = [
            storage_lease,
            b.input_lease,
            b.output_lease,
            b.packed,
            b.accumulation,
            b.flags,
            b.status,
            b.zero,
        ];
        for (i, lease) in all.iter().enumerate() {
            let range = BufferRange::new(&lease.handle())?;
            for peer in &all[..i] {
                if lease.id() == peer.id() || range.overlaps(&BufferRange::new(&peer.handle())?) {
                    return Err(Error::State(
                        "BF16 original projection buffers alias".into(),
                    ));
                }
            }
        }
        let packed = view(b.packed, 0, input_elements * 2)?;
        if pack_input {
            unsafe {
                pack::launch_unchecked(
                    service.client(),
                    grid(input_elements / 2),
                    threads(),
                    arg(b.input, input_elements),
                    arg(&packed, input_elements / 2),
                    arg(&b.status.handle(), 1),
                );
            }
        }
        let hoisted = self.uses_hoisted_indices();
        let singleton = coalesce_singleton_status;
        if singleton {
            view(b.flags, 0, output_elements * 4)?;
        }
        let mut commands = usize::from(pack_input);
        for first in (0..total_rows).step_by(self.rows_per_launch) {
            cancelled(flag)?;
            let rows = self.rows_per_launch.min(total_rows - first);
            let range = first..first + rows;
            let elements = rows * batch;
            let launches = if singleton || certified_status { 1 } else { 2 };
            let accumulation = view(b.accumulation, 0, elements * 4)?;
            let flags = if singleton {
                view(b.flags, first * 4, elements * 4)?
            } else {
                view(b.flags, 0, elements * 4)?
            };
            let destination_rows = row_destination(batch, total_rows, first, rows)?;
            let destination = b
                .output
                .clone()
                .offset_start((destination_rows.start * 4) as u64)
                .offset_end(((output_elements - destination_rows.end) * 4) as u64);
            let launch = |weights: &Handle| unsafe {
                if certified_status {
                    crate::bf16_projection_kernel::project_cpu_bf16_certified::launch_unchecked(
                        service.client(),
                        CubeCount::Static(elements as u32, 1, 1),
                        CubeDim::new_1d(64),
                        arg(weights, rows * columns / 2),
                        arg(&packed, batch * columns / 2),
                        arg(&b.zero.handle(), 1),
                        arg(&accumulation, elements),
                        arg(&destination, destination_rows.len()),
                        arg(&b.status.handle(), 1),
                        rows,
                        columns,
                        total_rows,
                    );
                } else {
                    // Compact accumulation. Singleton flags keep their final row
                    // positions so one status reduction can follow all weight tiles.
                    let project = if hoisted {
                        crate::bf16_projection_kernel::project_cpu_bf16_hoisted::launch_unchecked
                    } else {
                        crate::bf16_projection_kernel::project_cpu_bf16::launch_unchecked
                    };
                    project(
                        service.client(),
                        CubeCount::Static(elements as u32, 1, 1),
                        CubeDim::new_1d(64),
                        arg(weights, rows * columns / 2),
                        arg(&packed, batch * columns / 2),
                        arg(&b.zero.handle(), 1),
                        arg(&accumulation, elements),
                        arg(&destination, destination_rows.len()),
                        arg(&flags, elements),
                        rows,
                        columns,
                        total_rows,
                    );
                    if !singleton {
                        projection_status::launch_unchecked(
                            service.client(),
                            grid(elements),
                            threads(),
                            arg(&flags, elements),
                            arg(&b.status.handle(), 1),
                        );
                    }
                }
            };
            match self.storage {
                Bf16ProjectionStorage::Resident => {
                    let resident = self.resident.get().ok_or_else(|| {
                        Error::State("BF16 resident operand not initialized".into())
                    })?;
                    resident.with_rows(
                        flight,
                        &self.source,
                        &self.matrix,
                        range.clone(),
                        &mut |actual| {
                            if !actual.matrix().same_binding(&self.matrix)
                                || actual.rows() != range
                                || actual.bytes() != rows * columns * 2
                            {
                                return Err(Error::State("BF16 resident exact full-K view".into()));
                            }
                            launch(actual.handle());
                            Ok(launches)
                        },
                    )?;
                }
                Bf16ProjectionStorage::Streamed => {
                    page.with_rows(flight, &self.matrix, range.clone(), flag, |actual| {
                        if !actual.matrix().same_binding(&self.matrix)
                            || actual.rows() != range
                            || actual.bytes() != rows * columns * 2
                        {
                            return Err(Error::State("BF16 streamed exact full-K view".into()));
                        }
                        launch(actual.handle());
                        let mut outputs = vec![b.output_lease.clone(), b.accumulation.clone()];
                        if certified_status {
                            outputs.push(b.status.clone());
                        } else {
                            outputs.push(b.flags.clone());
                            if !singleton {
                                outputs.push(b.status.clone());
                            }
                        }
                        Ok(outputs)
                    })
                    .await?;
                }
            }
            commands += launches;
        }
        if singleton {
            let flags = view(b.flags, 0, output_elements * 4)?;
            unsafe {
                projection_status::launch_unchecked(
                    service.client(),
                    grid(output_elements),
                    threads(),
                    arg(&flags, output_elements),
                    arg(&b.status.handle(), 1),
                );
            }
            commands += 1;
        }
        Ok(commands)
    }
}
fn row_tile(input_rows: usize, outputs: usize, tile_rows: usize) -> Result<(usize, usize)> {
    row_extents(input_rows, 1, outputs)?;
    if tile_rows == 0 || tile_rows > outputs {
        return Err(Error::Limit("BF16 projection weight row tile".into()));
    }
    Ok((input_rows, input_rows * tile_rows))
}
fn row_destination(
    input_rows: usize,
    outputs: usize,
    first: usize,
    tile_rows: usize,
) -> Result<std::ops::Range<usize>> {
    let (_, complete) = row_extents(input_rows, 1, outputs)?;
    let last = first
        .checked_add(tile_rows)
        .filter(|last| tile_rows != 0 && *last <= outputs)
        .ok_or_else(|| Error::Limit("BF16 projection output row window".into()))?;
    let end = (input_rows - 1)
        .checked_mul(outputs)
        .and_then(|base| base.checked_add(last))
        .filter(|end| *end <= complete)
        .ok_or_else(|| Error::Limit("BF16 projection strided output extent".into()))?;
    Ok(first..end)
}
// One bounded cohort can flatten eight independent 64-row invocations.
// The weight tile, arithmetic tree and exact output view are unchanged.
pub(crate) const MAX_PROJECTION_INPUT_ROWS: usize = 8 * 64;
fn row_extents(input_rows: usize, columns: usize, outputs: usize) -> Result<(usize, usize)> {
    if !(1..=MAX_PROJECTION_INPUT_ROWS).contains(&input_rows) || columns == 0 || outputs == 0 {
        return Err(Error::Limit("BF16 projection source row geometry".into()));
    }
    let elements = |width: usize| {
        input_rows
            .checked_mul(width)
            .filter(|n| *n <= u32::MAX as usize && n.checked_mul(4).is_some())
            .ok_or_else(|| Error::Limit("BF16 projection row extent overflow".into()))
    };
    Ok((elements(columns)?, elements(outputs)?))
}
fn cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("BF16 prepared projection cancelled".into()));
    }
    Ok(())
}
fn require_view(lease: &ResidentLease, handle: &Handle, bytes: usize) -> Result<()> {
    if !bytes.is_multiple_of(4) || bytes == 0 {
        return Err(Error::State("BF16 widened F32 view extent".into()));
    }
    let layout = zerogravity_core::placement::PreparedTensorLayout::dense(4, 4, bytes / 4)?;
    BufferRange::new(&lease.handle())?.require_view(&BufferRange::new(handle)?, layout)
}
fn view(lease: &ResidentLease, offset: usize, bytes: usize) -> Result<Handle> {
    lease.view(offset, bytes)
}
fn arg(handle: &Handle, count: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(handle.clone(), count) }
}
fn grid(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(32), 1, 1)
}
fn threads() -> CubeDim {
    CubeDim::new_1d(32)
}

// Bit packing and terminal arithmetic flags are shared by every CPU64 consumer.
#[cube(launch_unchecked)]
pub(crate) fn pack(input: &[f32], output: &mut [u32], status: &[Atomic<u32>]) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let a = u32::reinterpret(input[2usize * i]);
        let b = u32::reinterpret(input[2usize * i + 1usize]);
        if (a & 0xffffu32) != 0u32
            || (b & 0xffffu32) != 0u32
            || (a & 0x7f800000u32) == 0x7f800000u32
            || (b & 0x7f800000u32) == 0x7f800000u32
        {
            status[0].fetch_or(1u32);
        }
        output[i] = (a >> 16u32) | (b & 0xffff0000u32);
    }
}
/// Projection-specific pack sibling. It preserves the canonical packed bytes,
/// but publishes the input-nonfinite bit here as well as the source-invalid bit.
/// Composed with `project_cpu_bf16_input_certified`, the final shared status is
/// identical to `pack` + canonical projection while the GEMM avoids rechecking
/// the same packed activation once per output row. Retained dormant because the
/// current gfx1150 consumer does not win direct projection device time.
#[allow(dead_code)]
#[cube(launch_unchecked)]
pub(crate) fn pack_projection_input(input: &[f32], output: &mut [u32], status: &[Atomic<u32>]) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let a = u32::reinterpret(input[2usize * i]);
        let b = u32::reinterpret(input[2usize * i + 1usize]);
        let non_bf16 = (a & 0xffffu32) != 0u32 || (b & 0xffffu32) != 0u32;
        let nonfinite =
            (a & 0x7f800000u32) == 0x7f800000u32 || (b & 0x7f800000u32) == 0x7f800000u32;
        let mut flags = 0u32;
        if non_bf16 || nonfinite {
            flags |= 1u32;
        }
        if nonfinite {
            flags |= 2u32;
        }
        if flags != 0u32 {
            status[0].fetch_or(flags);
        }
        output[i] = (a >> 16u32) | (b & 0xffff0000u32);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn projection_status(input: &[u32], status: &[Atomic<u32>]) {
    let i = ABSOLUTE_POS;
    if i < input.len() && input[i] != 0u32 {
        status[0].fetch_or(input[i]);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn widen(
    input: &[u32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] nonfinite_mask: u32,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let bits =
            ((input[i / 2usize] >> (u32::cast_from(i % 2usize) * 16u32)) & 0xffffu32) << 16u32;
        if (bits & 0x7f800000u32) == 0x7f800000u32 {
            status[0].fetch_or(nonfinite_mask);
        }
        output[i] = f32::reinterpret(bits);
    }
}

#[cfg(test)]
mod row_tests {
    use super::{row_destination, row_extents, row_tile};

    #[test]
    fn whole_and_partial_weight_windows_group_all_rows() {
        for input_rows in [1, 17, 47, 62, 64, 376, 512] {
            for outputs in [1, 128, 6144, 248320] {
                assert_eq!(
                    row_tile(input_rows, outputs, outputs).unwrap(),
                    (input_rows, input_rows * outputs)
                );
                if outputs > 1 {
                    assert_eq!(
                        row_tile(input_rows, outputs, outputs - 1).unwrap(),
                        (input_rows, input_rows * (outputs - 1))
                    );
                }
            }
        }
        for (inputs, outputs, tile) in [
            (0, 64, 64),
            (513, 64, 64),
            (64, 0, 1),
            (64, 64, 0),
            (64, 64, 65),
            (64, u32::MAX as usize, 1),
        ] {
            assert!(row_tile(inputs, outputs, tile).is_err());
        }
    }

    #[test]
    fn every_partial_window_fits_the_complete_output_owner() {
        for inputs in [1, 17, 47, 62, 64, 376, 512] {
            for outputs in [1usize, 128, 6144, 248320] {
                for tile in [1, outputs.div_ceil(3), outputs] {
                    for first in (0..outputs).step_by(tile) {
                        let rows = tile.min(outputs - first);
                        let span = row_destination(inputs, outputs, first, rows).unwrap();
                        assert_eq!(span.start, first);
                        assert_eq!(span.len(), (inputs - 1) * outputs + rows);
                        assert!(span.end <= inputs * outputs);
                        assert_eq!(span.end - 1, (inputs - 1) * outputs + first + rows - 1);
                        let (batch, scratch) = row_tile(inputs, outputs, rows).unwrap();
                        assert_eq!(batch, inputs);
                        assert_eq!(scratch, inputs * rows);
                        assert!(scratch <= inputs * outputs);
                    }
                }
            }
        }
        for (inputs, outputs, first, rows) in [
            (0, 64, 0, 64),
            (513, 64, 0, 64),
            (1, 0, 0, 1),
            (1, 64, 0, 0),
            (1, 64, 64, 1),
            (64, 64, 63, 2),
            (1, 64, usize::MAX, 1),
            (64, usize::MAX, 0, 1),
        ] {
            assert!(row_destination(inputs, outputs, first, rows).is_err());
        }
    }

    #[test]
    fn rows_price_complete_inputs_and_outputs_without_changing_weight_tiles() {
        assert_eq!(row_extents(1, 1024, 3072).unwrap(), (1024, 3072));
        assert_eq!(row_extents(47, 2048, 6144).unwrap(), (47 * 2048, 47 * 6144));
        assert_eq!(
            row_extents(64, 6144, 248320).unwrap(),
            (64 * 6144, 64 * 248320)
        );
    }

    #[test]
    fn invalid_or_overflowing_complete_row_extents_refuse() {
        for (rows, columns, outputs) in [
            (0, 64, 64),
            (513, 64, 64),
            (1, 0, 64),
            (1, 64, 0),
            (64, usize::MAX, 64),
            (64, 64, usize::MAX),
            (64, u32::MAX as usize, 64),
            (64, 64, u32::MAX as usize),
        ] {
            assert!(row_extents(rows, columns, outputs).is_err());
        }
    }
}
