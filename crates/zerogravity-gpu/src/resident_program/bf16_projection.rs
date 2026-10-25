//! Prepared exact source operand; one full-K64lane dot implementation and original flight.
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
    #[cfg(any(feature = "qwen-source-qualification", test))]
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
            resident: OnceLock::new(),
        }))
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn matrix(&self) -> &VerifiedMatrix {
        &self.matrix
    }
    pub(crate) fn storage(&self) -> Bf16ProjectionStorage {
        self.storage
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
    pub(crate) async fn upload(
        &self,
        init: &mut ResidentInitialization<'_>,
        flag: Option<&AtomicBool>,
    ) -> Result<Option<PendingResidentMatrix>> {
        match self.storage {
            Bf16ProjectionStorage::Streamed => Ok(None),
            Bf16ProjectionStorage::Resident => {
                let pending = init
                    .upload_bf16_matrix(self.source.clone(), self.matrix.clone(), flag)
                    .await?;
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
    pub(crate) async fn enqueue(
        &self,
        flight: &mut Flight<'_>,
        page: &Arc<source_page::ResidentSourcePage>,
        b: Bf16ProjectionBindings<'_>,
        flag: Option<&AtomicBool>,
    ) -> Result<usize> {
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
        require_view(b.input_lease, b.input, columns * 4)?;
        require_view(b.output_lease, b.output, total_rows * 4)?;
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
        view(b.accumulation, 0, self.max_launch_rows() * 4)?;
        view(b.flags, 0, self.max_launch_rows() * 4)?;
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
        let packed = view(b.packed, 0, columns * 2)?;
        unsafe {
            pack::launch_unchecked(
                service.client(),
                grid(columns / 2),
                threads(),
                arg(b.input, columns),
                arg(&packed, columns / 2),
                arg(&b.status.handle(), 1),
            );
        }
        let mut commands = 1;
        for first in (0..total_rows).step_by(self.rows_per_launch) {
            cancelled(flag)?;
            let rows = self.rows_per_launch.min(total_rows - first);
            let range = first..first + rows;
            let destination = b
                .output
                .clone()
                .offset_start((first * 4) as u64)
                .offset_end(((total_rows - range.end) * 4) as u64);
            let accumulation = view(b.accumulation, 0, rows * 4)?;
            let flags = view(b.flags, 0, rows * 4)?;
            let launch = |weights: &Handle| unsafe {
                crate::bf16_projection_kernel::project_cpu_bf16::launch_unchecked(
                    service.client(),
                    CubeCount::Static(rows as u32, 1, 1),
                    CubeDim::new_1d(64),
                    arg(weights, rows * columns / 2),
                    arg(&packed, columns / 2),
                    arg(&b.zero.handle(), 1),
                    arg(&accumulation, rows),
                    arg(&destination, rows),
                    arg(&flags, rows),
                    rows,
                    columns,
                );
                projection_status::launch_unchecked(
                    service.client(),
                    grid(rows),
                    threads(),
                    arg(&flags, rows),
                    arg(&b.status.handle(), 1),
                );
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
                            Ok(2)
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
                        Ok(vec![
                            b.output_lease.clone(),
                            b.accumulation.clone(),
                            b.flags.clone(),
                            b.status.clone(),
                        ])
                    })
                    .await?;
                }
            }
            commands += 2;
        }
        Ok(commands)
    }
}
fn cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("BF16 prepared projection cancelled".into()));
    }
    Ok(())
}
fn require_view(lease: &ResidentLease, handle: &Handle, bytes: usize) -> Result<()> {
    let owner = BufferRange::new(&lease.handle())?;
    let actual = BufferRange::new(handle)?;
    actual.require_exact_len(bytes)?;
    if actual.service != owner.service
        || actual.memory != owner.memory
        || actual.bytes.start < owner.bytes.start
        || actual.bytes.end > owner.bytes.end
    {
        return Err(Error::State(
            "BF16 borrowed view differs from original lease".into(),
        ));
    }
    Ok(())
}
fn view(lease: &ResidentLease, offset: usize, bytes: usize) -> Result<Handle> {
    if bytes == 0 || offset.checked_add(bytes).is_none_or(|n| n > lease.bytes()) {
        return Err(Error::State(
            "BF16 original projection scratch extent".into(),
        ));
    }
    Ok(lease
        .handle()
        .offset_start(offset as u64)
        .offset_end((lease.bytes() - offset - bytes) as u64))
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
