//! Closed MLP records attached to the original shared image/module owner.
use super::super::mlp_manifests::{MlpLaunch, MlpWorkspaceClear};
use super::super::projection_cubin::resident::{
    NativeRetentionState, NativeRetentionWitness, PreparedImages,
};
use super::*;
use crate::resident_leases::ResidentInitialization;
use cubecl::cuda::{CubinIo, CubinWorkspaceClear, OwnedCudaKernel};

struct Launch {
    kernel: Arc<OwnedCudaKernel>,
    bindings: &'static [ProjectionBinding],
    clear: Option<CubinWorkspaceClear>,
}

pub(in crate::cuda::native) struct MlpPrograms {
    programs: BTreeMap<MlpProjection, Vec<Launch>>,
    images: Arc<PreparedImages>,
}

pub(in crate::cuda::native) struct MlpBindings<'a> {
    pub(in crate::cuda::native) weights: &'a ResidentLease,
    pub(in crate::cuda::native) input: &'a ResidentLease,
    pub(in crate::cuda::native) output: &'a ResidentLease,
    pub(in crate::cuda::native) workspace: Option<&'a ResidentLease>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::cuda::native) struct MlpSubmitted {
    pub(in crate::cuda::native) kernels: usize,
    pub(in crate::cuda::native) clears: usize,
}

fn attach_clear(
    clear: Option<MlpWorkspaceClear>,
    bindings: &[ProjectionBinding],
) -> Result<Option<CubinWorkspaceClear>> {
    let Some(clear) = clear else {
        return Ok(None);
    };
    if clear.binding() != ProjectionBinding::Workspace
        || clear.binding_bytes() != WORKSPACE_BYTES
        || clear.byte_offset() != 0
        || clear.value() != 0
        || clear.io() != CubinIo::Write
        || bindings
            .iter()
            .filter(|binding| **binding == clear.binding())
            .count()
            != 1
    {
        return Err(Error::State(
            "closed original MLP workspace clear contract".into(),
        ));
    }
    let index = bindings
        .iter()
        .position(|binding| *binding == clear.binding())
        .ok_or_else(|| Error::State("MLP clear workspace binding absent".into()))?;
    let native = match clear.bytes() {
        32 => CubinWorkspaceClear::zero32(index),
        64 => CubinWorkspaceClear::zero64(index),
        _ => {
            return Err(Error::Unsupported(
                "closed original MLP clear extent".into(),
            ));
        }
    }
    .map_err(native_error)?;
    Ok(Some(native))
}

fn ordered_launches(
    operations: Vec<MlpOperation>,
) -> Result<Vec<(MlpLaunch, Option<CubinWorkspaceClear>)>> {
    let mut pending = None;
    let mut launches = Vec::new();
    for operation in operations {
        match operation {
            MlpOperation::Clear(clear) => {
                if pending.replace(clear).is_some() {
                    return Err(Error::State("MLP repeated workspace clear".into()));
                }
            }
            MlpOperation::Kernel(launch) => {
                let clear = attach_clear(pending.take(), launch.bindings())?;
                launches.push((launch, clear));
            }
        }
    }
    if pending.is_some() || launches.is_empty() {
        return Err(Error::State(
            "MLP clear requires its following original kernel".into(),
        ));
    }
    Ok(launches)
}

impl MlpPrograms {
    /// Caller owns the real original W-upload initialization before preparation.
    pub(in crate::cuda::native) fn prepare(
        images: Arc<PreparedImages>,
        projections: &[MlpProjection],
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        if projections.is_empty() || projections.len() > 6 {
            return Err(Error::Malformed("closed unique MLP records1..6".into()));
        }
        init.retain_completion_owner(images.clone())?;
        let mut programs = BTreeMap::new();
        for &projection in projections {
            if programs.contains_key(&projection) {
                return Err(Error::Malformed("duplicate original MLP program".into()));
            }
            let mut launches = Vec::new();
            for (launch, clear) in ordered_launches(projection.operations())? {
                not_cancelled(cancelled)?;
                let image = images.image(launch.image())?;
                let bindings = launch.bindings();
                // SAFETY: fixed observed ABI, admitted original issuer/image,
                // same original set retained by the armed W initialization.
                let kernel = unsafe { images.set().prepare(image, launch.into_manifest()) }
                    .map_err(native_error)?;
                init.retain_completion_owner(kernel.clone())?;
                launches.push(Launch {
                    kernel,
                    bindings,
                    clear,
                });
            }
            programs.insert(projection, launches);
        }
        let prepared = Arc::new(Self { programs, images });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }

    pub(in crate::cuda::native) fn retention_witness(&self) -> Result<MlpRetentionWitness> {
        let kernels = self
            .programs
            .values()
            .flatten()
            .map(|launch| launch.kernel.clone())
            .collect::<Vec<_>>();
        Ok(MlpRetentionWitness {
            images: Arc::downgrade(&self.images),
            native: self.images.retention_witness(&kernels)?,
        })
    }

    /// No host activation input exists here: all operands are original leases.
    pub(in crate::cuda::native) fn enqueue(
        &self,
        projection: MlpProjection,
        bindings: MlpBindings<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<MlpSubmitted> {
        not_cancelled(cancelled)?;
        let service = bindings.weights.service();
        service.ensure_usable()?;
        if bindings.weights.id() == bindings.input.id()
            || bindings.weights.id() == bindings.output.id()
            || bindings.input.id() == bindings.output.id()
            || bindings.weights.bytes() != projection.weight_bytes()
            || bindings.input.bytes() != projection.input_bytes()
            || bindings.output.bytes() != projection.output_bytes()
            || [bindings.input, bindings.output].into_iter().any(|lease| {
                !Arc::ptr_eq(service, lease.service())
                    || lease.source_identity() != bindings.weights.source_identity()
            })
        {
            return Err(Error::State("original MLP source/service/geometry".into()));
        }
        let launches = self
            .programs
            .get(&projection)
            .ok_or_else(|| Error::State("unprepared original MLP operation".into()))?;
        let mut submitted = MlpSubmitted::default();
        for launch in launches {
            not_cancelled(cancelled)?;
            let mut handles = Vec::with_capacity(launch.bindings.len());
            for binding in launch.bindings {
                let lease = match binding {
                    ProjectionBinding::W => bindings.weights,
                    ProjectionBinding::X => bindings.input,
                    ProjectionBinding::Y => bindings.output,
                    ProjectionBinding::Workspace => bindings
                        .workspace
                        .ok_or_else(|| Error::State("original MLP workspace absent".into()))?,
                };
                if !Arc::ptr_eq(service, lease.service())
                    || lease.source_identity() != bindings.weights.source_identity()
                    || *binding == ProjectionBinding::Workspace
                        && (lease.bytes() != WORKSPACE_BYTES
                            || [
                                bindings.weights.id(),
                                bindings.input.id(),
                                bindings.output.id(),
                            ]
                            .contains(&lease.id()))
                {
                    return Err(Error::State(
                        "original MLP full workspace/service differs".into(),
                    ));
                }
                handles.push(lease.handle());
            }
            // SAFETY: caller retains this program/source and all exact original
            // leases/exclusive accesses in its armed physical producer flight.
            unsafe {
                if let Some(clear) = launch.clear {
                    launch.kernel.enqueue_with_workspace_clear(handles, clear)
                } else {
                    launch.kernel.enqueue(handles)
                }
            }
            .map_err(native_error)?;
            submitted.kernels += 1;
            submitted.clears += usize::from(launch.clear.is_some());
        }
        Ok(submitted)
    }

    /// Shared-set callers retire all operation records before one set unload.
    pub(in crate::cuda::native) unsafe fn retire_kernels_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        for launches in self.programs.values() {
            for launch in launches {
                // SAFETY: caller proves every original submission completed;
                // enclosing armed flight retains all owners on partial error.
                unsafe { launch.kernel.retire_after_physical_completion() }?;
            }
        }
        Ok(())
    }

    pub(in crate::cuda::native) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        // SAFETY: same original completion proof covers records and shared set.
        unsafe { self.retire_kernels_after_physical_completion() }?;
        // SAFETY: no records remain; original read-only flight retains owner.
        unsafe { self.images.set().retire_after_physical_completion() }
    }
}

pub(in crate::cuda::native) struct MlpRetentionWitness {
    images: std::sync::Weak<PreparedImages>,
    native: NativeRetentionWitness,
}
impl MlpRetentionWitness {
    pub(in crate::cuda::native) fn inspect(&self) -> Result<NativeRetentionState> {
        if self.images.upgrade().is_none() {
            return Err(Error::State(
                "MLP shared images lost from original quarantine".into(),
            ));
        }
        self.native.inspect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_ordered_operations_keep_thirteen_kernels_and_two_native_clears() {
        let phases = [
            GdnPhase::InitialOne,
            GdnPhase::CachedDecode,
            GdnPhase::Initial64,
            GdnPhase::ContinuedTwo,
            GdnPhase::CachedDecode,
        ];
        let mut kernels = 0;
        let mut clears = 0;
        for projection in phases
            .into_iter()
            .flat_map(|phase| MlpProjection::for_phase(phase).unwrap())
        {
            let launches = ordered_launches(projection.operations()).unwrap();
            kernels += launches.len();
            clears += launches.iter().filter(|(_, clear)| clear.is_some()).count();
        }
        assert_eq!((kernels, clears), (13, 2));
    }

    #[test]
    fn native_clear_cannot_detach_or_repeat_or_attach_without_workspace() {
        let mut missing_kernel = MlpProjection::Down64.operations();
        missing_kernel.pop();
        assert!(ordered_launches(missing_kernel).is_err());
        let clear = match MlpProjection::Down64.operations().remove(0) {
            MlpOperation::Clear(clear) => clear,
            MlpOperation::Kernel(_) => panic!("original clear order"),
        };
        assert!(
            ordered_launches(vec![MlpOperation::Clear(clear), MlpOperation::Clear(clear)]).is_err()
        );
        assert!(
            attach_clear(
                Some(clear),
                &[
                    ProjectionBinding::W,
                    ProjectionBinding::X,
                    ProjectionBinding::Y
                ]
            )
            .is_err()
        );
        assert!(
            attach_clear(
                Some(clear),
                &[ProjectionBinding::Workspace, ProjectionBinding::Workspace]
            )
            .is_err()
        );
        assert!(
            attach_clear(
                Some(clear),
                &[ProjectionBinding::W, ProjectionBinding::Workspace]
            )
            .unwrap()
            .is_some()
        );
        assert!(ordered_launches(Vec::new()).is_err());
    }
}
