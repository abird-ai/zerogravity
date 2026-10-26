//! Root-free two-shape B4 geometry over one exclusive physical backing.
use super::*;
use crate::cuda::native::text_model::TextModelBinding;
use crate::resident_leases::backing_pool::{
    ResidentPlaneLayout, ResidentViewLayout, ResidentWorkspace, ResidentWorkspaceGeometry,
};
use zerogravity_core::accelerator::program::ActivationPrecision;

pub(crate) struct DecodeCohortWorkspace {
    geometries: [Option<Arc<ResidentWorkspaceGeometry>>; 2],
    source: VerifiedTensorSource,
    _backing: Arc<ResidentWorkspace>,
    _metadata: WindowWorkspace,
}
impl DecodeCohortWorkspace {
    fn layouts(
        capture: TextKernelCapture,
        scopes: &[Arc<TextModelBinding>],
    ) -> Result<(Vec<ResidentPlaneLayout>, [Vec<ResidentViewLayout>; 2])> {
        if scopes.len() != 6
            || scopes[0].call().rows().len() != 47
            || scopes[5].call().rows().len() != 1
        {
            return Err(Error::Unsupported(
                "B4 fixed-prompt immutable source geometries only".into(),
            ));
        }
        if scopes.iter().any(|scope| {
            scope.native_capture() != capture || !scope.source().same_issuer(scopes[0].source())
        }) {
            return Err(Error::State(
                "B4 workspace immutable source/profile differs".into(),
            ));
        }
        let phases = crate::cuda::native::cohort_native::CohortPhases::for_capture(capture);
        let cached_only = !phases.contains(true);
        let mut planes = Vec::with_capacity(327);
        let mut views = [Vec::with_capacity(327), Vec::with_capacity(327)];
        for (scope_index, scope) in [scopes[0].clone(), scopes[5].clone()].iter().enumerate() {
            if cached_only && scope_index == 0 {
                continue;
            }
            let plan = scope.record().plan();
            if plan.values().len() != 327
                || plan
                    .precisions()
                    .iter()
                    .any(|p| *p != ActivationPrecision::Bf16WidenedF32)
            {
                return Err(Error::State("B4 actual327 source BF16 values".into()));
            }
            for (index, value) in plan.values().iter().enumerate() {
                let bytes = value
                    .capacity_elements()?
                    .checked_mul(8)
                    .filter(|n| *n > 0 && *n <= 16 * 1024 * 1024)
                    .ok_or_else(|| Error::Limit("B4 source whole producer extent".into()))?;
                let aligned = bytes
                    .checked_add(511)
                    .map(|n| n & !511)
                    .ok_or_else(|| Error::Limit("B4 source alignment".into()))?;
                if scope_index == 0 || cached_only {
                    planes.push(ResidentPlaneLayout {
                        bytes: aligned,
                        alignment: 512,
                        role: BufferRole::Activation,
                    });
                } else if aligned > planes[index].bytes {
                    return Err(Error::State(
                        "B4 cached value exceeds admitted initial plane".into(),
                    ));
                }
                views[scope_index].push(ResidentViewLayout {
                    plane: index,
                    offset: 0,
                    bytes,
                    alignment: 512,
                    role: BufferRole::Activation,
                });
            }
        }
        if planes.len() != 327
            || (!cached_only && planes.iter().map(|p| p.bytes).sum::<usize>() != 316_002_304)
            || (cached_only && !views[0].is_empty())
        {
            return Err(Error::State("B4 exact188 source workspace bound".into()));
        }
        Ok((planes, views))
    }
    pub(crate) fn host_bytes() -> Result<usize> {
        ResidentWorkspace::metadata_bytes(327, 654)?
            .checked_add(1024 * 1024)
            .ok_or_else(|| Error::Limit("B4 shared geometry metadata".into()))
    }
    pub(crate) fn allocation_layout(
        capture: TextKernelCapture,
        scopes: &[Arc<TextModelBinding>],
    ) -> Result<Vec<usize>> {
        Ok(Self::layouts(capture, scopes)?
            .0
            .into_iter()
            .map(|plane| plane.bytes)
            .collect())
    }
    pub(crate) fn prepare(
        source: VerifiedTensorSource,
        capture: TextKernelCapture,
        service: Arc<ResidentService>,
        scopes: &[Arc<TextModelBinding>],
    ) -> Result<Arc<Self>> {
        let (planes, views) = Self::layouts(capture, scopes)?;
        if service.source_identity() != source.source_identity() {
            return Err(Error::State("B4 immutable source/service workspace".into()));
        }
        let metadata = source.reserve_workspace(1024 * 1024)?;
        let admission = source
            .reserve_workspace(ResidentWorkspace::metadata_bytes(planes.len(), 654)?)?
            .into_host_workspace()?;
        let backing = ResidentWorkspace::prepare(service, admission, &planes, 654)?;
        let geometries = [
            if views[0].is_empty() {
                None
            } else {
                Some(backing.geometry(&views[0])?)
            },
            Some(backing.geometry(&views[1])?),
        ];
        Ok(Arc::new(Self {
            geometries,
            source,
            _backing: backing,
            _metadata: metadata,
        }))
    }
    pub(crate) fn geometry(
        &self,
        inputs: &DecodeCohortInputs,
    ) -> Result<Arc<ResidentWorkspaceGeometry>> {
        crate::cuda::native::cohort_native::require_input_shape(inputs.rows())?;
        if inputs.rows().len() != 4
            || inputs.rows().iter().any(|row| {
                !row.source().same_issuer(&self.source)
                    || row.prompt_tokens() != 47
                    || row.rows() != inputs.rows()[0].rows()
            })
        {
            return Err(Error::State(
                "B4 workspace every original source/profile/frontier".into(),
            ));
        }
        let index = match (inputs.rows()[0].rows(), inputs.rows()[0].base()) {
            (47, 0) => 0,
            (1, 47..=53) => 1,
            _ => {
                return Err(Error::Unsupported(
                    "B4 fixed-prompt exact prefill/decode geometry".into(),
                ));
            }
        };
        self.geometries[index].clone().ok_or_else(|| {
            Error::Unsupported("INITIAL B4 is outside cached-only captured capability".into())
        })
    }
}
