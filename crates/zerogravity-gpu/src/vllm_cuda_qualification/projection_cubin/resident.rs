//! Original actor programs and retained same-service native admission only.
use super::super::cubin_manifests::{ProjectionBinding, WORKSPACE_BYTES};
use super::*;
use crate::resident_leases::{ForeignProviderCharge, ResidentInitialization};
use cubecl::cuda::{
    Bf16BlasSources, OwnedCudaKernel, OwnedCudaModuleSet, OwnedCudaQueue, VerifiedCubin,
};

struct NativeAdmission {
    _source_owner: CompletionOwner,
    _image_admissions: Vec<WindowWorkspace>,
    _native_charge: Arc<ForeignProviderCharge>,
    expected_native_bytes: usize,
}

/// Unsealed payloads drop before their original capacity admissions.
pub(in crate::vllm_cuda_qualification) struct AdmittedImages {
    payloads: Vec<(ProjectionImage, Vec<u8>)>,
    activation: Option<Vec<u8>>,
    owner: CompletionOwner,
    queue: Arc<OwnedCudaQueue>,
}

/// One original actor/module set, shared by closed mathematical operations.
pub(crate) struct PreparedImages {
    images: BTreeMap<ProjectionImage, Arc<VerifiedCubin>>,
    activation: Option<Arc<VerifiedCubin>>,
    set: Arc<OwnedCudaModuleSet>,
    owner: CompletionOwner,
}
impl PreparedImages {
    pub(in crate::vllm_cuda_qualification) fn provider_bytes(&self) -> Result<usize> {
        self.owner
            .downcast_ref::<NativeAdmission>()
            .map(|owner| owner._native_charge.bytes())
            .ok_or_else(|| Error::State("actual native admission owner type".into()))
    }

    pub(in crate::vllm_cuda_qualification) fn require_service(
        &self,
        service: &Arc<ResidentService>,
    ) -> Result<()> {
        let owner = self
            .owner
            .downcast_ref::<NativeAdmission>()
            .ok_or_else(|| Error::State("original native composite owner type".into()))?;
        owner._native_charge.require_service(service)?;
        if owner._native_charge.bytes() != owner.expected_native_bytes
            || owner._image_admissions.is_empty()
            || owner
                ._image_admissions
                .iter()
                .any(|admission| admission.source().source_identity() != service.source_identity())
        {
            return Err(Error::State(
                "original native image/source admission differs".into(),
            ));
        }
        Ok(())
    }
    pub(in crate::vllm_cuda_qualification) fn retention_witness(
        &self,
        kernels: &[Arc<OwnedCudaKernel>],
    ) -> Result<NativeRetentionWitness> {
        let owner = self
            .owner
            .downcast_ref::<NativeAdmission>()
            .ok_or_else(|| Error::State("original shared native composite owner type".into()))?;
        Ok(NativeRetentionWitness {
            images: self
                .images
                .values()
                .chain(self.activation.iter())
                .map(Arc::downgrade)
                .collect(),
            kernels: kernels.iter().map(Arc::downgrade).collect(),
            set: Arc::downgrade(&self.set),
            owner: Arc::downgrade(&self.owner),
            charge: Arc::downgrade(&owner._native_charge),
        })
    }
    pub(in crate::vllm_cuda_qualification) fn activation(&self) -> Result<Arc<VerifiedCubin>> {
        self.activation
            .clone()
            .ok_or_else(|| Error::State("closed original activation image absent".into()))
    }
    pub(in crate::vllm_cuda_qualification) fn image(
        &self,
        kind: ProjectionImage,
    ) -> Result<Arc<VerifiedCubin>> {
        self.images
            .get(&kind)
            .cloned()
            .ok_or_else(|| Error::State("closed original operation image absent".into()))
    }
    pub(in crate::vllm_cuda_qualification) fn set(&self) -> &Arc<OwnedCudaModuleSet> {
        &self.set
    }
    pub(in crate::vllm_cuda_qualification) fn completion_owner(&self) -> CompletionOwner {
        self.owner.clone()
    }
}

struct Launch {
    kernel: Arc<OwnedCudaKernel>,
    bindings: &'static [ProjectionBinding],
}

pub(in crate::vllm_cuda_qualification) struct ProjectionPrograms {
    programs: BTreeMap<Projection, Vec<Launch>>,
    _images: Vec<Arc<VerifiedCubin>>,
    set: Arc<OwnedCudaModuleSet>,
    _owner: CompletionOwner,
}

pub(in crate::vllm_cuda_qualification) struct ProjectionBindings<'a> {
    pub(in crate::vllm_cuda_qualification) weights: &'a ResidentLease,
    pub(in crate::vllm_cuda_qualification) input: &'a ResidentLease,
    pub(in crate::vllm_cuda_qualification) output: &'a ResidentLease,
    pub(in crate::vllm_cuda_qualification) workspace: Option<&'a ResidentLease>,
}

enum ImageScope {
    SourceRotary,
    ShortHf(
        zerogravity_core::accelerator::program::FullModelProfile,
        bool,
    ),
    Projection,
    Connected,
    Canonical47Mlp,
    Qwen2bMlp,
    Canonical47Qkv,
    Canonical47QkKv,
    Canonical47Fa,
    Canonical47Layer3,
    Canonical47Full,
    Canonical47Natural,
    Attention,
    AttentionGate,
    FullModel,
    VocabularyHead,
}

impl AdmittedImages {
    pub(in crate::vllm_cuda_qualification) fn reserve(
        service: &Arc<ResidentService>,
        source_owner: CompletionOwner,
        images: Vec<CubinImage>,
    ) -> Result<Self> {
        Self::reserve_with_activation(service, source_owner, images, None)
    }
    pub(in crate::vllm_cuda_qualification) fn reserve_with_activation(
        service: &Arc<ResidentService>,
        source_owner: CompletionOwner,
        images: Vec<CubinImage>,
        activation: Option<CubinActivationImage>,
    ) -> Result<Self> {
        Self::reserve_native(service, source_owner, images, activation, FOREIGN_BYTES)
    }
    pub(in crate::vllm_cuda_qualification) fn reserve_short_hf(
        service: &Arc<ResidentService>,
        source_owner: CompletionOwner,
        images: Vec<CubinImage>,
        activation: Option<CubinActivationImage>,
        profile: zerogravity_core::accelerator::program::FullModelProfile,
        cohort_b4: bool,
    ) -> Result<Self> {
        let bytes = super::super::short_hf_native::short_hf_native_provider_bytes_for_cohort(
            profile, cohort_b4,
        );
        Self::reserve_native(service, source_owner, images, activation, bytes)
    }
    fn reserve_native(
        service: &Arc<ResidentService>,
        source_owner: CompletionOwner,
        images: Vec<CubinImage>,
        activation: Option<CubinActivationImage>,
        bytes: usize,
    ) -> Result<Self> {
        let charge = service.reserve_foreign_provider(bytes)?;
        if charge.bytes() != bytes {
            return Err(Error::State(
                "original exact native module reservation".into(),
            ));
        }
        let mut payloads = Vec::with_capacity(images.len());
        let mut admissions = Vec::with_capacity(images.len());
        for image in images {
            payloads.push((image.kind, image.bytes));
            admissions.push(image.admission);
        }
        let activation = activation.map(|image| {
            admissions.push(image.admission);
            image.bytes
        });
        let owner: CompletionOwner = Arc::new(NativeAdmission {
            _source_owner: source_owner,
            _image_admissions: admissions,
            _native_charge: charge,
            expected_native_bytes: bytes,
        });
        Ok(Self {
            payloads,
            activation,
            owner,
            queue: service.pinned_cuda_queue()?.clone(),
        })
    }
    pub(in crate::vllm_cuda_qualification) fn completion_owner(&self) -> CompletionOwner {
        self.owner.clone()
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_short_hf_images(
        self,
        profile: zerogravity_core::accelerator::program::FullModelProfile,
        cohort_b4: bool,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let required =
            super::super::short_hf_native::short_hf_required_images_for_cohort(profile, cohort_b4)?;
        if self.activation.is_none()
            || self.payloads.len() != required.len()
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required.into_iter().collect()
        {
            return Err(Error::Malformed(
                "short-HF exact source image union and activation".into(),
            ));
        }
        self.prepare_images_inner(
            sources,
            init,
            cancelled,
            ImageScope::ShortHf(profile, cohort_b4),
        )
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Projection)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_connected_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let required = BTreeSet::from([
            ProjectionImage::Bf16Gemv2TSm86,
            ProjectionImage::Bf16GemvxSm86,
            ProjectionImage::CutlassWmmaSm80,
            ProjectionImage::Bf16SplitKReduceSm86,
            ProjectionImage::AmpereBf16Gemm128x64Sm80,
            ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
        ]);
        if self.activation.is_none()
            || self.payloads.len() != required.len()
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact connected six projection images and activation".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Connected)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_canonical47_mlp_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let required = super::super::canonical47_mlp::required_images();
        if self.activation.is_none()
            || self.payloads.len() != 4
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact original47 four projection images plus activation".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Canonical47Mlp)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_qwen2b_mlp_images(
        self,
        rows: usize,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let required = super::super::qwen2b_mlp::required_images(rows)?;
        if self.activation.is_none()
            || self.payloads.len() != 4
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact2B four projection images plus activation".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Qwen2bMlp)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_canonical47_qkv_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let mut required = super::super::canonical47_mlp::required_images();
        required.insert(ProjectionImage::AttentionAmpere256x128Sm80);
        if self.activation.is_none()
            || self.payloads.len() != 5
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact canonical47 five projection images plus activation".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Canonical47Qkv)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_canonical47_qkkv_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let mut required = super::super::canonical47_mlp::required_images();
        required.insert(ProjectionImage::AttentionAmpere256x128Sm80);
        required.extend([
            ProjectionImage::AttentionQkSm86,
            ProjectionImage::AttentionKvSm86,
        ]);
        if self.activation.is_none()
            || self.payloads.len() != 7
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact canonical47 seven native images plus activation".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Canonical47QkKv)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_canonical47_fa_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let mut required = super::super::canonical47_mlp::required_images();
        required.insert(ProjectionImage::AttentionAmpere256x128Sm80);
        required.extend([
            ProjectionImage::AttentionQkSm86,
            ProjectionImage::AttentionKvSm86,
            ProjectionImage::AttentionFaCausalCombineSm80,
        ]);
        if self.activation.is_none()
            || self.payloads.len() != 8
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact canonical47 eight native images plus activation".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Canonical47Fa)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_canonical47_layer3_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let mut required = super::super::canonical47_mlp::required_images();
        required.insert(ProjectionImage::AttentionAmpere256x128Sm80);
        required.extend([
            ProjectionImage::AttentionQkSm86,
            ProjectionImage::AttentionKvSm86,
            ProjectionImage::AttentionFaCausalCombineSm80,
            ProjectionImage::AttentionSigmoidSm86,
            ProjectionImage::AttentionMultiplySm86,
        ]);
        if self.activation.is_none()
            || self.payloads.len() != 10
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact canonical47 ten native images plus activation".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Canonical47Layer3)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_canonical47_full_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let required = super::super::canonical47_model_native::head::required_images();
        if self.activation.is_none()
            || self.payloads.len() != 11
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact canonical47 eleven native images plus activation".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Canonical47Full)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_canonical47_natural_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let exact = super::super::canonical47_model_native::natural::images::required_images();
        if self.activation.is_none()
            || self.payloads.len() != 14
            || self
                .payloads
                .iter()
                .map(|(k, _)| *k)
                .collect::<BTreeSet<_>>()
                != exact
        {
            return Err(Error::Malformed(
                "canonical natural exact fourteen+activation images".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Canonical47Natural)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_attention_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let required = super::super::attention_cubin::required_images();
        if self.activation.is_some()
            || self.payloads.len() != 9
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact nine captured-attention images".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::Attention)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_attention_gate_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let required = super::super::attention_cubin::required_gate_images();
        if self.activation.is_some()
            || self.payloads.len() != 11
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact eleven computed-attention-gate images".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::AttentionGate)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_vocabulary_head_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        if self.activation.is_some()
            || self.payloads.len() != 1
            || self.payloads[0].0 != ProjectionImage::Bf16Gemv2TSm86
        {
            return Err(Error::Malformed(
                "exact single original vocabulary-head image".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::VocabularyHead)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_source_rotary_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let required = super::super::source_rotary::required_images();
        if self.activation.is_some()
            || self.payloads.len() != 10
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact ten original source rotary images".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::SourceRotary)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare_full_model_images(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedImages>> {
        let required = super::super::full_model_images::required_images();
        if self.activation.is_none()
            || self.payloads.len() != 13
            || self
                .payloads
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<BTreeSet<_>>()
                != required
        {
            return Err(Error::Malformed(
                "exact full24 thirteen projection images plus activation".into(),
            ));
        }
        self.prepare_images_inner(sources, init, cancelled, ImageScope::FullModel)
    }
    fn prepare_images_inner(
        self,
        sources: Arc<Bf16BlasSources>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
        scope: ImageScope,
    ) -> Result<Arc<PreparedImages>> {
        let Self {
            payloads,
            activation,
            owner,
            queue,
        } = self;
        init.retain_completion_owner(owner.clone())?;
        let mut images = BTreeMap::new();
        for (kind, bytes) in payloads {
            not_cancelled(cancelled)?;
            // SAFETY: full original capacity/issuer checked before admission;
            // armed real W initialization retains this SAME composite owner.
            // Each branch enforces its distinct closed source provenance.
            let image = unsafe {
                match kind.provenance() {
                    ProjectionImageSource::Derived(pinned) => {
                        sources.derived_cubin(owner.clone(), pinned, bytes)
                    }
                    ProjectionImageSource::Captured(pinned) => {
                        sources.captured_cubin(owner.clone(), pinned, bytes)
                    }
                }
            }
            .map_err(native_error)?;
            init.retain_completion_owner(image.clone())?;
            images.insert(kind, image);
        }
        let activation = if let Some(bytes) = activation {
            not_cancelled(cancelled)?;
            // SAFETY: same original capacity admission and composite owner;
            // this closed category claims only the actual selected resource.
            let image = unsafe {
                sources.captured_cubin(
                    owner.clone(),
                    cubecl::cuda::PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86,
                    bytes,
                )
            }
            .map_err(native_error)?;
            init.retain_completion_owner(image.clone())?;
            Some(image)
        } else {
            None
        };
        not_cancelled(cancelled)?;
        // SAFETY: same original service charge and identical sealed image owner
        // are retained before native module admission/load on this original actor.
        let set = unsafe {
            match scope {
                ImageScope::ShortHf(profile, true) => queue.admit_short_hf_b4_cubin_set(
                    owner.clone(),
                    profile == zerogravity_core::accelerator::program::FullModelProfile::Qwen2B,
                ),
                ImageScope::ShortHf(profile, false) => queue.admit_short_hf_cubin_set(
                    owner.clone(),
                    profile == zerogravity_core::accelerator::program::FullModelProfile::Qwen2B,
                ),
                ImageScope::SourceRotary => queue.admit_source_rotary_cubin_set(owner.clone()),
                ImageScope::Connected => queue.admit_connected_qwen08_cubin_set(owner.clone()),
                ImageScope::Canonical47Mlp => queue.admit_canonical47_mlp_cubin_set(owner.clone()),
                ImageScope::Qwen2bMlp => queue.admit_qwen2b_mlp_cubin_set(owner.clone()),
                ImageScope::Canonical47Qkv => queue.admit_canonical47_qkv_cubin_set(owner.clone()),
                ImageScope::Canonical47QkKv => {
                    queue.admit_canonical47_qkkv_cubin_set(owner.clone())
                }
                ImageScope::Canonical47Fa => queue.admit_canonical47_fa_cubin_set(owner.clone()),
                ImageScope::Canonical47Layer3 => {
                    queue.admit_canonical47_layer3_cubin_set(owner.clone())
                }
                ImageScope::Canonical47Natural => {
                    queue.admit_canonical47_natural_cubin_set(owner.clone())
                }
                ImageScope::Canonical47Full => {
                    queue.admit_canonical47_full_cubin_set(owner.clone())
                }
                ImageScope::Attention => queue.admit_first_attention_cubin_set(owner.clone()),
                ImageScope::AttentionGate => {
                    queue.admit_first_attention_gate_cubin_set(owner.clone())
                }
                ImageScope::FullModel => queue.admit_full_qwen08_cubin_set(owner.clone()),
                ImageScope::Projection => queue.admit_cubin_set(owner.clone()),
                ImageScope::VocabularyHead => queue.admit_vocabulary_head_cubin_set(owner.clone()),
            }
        }
        .map_err(native_error)?;
        init.retain_completion_owner(set.clone())?;
        let prepared = Arc::new(PreparedImages {
            images,
            activation,
            set,
            owner,
        });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }
    pub(in crate::vllm_cuda_qualification) fn prepare(
        self,
        sources: Arc<Bf16BlasSources>,
        projections: &[Projection],
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ProjectionPrograms>> {
        let prepared = self.prepare_images(sources, init, cancelled)?;
        ProjectionPrograms::prepare(prepared, projections, init, cancelled)
    }
}

impl ProjectionPrograms {
    pub(in crate::vllm_cuda_qualification) fn prepare(
        prepared: Arc<PreparedImages>,
        projections: &[Projection],
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        init.retain_completion_owner(prepared.clone())?;
        let mut programs = BTreeMap::new();
        for &projection in projections {
            let mut kernels = Vec::new();
            for launch in projection.launches() {
                not_cancelled(cancelled)?;
                let image = prepared.image(launch.image())?;
                let bindings = launch.bindings();
                // SAFETY: only the fixed observed template/ABI and original
                // source issuer reach this original module set.
                let kernel = unsafe { prepared.set().prepare(image, launch.into_manifest()) }
                    .map_err(native_error)?;
                init.retain_completion_owner(kernel.clone())?;
                kernels.push(Launch { kernel, bindings });
            }
            programs.insert(projection, kernels);
        }
        let prepared = Arc::new(ProjectionPrograms {
            programs,
            _images: prepared.images.values().cloned().collect(),
            set: prepared.set().clone(),
            _owner: prepared.completion_owner(),
        });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }
}

impl ProjectionPrograms {
    pub(in crate::vllm_cuda_qualification) fn enqueue(
        &self,
        projection: Projection,
        bindings: ProjectionBindings<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        let service = bindings.weights.service();
        if bindings.weights.bytes() != projection.weight_bytes()
            || bindings.input.bytes() != projection.input_bytes()
            || bindings.output.bytes() != projection.output_bytes()
            || [bindings.input, bindings.output].into_iter().any(|lease| {
                !Arc::ptr_eq(service, lease.service())
                    || lease.source_identity() != bindings.weights.source_identity()
            })
        {
            return Err(Error::State(
                "closed original projection source/service/geometry".into(),
            ));
        }
        let launches = self
            .programs
            .get(&projection)
            .ok_or_else(|| Error::State("unprepared closed projection".into()))?;
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
                        .ok_or_else(|| Error::State("original splitK workspace absent".into()))?,
                };
                if !Arc::ptr_eq(service, lease.service())
                    || *binding == ProjectionBinding::Workspace && lease.bytes() != WORKSPACE_BYTES
                {
                    return Err(Error::State(
                        "original full managed workspace/service differs".into(),
                    ));
                }
                handles.push(lease.handle());
            }
            // SAFETY: caller's armed original flight retains this program and
            // all original exact leases/exclusive writes before the first call.
            unsafe { launch.kernel.enqueue(handles) }.map_err(native_error)?;
        }
        Ok(launches.len())
    }
    /// Caller retains self inside an armed real source-W read-only flight.
    pub(in crate::vllm_cuda_qualification) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        // SAFETY: same caller proof covers every original record.
        unsafe { self.retire_kernels_after_physical_completion() }?;
        // SAFETY: all original records retired; caller's armed guard retains
        // every source/image/native owner on partial unload error.
        unsafe { self.set.retire_after_physical_completion() }
    }
    /// Shared-set callers retire every operation's records before one set unload.
    pub(in crate::vllm_cuda_qualification) unsafe fn retire_kernels_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        for launches in self.programs.values() {
            for launch in launches {
                // SAFETY: caller proved every original launch physically
                // completed, and there is no concurrent prepare/enqueue.
                unsafe { launch.kernel.retire_after_physical_completion() }?;
            }
        }
        Ok(())
    }
}

/// Weak-only observations of the actual original native ownership graph.
/// No token here can enqueue, unload, release admission or establish Ready.
pub(in crate::vllm_cuda_qualification) struct NativeRetentionWitness {
    images: Vec<std::sync::Weak<VerifiedCubin>>,
    kernels: Vec<std::sync::Weak<OwnedCudaKernel>>,
    set: std::sync::Weak<OwnedCudaModuleSet>,
    owner: std::sync::Weak<dyn std::any::Any + Send + Sync>,
    charge: std::sync::Weak<ForeignProviderCharge>,
}
/// Immutable metadata read from the actual Weak-upgraded retained image.
#[derive(Clone, Copy)]
pub(in crate::vllm_cuda_qualification) struct NativeImageIdentity {
    pub(in crate::vllm_cuda_qualification) bytes: usize,
    pub(in crate::vllm_cuda_qualification) sha256: [u8; 32],
    pub(in crate::vllm_cuda_qualification) binary_sm: u8,
    pub(in crate::vllm_cuda_qualification) derived_from: Option<cubecl::cuda::PinnedBf16Cubin>,
    pub(in crate::vllm_cuda_qualification) captured_from:
        Option<cubecl::cuda::PinnedCapturedBf16Cubin>,
}
pub(in crate::vllm_cuda_qualification) struct NativeRetentionState {
    pub(in crate::vllm_cuda_qualification) image: NativeImageIdentity,
    pub(in crate::vllm_cuda_qualification) image_owners: usize,
    pub(in crate::vllm_cuda_qualification) kernel_owners: usize,
    pub(in crate::vllm_cuda_qualification) image_admission_bytes: usize,
    pub(in crate::vllm_cuda_qualification) native_charge_bytes: usize,
}
impl ProjectionPrograms {
    pub(in crate::vllm_cuda_qualification) fn retention_witness(
        &self,
    ) -> Result<NativeRetentionWitness> {
        let owner = self
            ._owner
            .downcast_ref::<NativeAdmission>()
            .ok_or_else(|| Error::State("original native composite owner type".into()))?;
        Ok(NativeRetentionWitness {
            images: self._images.iter().map(Arc::downgrade).collect(),
            kernels: self
                .programs
                .values()
                .flatten()
                .map(|launch| Arc::downgrade(&launch.kernel))
                .collect(),
            set: Arc::downgrade(&self.set),
            owner: Arc::downgrade(&self._owner),
            charge: Arc::downgrade(&owner._native_charge),
        })
    }
}
impl NativeRetentionWitness {
    /// Host-only Weak upgrades after every caller-owned strong reference drops.
    pub(in crate::vllm_cuda_qualification) fn inspect(&self) -> Result<NativeRetentionState> {
        let images = self
            .images
            .iter()
            .filter_map(std::sync::Weak::upgrade)
            .collect::<Vec<_>>();
        let kernels = self
            .kernels
            .iter()
            .filter_map(std::sync::Weak::upgrade)
            .collect::<Vec<_>>();
        let set = self.set.upgrade();
        let owner = self.owner.upgrade();
        let charge = self.charge.upgrade();
        if images.len() != self.images.len()
            || kernels.len() != self.kernels.len()
            || images.is_empty()
            || kernels.is_empty()
            || set.is_none()
            || owner.is_none()
            || charge.is_none()
        {
            return Err(Error::State(
                "original native quarantine lost image/kernel/set/admission owner".into(),
            ));
        }
        let owner = owner.ok_or_else(|| Error::State("original native owner absent".into()))?;
        let admission = owner
            .downcast_ref::<NativeAdmission>()
            .ok_or_else(|| Error::State("original retained native owner type".into()))?;
        let charge =
            charge.ok_or_else(|| Error::State("original retained native charge absent".into()))?;
        if !Arc::ptr_eq(&admission._native_charge, &charge)
            || charge.bytes() != admission.expected_native_bytes
        {
            return Err(Error::State(
                "original retained native charge changed".into(),
            ));
        }
        let image = &images[0];
        Ok(NativeRetentionState {
            image: NativeImageIdentity {
                bytes: image.byte_len(),
                sha256: image.sha256(),
                binary_sm: image.binary_sm(),
                derived_from: image.derived_from(),
                captured_from: image.captured_from(),
            },
            image_owners: images.len(),
            kernel_owners: kernels.len(),
            image_admission_bytes: admission
                ._image_admissions
                .iter()
                .map(WindowWorkspace::bytes)
                .sum(),
            native_charge_bytes: charge.bytes(),
        })
    }
}
