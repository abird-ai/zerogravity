//! Data-only names for exact compiled kinds. No caller entrypoint/hash/scalar fields.
use super::*;
use crate::qwen35::plan::QwenSourceProfile;
use crate::qwen35_cuda::fixed_prompt::paths;
use native::{
    CubinActivationImage, CubinImage, CubinImageKind as K, CubinQualificationInputs,
    FixedPromptMlpQualificationInputs,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Image {
    kind: String,
    path: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    images: Vec<Image>,
    activation: String,
    rotary: Vec<Image>,
    cublas: String,
    cublas_lt: String,
}
pub(super) struct Bundle {
    root: paths::Directory,
    images: BTreeMap<K, String>,
    rotary: BTreeMap<K, String>,
    activation: String,
    cublas: PathBuf,
    cublas_lt: PathBuf,
}
fn image_map(rows: Vec<Image>, expected: &[K]) -> Result<BTreeMap<K, String>> {
    if rows.len() != expected.len() {
        return Err(Error::Malformed(
            "exact Qwen CUDA native image cohort".into(),
        ));
    }
    let mut result = BTreeMap::new();
    for row in rows {
        zerogravity_artifacts::cache::paths::validate_relative_path(&row.path)?;
        let kind = expected
            .iter()
            .copied()
            .find(|kind| format!("{kind:?}") == row.kind)
            .ok_or_else(|| Error::Unsupported("unregistered Qwen CUDA image kind".into()))?;
        if result.insert(kind, row.path).is_some() {
            return Err(Error::Malformed("duplicate Qwen CUDA native image".into()));
        }
    }
    Ok(result)
}
impl Bundle {
    pub(super) fn open(
        directory: &Path,
        profile: QwenSourceProfile,
        cohort_b4: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        let root = paths::Directory::open(directory)?;
        let raw = root.bounded("cuda-kernels.json", 64 * 1024, cancelled)?;
        let spec: Manifest = serde_json::from_slice(&raw)
            .map_err(|e| Error::Malformed(format!("Qwen CUDA native manifest: {e}")))?;
        if spec.schema != "zerogravity.cuda-kernels.v2" {
            return Err(Error::Unsupported("Qwen CUDA native bundle schema".into()));
        }
        let expected = crate::qwen35::source::binding::ReferenceKernelCapture::for_profile(profile);
        let capture = native::TextKernelCapture::bind(expected.recipe_sha256())?;
        let required = native::text_required_images_for_cohort(capture, cohort_b4)?;
        if required.is_empty() || required.len() > 64 {
            return Err(Error::Limit("Qwen CUDA compiled image cohort".into()));
        }
        if required.iter().copied().collect::<BTreeSet<_>>().len() != required.len()
            || !required.contains(&K::AttentionMultiplySm86)
        {
            return Err(Error::State(
                "Qwen CUDA compiled image inventory differs".into(),
            ));
        }
        let images = image_map(spec.images, &required)?;
        let rotary = image_map(spec.rotary, &ROTARY)?;
        zerogravity_artifacts::cache::paths::validate_relative_path(&spec.activation)?;
        let mut seen = BTreeSet::new();
        for p in images
            .values()
            .chain(rotary.values())
            .chain([&spec.activation])
        {
            if !seen.insert(p) {
                return Err(Error::Malformed(
                    "native images require distinct paths".into(),
                ));
            }
        }
        let provider = |name: String| -> Result<PathBuf> {
            let path = PathBuf::from(&name);
            if path.is_absolute() {
                if !std::fs::symlink_metadata(&path)?.is_file() {
                    return Err(Error::Malformed("native provider regular file".into()));
                }
                Ok(path)
            } else {
                zerogravity_artifacts::cache::paths::validate_relative_path(&name)?;
                drop(root.file(&name)?);
                Ok(directory.join(name))
            }
        };
        let cublas = provider(spec.cublas)?;
        let cublas_lt = provider(spec.cublas_lt)?;
        Ok(Self {
            root,
            images,
            rotary,
            activation: spec.activation,
            cublas,
            cublas_lt,
        })
    }
    fn image(
        &self,
        source: &VerifiedTensorSource,
        kind: K,
        name: &str,
        cap: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<CubinImage> {
        if kind.bytes() > cap {
            return Err(Error::Limit(
                "native image exceeds reviewed capacity".into(),
            ));
        }
        let admission = source.reserve_workspace(cap)?;
        let bytes = self
            .root
            .read(name, kind.bytes(), Some(&kind.sha256()), cancelled)?;
        Ok(CubinImage {
            kind,
            bytes,
            admission,
        })
    }
    pub(super) fn rotary(
        &self,
        source: &VerifiedTensorSource,
        cancelled: Option<&AtomicBool>,
    ) -> Result<CubinQualificationInputs> {
        let mut images = Vec::with_capacity(10);
        for kind in ROTARY {
            images.push(self.image(source, kind, &self.rotary[&kind], kind.bytes(), cancelled)?);
        }
        let k = K::AttentionMultiplySm86;
        images.push(self.image(source, k, &self.images[&k], k.bytes(), cancelled)?);
        Ok(CubinQualificationInputs {
            images,
            cublas: self.cublas.clone(),
            cublas_lt: self.cublas_lt.clone(),
            device_cap: (DEVICE_MIB * MIB) as u64,
        })
    }
    pub(super) fn model(
        &self,
        source: &VerifiedTensorSource,
        cancelled: Option<&AtomicBool>,
    ) -> Result<FixedPromptMlpQualificationInputs> {
        let mut images = Vec::with_capacity(self.images.len());
        for (&kind, name) in &self.images {
            images.push(self.image(source, kind, name, kind.bytes(), cancelled)?);
        }
        let admission = source.reserve_workspace(901_664)?;
        let bytes = self.root.read(
            &self.activation,
            901_664,
            Some(&paths::digest(ACTIVATION_SHA)),
            cancelled,
        )?;
        Ok(FixedPromptMlpQualificationInputs {
            projections: CubinQualificationInputs {
                images,
                cublas: self.cublas.clone(),
                cublas_lt: self.cublas_lt.clone(),
                device_cap: (DEVICE_MIB * MIB) as u64,
            },
            activation: Some(CubinActivationImage { bytes, admission }),
        })
    }
}

const ACTIVATION_SHA: &str = "7ec9414283b09942de90011eb5bdbaeb63b352cd0dc500f2c7ba3854f660fadf";
const ROTARY: [K; 9] = [
    K::RotaryArangeSm86,
    K::RotaryDivideSm86,
    K::RotaryFillSm86,
    K::RotaryPowSm86,
    K::RotaryReciprocalSm86,
    K::RotaryCosSm86,
    K::RotarySinSm86,
    K::RotaryCatSm86,
    K::RotaryBf16CastSm86,
];
