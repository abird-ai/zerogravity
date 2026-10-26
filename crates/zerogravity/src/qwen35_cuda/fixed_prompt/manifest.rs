//! Data-only names for exact compiled kinds. No caller entrypoint/hash/scalar fields.
use super::*;
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
            "exact fixed-prompt native image cohort".into(),
        ));
    }
    let mut result = BTreeMap::new();
    for row in rows {
        zerogravity_artifacts::cache::paths::validate_relative_path(&row.path)?;
        let kind = expected
            .iter()
            .copied()
            .find(|kind| format!("{kind:?}") == row.kind)
            .ok_or_else(|| Error::Unsupported("unregistered fixed-prompt image kind".into()))?;
        if result.insert(kind, row.path).is_some() {
            return Err(Error::Malformed(
                "duplicate fixed-prompt native image".into(),
            ));
        }
    }
    Ok(result)
}
impl Bundle {
    pub(super) fn open(directory: &Path, cancelled: Option<&AtomicBool>) -> Result<Self> {
        let root = paths::Directory::open(directory)?;
        let raw = root.bounded("fixed-prompt-kernels.json", 64 * 1024, cancelled)?;
        let spec: Manifest = serde_json::from_slice(&raw)
            .map_err(|e| Error::Malformed(format!("fixed-prompt native manifest: {e}")))?;
        if spec.schema != "zerogravity.cuda-fixed-prompt-kernels.v2" {
            return Err(Error::Unsupported(
                "fixed-prompt native bundle schema".into(),
            ));
        }
        let images = image_map(spec.images, &registry::MODEL)?;
        let rotary = image_map(spec.rotary, &registry::ROTARY)?;
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
        for kind in registry::ROTARY {
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
        let mut images = Vec::with_capacity(14);
        for kind in registry::MODEL {
            images.push(self.image(
                source,
                kind,
                &self.images[&kind],
                registry::image_cap(kind),
                cancelled,
            )?);
        }
        let admission = source.reserve_workspace(2 * MIB)?;
        let bytes = self.root.read(
            &self.activation,
            registry::ACTIVATION_BYTES,
            Some(&paths::digest(registry::ACTIVATION_SHA)),
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
