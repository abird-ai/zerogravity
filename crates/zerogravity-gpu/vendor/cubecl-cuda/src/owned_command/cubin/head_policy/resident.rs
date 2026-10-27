//! Source-exact whole-table projection; no model or publication authority.
use super::*;
use crate::{Bf16MatrixLayout, SourceWeightLayout};
use cubecl_core::server::ServerError;

fn layout(shape: Bf16MatrixLayout) -> Result<SourceWeightLayout, ServerError> {
    if shape.rows() != streamed::VOCABULARY {
        return Err(failure("resident head original vocabulary"));
    }
    match shape.row_bytes() {
        2048 => Ok(SourceWeightLayout::Vocabulary08),
        4096 => Ok(SourceWeightLayout::Vocabulary2b),
        _ => Err(failure("resident head original source width")),
    }
}

pub(crate) fn manifest(shape: Bf16MatrixLayout) -> Result<CubinKernelManifest, ServerError> {
    let layout = layout(shape)?;
    let packet = streamed::original_packet(layout)?;
    let mut m = super::manifest();
    m.bindings[0].bytes = shape.bytes() as u64;
    m.bindings[1].bytes = shape.row_bytes() as u64;
    m.parameters[0].template = packet[..152].to_vec();
    m.parameters[1].template = packet[152..156].to_vec();
    m.parameters[2].template = packet[156..].to_vec();
    m.fixups[0].extent = shape.bytes() as u64;
    m.fixups[1].extent = shape.row_bytes() as u64;
    Ok(m)
}

pub(crate) fn require(
    image: &VerifiedCubin,
    m: &CubinKernelManifest,
    shape: Bf16MatrixLayout,
) -> Result<(), ServerError> {
    let pin = IMAGE.derivation();
    if image.derived_from() != Some(IMAGE)
        || image.captured_from().is_some()
        || image.byte_len() != pin.image_bytes
        || image.sha256() != pin.image_sha256
        || image.binary_sm() != pin.binary_sm
        || *m != manifest(shape)?
    {
        return Err(failure("resident head entire source image/ABI/geometry"));
    }
    Ok(())
}

pub(crate) fn matches_policy(
    policy: super::super::policy::CubinSetPolicy,
    image: &VerifiedCubin,
    m: &CubinKernelManifest,
) -> Result<(), ServerError> {
    use super::super::policy::CubinSetPolicy;
    let columns = match policy {
        CubinSetPolicy::Text08
        | CubinSetPolicy::Text08FreshPrefill
        | CubinSetPolicy::Text08Cohort4 => 1024,
        CubinSetPolicy::Text2B | CubinSetPolicy::Text2BCachedCohort4 => 2048,
        _ => return Err(failure("resident head actual source policy")),
    };
    require(
        image,
        m,
        Bf16MatrixLayout::new(streamed::VOCABULARY, columns)?,
    )
}

#[derive(Clone, Copy)]
pub(crate) struct Selection {
    pub(crate) shape: Bf16MatrixLayout,
    pub(crate) input_rows: usize,
}
impl Selection {
    fn validate(self) -> Result<(), ServerError> {
        let source_layout = layout(self.shape)?;
        let historical = [1, 47, 57, 62, 67, 72].contains(&self.input_rows);
        let fresh = source_layout == SourceWeightLayout::Vocabulary08
            && [48, 49, 50].contains(&self.input_rows);
        if !historical && !fresh {
            return Err(failure("resident head genuine input row geometry"));
        }
        Ok(())
    }
    pub(crate) fn full_bytes(self, index: usize) -> Result<u64, ServerError> {
        self.validate()?;
        match index {
            0 => Ok(self.shape.bytes() as u64),
            1 => Ok((self.input_rows * self.shape.row_bytes()) as u64),
            2 => Ok(OUTPUT_BYTES),
            _ => Err(failure("resident head exact three owners")),
        }
    }
    pub(crate) fn checked_addresses(
        self,
        resources: &[(u64, u64)],
    ) -> Result<[u64; 3], ServerError> {
        self.validate()?;
        if resources.len() != 3 {
            return Err(failure("resident head three actual ranges"));
        }
        let mut ends = [0; 3];
        for (index, &(pointer, bytes)) in resources.iter().enumerate() {
            if pointer == 0 || pointer % 16 != 0 || bytes != self.full_bytes(index)? {
                return Err(failure("resident head actual range/alignment"));
            }
            ends[index] = pointer
                .checked_add(bytes)
                .ok_or_else(|| failure("resident head range overflow"))?;
        }
        for a in 0..3 {
            for b in a + 1..3 {
                if resources[a].0 < ends[b] && resources[b].0 < ends[a] {
                    return Err(failure("resident head physical ranges alias"));
                }
            }
        }
        Ok([
            resources[0].0,
            ends[1] - self.shape.row_bytes() as u64,
            resources[2].0,
        ])
    }
    pub(crate) fn require(
        self,
        policy: super::super::policy::CubinSetPolicy,
        image: &VerifiedCubin,
        m: &CubinKernelManifest,
    ) -> Result<(), ServerError> {
        use super::super::policy::CubinSetPolicy;
        self.validate()?;
        let rows_allowed = match policy {
            CubinSetPolicy::Text08FreshPrefill => [47, 48, 49, 50].contains(&self.input_rows),
            CubinSetPolicy::Text08
            | CubinSetPolicy::Text08Cohort4
            | CubinSetPolicy::Text2B
            | CubinSetPolicy::Text2BCachedCohort4 => {
                [1, 47, 57, 62, 67, 72].contains(&self.input_rows)
            }
            _ => false,
        };
        if !rows_allowed {
            return Err(failure("resident head rows differ from the active text policy"));
        }
        matches_policy(policy, image, m)?;
        require(image, m, self.shape)
    }
}

impl Bf16VocabularyHead {
    pub fn resident_manifest(shape: Bf16MatrixLayout) -> Result<CubinKernelManifest, ServerError> {
        manifest(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn genuine_full_packets_and_all_input_extents_share_one_shape_path() {
        for columns in [1024, 2048] {
            let shape = Bf16MatrixLayout::new(248320, columns).unwrap();
            let m = manifest(shape).unwrap();
            let packet = streamed::original_packet(layout(shape).unwrap()).unwrap();
            assert_eq!(
                m.parameters
                    .iter()
                    .flat_map(|p| p.template.iter().copied())
                    .collect::<Vec<_>>(),
                packet
            );
            assert_eq!(m.bindings[0].bytes, (248320 * columns * 2) as u64);
            assert_eq!(m.grid, [31040, 1, 1]);
            let input_rows = if columns == 1024 {
                vec![1, 47, 48, 49, 50, 57, 62, 67, 72]
            } else {
                vec![1, 47, 57, 62, 67, 72]
            };
            for input_rows in input_rows {
                let s = Selection { shape, input_rows };
                let ranges = [
                    (0x1000_0000, s.full_bytes(0).unwrap()),
                    (0x8000_0000, s.full_bytes(1).unwrap()),
                    (0x9000_0000, OUTPUT_BYTES),
                ];
                let p = s.checked_addresses(&ranges).unwrap();
                assert_eq!(p[1], ranges[1].0 + ((input_rows - 1) * columns * 2) as u64);
                let mut alias = ranges;
                alias[1].0 = alias[0].0 + 16;
                assert!(s.checked_addresses(&alias).is_err());
                let mut wrong = ranges;
                wrong[1].1 -= 2;
                assert!(s.checked_addresses(&wrong).is_err());
            }
        }
        assert!(manifest(Bf16MatrixLayout::new(262144, 2048).unwrap()).is_err());
        assert!(manifest(Bf16MatrixLayout::new(248320, 1536).unwrap()).is_err());
    }
}
