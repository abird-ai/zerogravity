//! Finite source-row head windows; no model/Ready authority.
use super::*;
use crate::owned_command::weight_upload::SourceWeightLayout;

pub(crate) const WINDOW_BYTES: u64 = 1024 * 1024;
pub(crate) const VOCABULARY: usize = 248_320;
// All40 genuine2B Head packets share these exact normalized160B fields.
const ORIGINAL_2B: Option<[u8; 160]> = Some([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 8, 0, 0, 0, 202, 3, 0, 0, 8, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0,
    0, 0, 0,
]);

pub(crate) fn geometry(
    layout: SourceWeightLayout,
) -> Result<(usize, usize), cubecl_core::server::ServerError> {
    match layout {
        SourceWeightLayout::Vocabulary08 => Ok((1024, 512)),
        SourceWeightLayout::Vocabulary2b => Ok((2048, 256)),
        _ => Err(failure("streamed head exact original tied source layout")),
    }
}
pub(crate) fn manifest(
    layout: SourceWeightLayout,
) -> Result<CubinKernelManifest, cubecl_core::server::ServerError> {
    let (hidden, rows) = geometry(layout)?;
    let original = original_packet(layout)?;
    // Only n changes. Width/lda and every other scalar come from its actual
    // original source packet, never from a guessed Params structure.
    if u32::from_le_bytes(original[64..68].try_into().unwrap()) != hidden as u32
        || u32::from_le_bytes(original[68..72].try_into().unwrap()) != VOCABULARY as u32
        || u32::from_le_bytes(original[72..76].try_into().unwrap()) != hidden as u32
    {
        return Err(failure("actual streamed head width/lda/full vocabulary"));
    }
    let mut m = super::manifest();
    let mut packet = original;
    packet[68..72].copy_from_slice(&(rows as u32).to_le_bytes());
    m.grid = [(rows / 8) as u32, 1, 1];
    m.bindings[0].bytes = WINDOW_BYTES;
    m.bindings[1].bytes = (hidden * 2) as u64;
    m.parameters[0].template = packet[..152].to_vec();
    m.parameters[1].template = packet[152..156].to_vec();
    m.parameters[2].template = packet[156..].to_vec();
    m.fixups[0].extent = WINDOW_BYTES;
    m.fixups[1].extent = (hidden * 2) as u64;
    m.fixups[2].extent = (rows * 2) as u64;
    m.fixups[3].extent = (rows * 2) as u64;
    Ok(m)
}
pub(super) fn original_packet(
    layout: SourceWeightLayout,
) -> Result<[u8; 160], cubecl_core::server::ServerError> {
    match layout {
        SourceWeightLayout::Vocabulary08 => Ok(TEMPLATE),
        SourceWeightLayout::Vocabulary2b => {
            ORIGINAL_2B.ok_or_else(|| failure("actual2B whole head packet required"))
        }
        _ => Err(failure("streamed head source layout")),
    }
}
pub(crate) fn require(
    image: &VerifiedCubin,
    m: &CubinKernelManifest,
) -> Result<SourceWeightLayout, cubecl_core::server::ServerError> {
    let pin = IMAGE.derivation();
    if image.derived_from() != Some(IMAGE)
        || image.captured_from().is_some()
        || image.byte_len() != pin.image_bytes
        || image.sha256() != pin.image_sha256
        || image.binary_sm() != pin.binary_sm
    {
        return Err(failure("streamed head same exact original source image"));
    }
    matching_layout(m)
}
fn matching_layout(
    m: &CubinKernelManifest,
) -> Result<SourceWeightLayout, cubecl_core::server::ServerError> {
    for layout in [
        SourceWeightLayout::Vocabulary08,
        SourceWeightLayout::Vocabulary2b,
    ] {
        if manifest(layout).is_ok_and(|expected| expected == *m) {
            return Ok(layout);
        }
    }
    Err(failure(
        "streamed head complete finite packet/launch/bindings",
    ))
}

/// Canonical peer joins ONLY its new complete finite ShortHfModel policy here.
/// Old policies cannot obtain streamed execution by adding a head record.
pub(crate) fn policy_allowed(policy: super::super::policy::CubinSetPolicy) -> bool {
    matches!(
        policy,
        super::super::policy::CubinSetPolicy::Text08
            | super::super::policy::CubinSetPolicy::Text08Cohort4
            | super::super::policy::CubinSetPolicy::Text2B
            | super::super::policy::CubinSetPolicy::Text2BCachedCohort4
    )
}

#[derive(Clone, Copy)]
pub(crate) struct Window {
    pub(crate) layout: SourceWeightLayout,
    pub(crate) input_rows: usize,
    pub(crate) ordinal: usize,
}
impl Window {
    pub(crate) fn validate(self) -> Result<(usize, usize), cubecl_core::server::ServerError> {
        let (hidden, rows) = geometry(self.layout)?;
        if ![1, 47, 57, 62, 67, 72].contains(&self.input_rows) || self.ordinal >= VOCABULARY / rows
        {
            return Err(failure(
                "streamed head finite real input rows/window ordinal",
            ));
        }
        Ok((hidden, rows))
    }
    pub(crate) fn full_bytes(self, index: usize) -> Result<u64, cubecl_core::server::ServerError> {
        let (hidden, _) = self.validate()?;
        match index {
            0 => Ok(WINDOW_BYTES),
            1 => Ok((self.input_rows * hidden * 2) as u64),
            2 => Ok(OUTPUT_BYTES),
            _ => Err(failure("streamed head exact three physical owners")),
        }
    }
    pub(crate) fn output_range(
        self,
    ) -> Result<std::ops::Range<u64>, cubecl_core::server::ServerError> {
        let (_, rows) = self.validate()?;
        let start = (self.ordinal * rows * 2) as u64;
        Ok(start..start + (rows * 2) as u64)
    }
    pub(crate) fn checked_addresses(
        self,
        resources: &[(u64, u64)],
    ) -> Result<[u64; 3], cubecl_core::server::ServerError> {
        let (hidden, _) = self.validate()?;
        if resources.len() != 3 {
            return Err(failure("streamed head full owners"));
        }
        let mut ends = [0; 3];
        for (i, &(ptr, bytes)) in resources.iter().enumerate() {
            if bytes != self.full_bytes(i)? || ptr == 0 || ptr % 16 != 0 {
                return Err(failure("streamed head full native owner extent/alignment"));
            }
            ends[i] = ptr
                .checked_add(bytes)
                .ok_or_else(|| failure("streamed head native end"))?;
        }
        for a in 0..3 {
            for b in a + 1..3 {
                if resources[a].0 < ends[b] && resources[b].0 < ends[a] {
                    return Err(failure("streamed head full physical owners alias"));
                }
            }
        }
        let input = ends[1] - (hidden * 2) as u64;
        let range = self.output_range()?;
        let output = resources[2]
            .0
            .checked_add(range.start)
            .ok_or_else(|| failure("streamed head D start"))?;
        if output
            .checked_add(range.end - range.start)
            .is_none_or(|end| end > ends[2])
        {
            return Err(failure("streamed head D physical subrange"));
        }
        Ok([resources[0].0, input, output])
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Selection {
    Original(VocabularyHeadSelection),
    Streamed(Window),
    Resident(super::resident::Selection),
}
impl Selection {
    pub(crate) fn full_bytes(self, index: usize) -> Result<u64, cubecl_core::server::ServerError> {
        match self {
            Self::Original(s) => s.full_bytes(index),
            Self::Streamed(s) => s.full_bytes(index),
            Self::Resident(s) => s.full_bytes(index),
        }
    }
    pub(crate) fn checked_addresses(
        self,
        ranges: &[(u64, u64)],
    ) -> Result<[u64; 3], cubecl_core::server::ServerError> {
        match self {
            Self::Original(s) => s.checked_addresses(ranges),
            Self::Streamed(s) => s.checked_addresses(ranges),
            Self::Resident(s) => s.checked_addresses(ranges),
        }
    }
    pub(crate) fn require(
        self,
        policy: super::super::policy::CubinSetPolicy,
        image: &VerifiedCubin,
        m: &CubinKernelManifest,
    ) -> Result<(), cubecl_core::server::ServerError> {
        match self {
            Self::Original(s) if super::selection_allowed(policy, s) => super::require(image, m),
            Self::Resident(s) => s.require(policy, image, m),
            Self::Streamed(s) if policy_allowed(policy) => {
                s.validate()?;
                if require(image, m)? != s.layout {
                    return Err(failure("streamed head packet/source-family differs"));
                }
                Ok(())
            }
            _ => Err(failure("selected head exact closed policy")),
        }
    }
}

impl Bf16VocabularyHead {
    pub fn streamed_manifest(
        layout: SourceWeightLayout,
    ) -> Result<CubinKernelManifest, cubecl_core::server::ServerError> {
        manifest(layout)
    }
}

#[cfg(test)]
mod tests;
