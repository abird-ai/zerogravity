//! Qualified provider adapter; shared policy decides from complete live costs.
use zerogravity_core::optimization::{ResidencyChoice, ResidencyPreference};
use zerogravity_core::{Error, Result};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadMode {
    Resident,
    Streamed,
}
impl HeadMode {
    pub(crate) fn resident_available(shape: cubecl::cuda::Bf16MatrixLayout) -> bool {
        // Geometry chooses a pinned source packet; source/owner and physical
        // correctness remain independent checks at the original consumer.
        cubecl::cuda::Bf16VocabularyHead::resident_manifest(shape).is_ok()
    }
    pub(crate) fn preference(selected: Option<&str>) -> Result<ResidencyPreference> {
        match selected {
            None | Some("auto") => Ok(ResidencyPreference::Auto),
            Some("resident") => Ok(ResidencyPreference::RequireResident),
            Some("streamed") => Ok(ResidencyPreference::RequireStreamed),
            _ => Err(Error::Unsupported(
                "text head mode: auto, resident or streamed".into(),
            )),
        }
    }
    pub(crate) fn environment_preference() -> Result<ResidencyPreference> {
        let selected = std::env::var_os("ZG_SHORT_HF_HEAD");
        Self::preference(
            selected
                .as_deref()
                .map(|s| {
                    s.to_str()
                        .ok_or_else(|| Error::State("text head mode must be UTF-8".into()))
                })
                .transpose()?,
        )
    }
    pub(crate) fn from_choice(choice: ResidencyChoice) -> Self {
        match choice {
            ResidencyChoice::Resident => Self::Resident,
            ResidencyChoice::Streamed => Self::Streamed,
        }
    }
    pub(crate) fn resident(self) -> bool {
        self == Self::Resident
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qualified_provider_and_requested_policy_are_separate() {
        for columns in [1024, 2048] {
            assert!(HeadMode::resident_available(
                cubecl::cuda::Bf16MatrixLayout::new(248320, columns).unwrap(),
            ));
        }
        assert!(!HeadMode::resident_available(
            cubecl::cuda::Bf16MatrixLayout::new(262144, 1536).unwrap(),
        ));
        for (name, preference) in [
            (None, ResidencyPreference::Auto),
            (Some("auto"), ResidencyPreference::Auto),
            (Some("resident"), ResidencyPreference::RequireResident),
            (Some("streamed"), ResidencyPreference::RequireStreamed),
        ] {
            assert_eq!(HeadMode::preference(name).unwrap(), preference);
        }
        assert!(HeadMode::preference(Some("unknown")).is_err());
        assert_eq!(
            HeadMode::from_choice(ResidencyChoice::Resident),
            HeadMode::Resident
        );
        assert_eq!(
            HeadMode::from_choice(ResidencyChoice::Streamed),
            HeadMode::Streamed
        );
    }
}
