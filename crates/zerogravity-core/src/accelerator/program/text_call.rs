//! Source-bound call geometry; no model, arithmetic or execution permission.
use super::SourceIdentity;
use crate::session_root::RootAttempt;
use crate::{Error, Result};

/// Family adapters derive these limits from their immutable source and selected
/// context policy. Construction validates metadata, not source authenticity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextCallGeometry {
    source: SourceIdentity,
    vocabulary: usize,
    capacity: usize,
}
impl TextCallGeometry {
    pub fn new(source: SourceIdentity, vocabulary: usize, capacity: usize) -> Result<Self> {
        if !source.is_bound()
            || vocabulary == 0
            || vocabulary as u128 > u32::MAX as u128 + 1
            || capacity == 0
            || u64::try_from(capacity).is_err()
        {
            return Err(Error::Malformed(
                "text call source/vocabulary/capacity".into(),
            ));
        }
        Ok(Self {
            source,
            vocabulary,
            capacity,
        })
    }
    pub fn source(self) -> SourceIdentity {
        self.source
    }
    pub fn vocabulary(self) -> usize {
        self.vocabulary
    }
    pub fn capacity(self) -> usize {
        self.capacity
    }
    /// Preserve the original attempt; this neither tests freshness nor grants
    /// state publication. The consuming owner must still check the current root.
    pub fn validate_call(self, attempt: &RootAttempt, ids: &[u32]) -> Result<()> {
        let end = usize::try_from(attempt.frontier())
            .map_err(|_| Error::Limit("text call frontier exceeds host index".into()))?;
        let rows =
            u64::try_from(ids.len()).map_err(|_| Error::Limit("text call row count".into()))?;
        if attempt.base().source() != self.source
            || ids.is_empty()
            || ids.len() > self.capacity
            || end > self.capacity
            || attempt.base().frontier().checked_add(rows) != Some(attempt.frontier())
            || ids.iter().any(|id| *id as usize >= self.vocabulary)
        {
            return Err(Error::State(
                "text call actual source/attempt/IDs/extent".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_root::SessionAuthority;

    #[test]
    fn distinct_source_geometries_keep_exact_bounds_and_original_roots() {
        for (vocabulary, capacity) in [(17, 5), (300_000, 128)] {
            let source = SourceIdentity::for_new_allocation().unwrap();
            let geometry = TextCallGeometry::new(source, vocabulary, capacity).unwrap();
            let authority = SessionAuthority::new(source).unwrap();
            let before = authority.binding();
            let attempt = authority.begin_advance(2).unwrap();
            assert!(
                geometry
                    .validate_call(&attempt, &[0, vocabulary as u32 - 1])
                    .is_ok()
            );
            assert!(
                geometry
                    .validate_call(&attempt, &[0, vocabulary as u32])
                    .is_err()
            );
            assert!(geometry.validate_call(&attempt, &[0]).is_err());
            assert!(geometry.validate_call(&attempt, &[]).is_err());
            let oversized = authority.begin_advance(capacity as u64 + 1).unwrap();
            assert!(
                geometry
                    .validate_call(&oversized, &vec![0; capacity + 1])
                    .is_err()
            );
            let foreign = SessionAuthority::new(SourceIdentity::for_new_allocation().unwrap())
                .unwrap()
                .begin_advance(2)
                .unwrap();
            assert!(geometry.validate_call(&foreign, &[0, 1]).is_err());
            assert_eq!(authority.binding(), before);
        }
    }

    #[test]
    fn geometry_is_metadata_and_does_not_impose_a_model_phase() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let geometry = TextCallGeometry::new(source, 17, 5).unwrap();
        let mut authority = SessionAuthority::new(source).unwrap();
        authority
            .prepare_publish(authority.begin_advance(2).unwrap())
            .unwrap()
            .commit();
        let attempt = authority.begin_advance(3).unwrap();
        assert!(geometry.validate_call(&attempt, &[1, 2, 3]).is_ok());
        let beyond = authority.begin_advance(4).unwrap();
        assert!(geometry.validate_call(&beyond, &[1, 2, 3, 4]).is_err());
        assert!(TextCallGeometry::new(SourceIdentity::unbound(), 17, 5).is_err());
        assert!(TextCallGeometry::new(source, 0, 5).is_err());
        assert!(TextCallGeometry::new(source, 17, 0).is_err());
    }
}
