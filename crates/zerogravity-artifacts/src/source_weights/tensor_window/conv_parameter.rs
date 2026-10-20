//! Sealed original BF16 depthwise four-tap parameter transport only.
use super::*;
impl VerifiedTensorSource {
    /// Original rank3 metadata and raw elements, without model eligibility.
    /// `parameter()` remains rank1-only; no synthetic flattened tensor exists.
    pub fn conv4_parameter(&self, name: &str) -> Result<VerifiedParameter> {
        let location = self
            .0
            .tensors
            .get(name)
            .ok_or_else(|| Error::Malformed("original conv4 parameter missing".into()))?;
        let [channels, 1, 4] = location.info.shape.as_slice() else {
            return Err(Error::Unsupported("original conv4 raw rank/shape".into()));
        };
        let channels = usize::try_from(*channels)
            .ok()
            .filter(|n| *n > 0 && *n <= 16384 && n.is_multiple_of(32))
            .ok_or_else(|| Error::Limit("original conv4 channel bound".into()))?;
        let elements = channels
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("original conv4 element count".into()))?;
        let bytes = elements
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("original conv4 byte count".into()))?;
        if location.info.kind != QuantKind::Bf16 || location.info.byte_len != bytes as u64 {
            return Err(Error::Unsupported("original conv4 raw dtype/length".into()));
        }
        Ok(VerifiedParameter {
            location: location.clone(),
            owner: self.0.clone(),
            elements,
            element_bytes: 2,
            bytes,
        })
    }
}
