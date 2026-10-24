//! Exact retained cold source alignment, no IO/free-RAM claim/cap increase.
use super::*;
const MIB: usize = 1024 * 1024;
pub(super) const EXTRA_METADATA_BOUND: usize = 128 * MIB;
fn charge(offset: u64, bytes: usize) -> Result<usize> {
    if bytes == 0 || bytes > MIB {
        return Err(Error::Limit(
            "natural small authenticated source range".into(),
        ));
    }
    let start = usize::try_from(offset % (MIB as u64))
        .map_err(|_| Error::Limit("natural source offset".into()))?;
    start
        .checked_add(bytes)
        .and_then(|v| v.checked_add(MIB - 1))
        .map(|v| v / MIB)
        .and_then(|chunks| {
            chunks.checked_mul(MIB + std::mem::size_of::<(usize, std::ops::Range<usize>)>())
        })
        .ok_or_else(|| Error::Limit("natural checked source-chunk charge".into()))
}
pub(super) fn require_host_bound(
    source: &OriginalWindowModelSource,
    records: &[RecordedVllm08Model; 5],
) -> Result<()> {
    if source.source().staging_chunk_bytes() != MIB {
        return Err(Error::Unsupported(
            "natural original immutable1MiB source geometry".into(),
        ));
    }
    let mut bytes = 0usize;
    let mut windows = 0usize;
    let mut add = |name: &str, relative: usize, length: usize| -> Result<()> {
        let tensor = source.source().tensor_info(name)?;
        if relative
            .checked_add(length)
            .is_none_or(|end| end as u64 > tensor.byte_len)
        {
            return Err(Error::State(
                "natural exact source-window header extent".into(),
            ));
        }
        let offset = tensor
            .byte_offset
            .checked_add(relative as u64)
            .ok_or_else(|| Error::Limit("natural source-window offset overflow".into()))?;
        bytes = bytes
            .checked_add(charge(offset, length)?)
            .ok_or_else(|| Error::Limit("natural retained source sum".into()))?;
        windows += 1;
        Ok(())
    };
    for record in records {
        FullModelProgram::require_record(source, record)?;
        for node in record.plan().nodes() {
            if let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation
                && matches!(
                    descriptor.kind(),
                    Vllm08Kind::InputRms | Vllm08Kind::ResidualRms(_)
                )
            {
                let [parameter] = descriptor.parameters() else {
                    return Err(Error::State("natural RMS source arity".into()));
                };
                add(&parameter.name, 0, parameter.bytes)?;
            }
        }
        for id in record.call().request().teachers().ids() {
            add(
                "model.language_model.embed_tokens.weight",
                *id as usize * 2048,
                2048,
            )?;
        }
    }
    for node in records[0].plan().nodes() {
        if let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation
            && matches!(
                descriptor.kind(),
                Vllm08Kind::GdnConv | Vllm08Kind::GdnRecurrent | Vllm08Kind::AttentionQk
            )
        {
            for parameter in descriptor.parameters() {
                add(&parameter.name, 0, parameter.bytes)?;
            }
        }
    }
    if windows != 398 {
        return Err(Error::State(
            "natural exact245gamma69coldrows72GDN12QK windows".into(),
        ));
    }
    let old = 398 * 2 * (MIB + std::mem::size_of::<(usize, std::ops::Range<usize>)>());
    // Same original cap: replace only overly conservative2-chunk-per-window
    // term with source-header-derived exact chunk COVERAGE. Never share reads.
    if bytes
        .checked_add(EXTRA_METADATA_BOUND)
        .is_none_or(|new| new > old)
    {
        return Err(Error::Limit(
            "natural metadata+actual retained cold chunks do not fit unchanged host envelope"
                .into(),
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn charge_counts_complete_chunks_including_crossing_and_offset_metadata() {
        let one = MIB + std::mem::size_of::<(usize, std::ops::Range<usize>)>();
        assert_eq!(charge(0, 2048).unwrap(), one);
        assert_eq!(charge((MIB - 2048) as u64, 2048).unwrap(), one);
        assert_eq!(charge((MIB - 1024) as u64, 2048).unwrap(), 2 * one);
        assert!(charge(0, 0).is_err());
        assert!(charge(0, MIB + 1).is_err());
        assert_eq!(charge((3 * MIB + 16) as u64, 4096).unwrap(), one);
    }
}
