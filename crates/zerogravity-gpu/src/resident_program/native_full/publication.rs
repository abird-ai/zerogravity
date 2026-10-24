//! Completion-bound access to the actual same-map BF16 sampler packet.
use super::*;

impl FullModelProgram {
    pub(crate) fn publication_packet(&self) -> Result<&ResidentLease> {
        let greedy = self
            .greedy
            .get()
            .ok_or_else(|| Error::State("natural same-map BF16 output consumer absent".into()))?;
        let packet = greedy.packet_lease()?;
        if packet.source_identity() != self.record.call().source()
            || !Arc::ptr_eq(packet.service(), self.values.service())
            || packet.role() != BufferRole::Output
            || packet.bytes() != 12
        {
            return Err(Error::State(
                "natural original BF16 packet binding differs".into(),
            ));
        }
        Ok(packet)
    }

    /// Submission facts only. Original physical completion/Session commit still
    /// own Ready and root publication; an attempted partial traversal refuses.
    pub(crate) fn require_publication_finished(&self) -> Result<()> {
        if !self.launched.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != self.record.plan().nodes().len()
            || !self.output_submitted.load(Ordering::Acquire)
        {
            return Err(Error::State(
                "natural output requires completed Head traversal and actual sampler submission"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Only call inside the ORIGINAL trusted completion callback, on this
    /// body's actual readback order; it accepts no public token/selected ID.
    pub(crate) fn publication_sample(&self, raw: &[cubecl::bytes::Bytes]) -> Result<(u32, u32)> {
        self.require_publication_finished()?;
        self.validate_readbacks(raw)?;
        self.greedy
            .get()
            .ok_or_else(|| Error::State("natural BF16 sampler absent".into()))?
            .validate_readback(&raw[raw.len() - 2..])
    }
}
