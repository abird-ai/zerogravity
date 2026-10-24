//! Original retained packet; no constructor from caller-selected metadata.
use super::*;

impl PreparedBf16Greedy {
    pub(crate) fn packet_lease(&self) -> Result<&ResidentLease> {
        self.validate_bindings()?;
        Ok(&self.packet)
    }
}
