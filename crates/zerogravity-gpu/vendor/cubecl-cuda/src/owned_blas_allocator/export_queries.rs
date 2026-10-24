//! Requested UUID metadata only; no opaque export-table access or forwarding.
#[derive(Clone, Copy)]
pub struct ExportQuery {
    pub uuid: [u8; 16],
    pub calls: u64,
}
const EMPTY: ExportQuery = ExportQuery {
    uuid: [0; 16],
    calls: 0,
};
#[derive(Clone, Copy)]
pub struct ExportQueries {
    entries: [ExportQuery; 16],
    len: usize,
    pub overflow: bool,
    pub null_calls: u64,
}
impl ExportQueries {
    pub(super) const fn new() -> Self {
        Self {
            entries: [EMPTY; 16],
            len: 0,
            overflow: false,
            null_calls: 0,
        }
    }
    pub fn queries(&self) -> &[ExportQuery] {
        &self.entries[..self.len]
    }
    pub(super) fn record(&mut self, uuid: Option<[u8; 16]>) {
        let Some(uuid) = uuid else {
            self.null_calls = self.null_calls.saturating_add(1);
            return;
        };
        let index = if let Some(index) = self.entries[..self.len]
            .iter()
            .position(|item| item.uuid == uuid)
        {
            index
        } else {
            if self.len == 16 {
                self.overflow = true;
                return;
            }
            let index = self.len;
            self.len += 1;
            self.entries[index].uuid = uuid;
            index
        };
        self.entries[index].calls = self.entries[index].calls.saturating_add(1);
    }
}
impl std::fmt::Debug for ExportQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ExportQuery { uuid_bytes: ")?;
        for byte in self.uuid {
            write!(f, "{byte:02x}")?;
        }
        write!(f, ", calls: {} }}", self.calls)
    }
}
impl std::fmt::Debug for ExportQueries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.queries()).finish()
    }
}
