//! Fixed metadata only: distinguish function lookup from an actual refused call.
const CAPACITY: usize = 64;
#[derive(Clone, Copy)]
pub struct ResolverEvent {
    pub name: [u8; 64],
    pub lookups: u64,
    pub null_lookups: u64,
    pub stub_lookups: u64,
    pub denied_calls: u64,
    pub forwarded_calls: u64,
}
const EMPTY: ResolverEvent = ResolverEvent {
    name: [0; 64],
    lookups: 0,
    null_lookups: 0,
    stub_lookups: 0,
    denied_calls: 0,
    forwarded_calls: 0,
};
#[derive(Clone, Copy)]
pub struct ResolverEvents {
    entries: [ResolverEvent; CAPACITY],
    len: usize,
    pub overflow: bool,
}
impl ResolverEvents {
    pub(super) const fn new() -> Self {
        Self {
            entries: [EMPTY; CAPACITY],
            len: 0,
            overflow: false,
        }
    }
    pub fn events(&self) -> &[ResolverEvent] {
        &self.entries[..self.len]
    }
    pub(super) fn record(&mut self, name: &[u8], kind: EventKind) {
        if name.is_empty() || name.len() >= 64 {
            self.overflow = true;
            return;
        }
        let index = if let Some(index) = self.entries[..self.len]
            .iter()
            .position(|event| event.name[..name.len()] == *name && event.name[name.len()] == 0)
        {
            index
        } else {
            if self.len == CAPACITY {
                self.overflow = true;
                return;
            }
            let index = self.len;
            self.len += 1;
            self.entries[index].name[..name.len()].copy_from_slice(name);
            index
        };
        let event = &mut self.entries[index];
        match kind {
            EventKind::NullLookup => {
                event.lookups = event.lookups.saturating_add(1);
                event.null_lookups = event.null_lookups.saturating_add(1);
            }
            EventKind::StubLookup => {
                event.lookups = event.lookups.saturating_add(1);
                event.stub_lookups = event.stub_lookups.saturating_add(1);
            }
            EventKind::DeniedCall => event.denied_calls = event.denied_calls.saturating_add(1),
            EventKind::ForwardedCall => {
                event.forwarded_calls = event.forwarded_calls.saturating_add(1)
            }
        }
    }
}
pub(super) enum EventKind {
    NullLookup,
    StubLookup,
    DeniedCall,
    ForwardedCall,
}
impl std::fmt::Debug for ResolverEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.name.iter().position(|&b| b == 0).unwrap_or(64);
        f.debug_struct("ResolverEvent")
            .field("name", &String::from_utf8_lossy(&self.name[..n]))
            .field("lookup", &self.lookups)
            .field("null", &self.null_lookups)
            .field("stub", &self.stub_lookups)
            .field("denied_call", &self.denied_calls)
            .field("forwarded_call", &self.forwarded_calls)
            .finish()
    }
}
impl std::fmt::Debug for ResolverEvents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.events()).finish()
    }
}
