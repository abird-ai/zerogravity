//! Compiled literal packets keyed by actual source workload/call/site.
use super::super::cubin_manifests::ProjectionImage;
use super::TextKernelCapture;
use cubecl::cuda::{CubinIo, CubinKernelManifest};
use std::collections::{BTreeMap, BTreeSet};
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::program::{SourceCall, SourceCallMode, SourceOperatorKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Role {
    W,
    X,
    Y,
    Workspace,
    Packed,
    Q,
    K,
    Gate,
    QGamma,
    KGamma,
    Rotary,
    Positions,
    Cache,
    Slots,
    KScale,
    VScale,
    O,
    Lse,
    CuQ,
    DummyCuK,
    SeqUsed,
    Table,
    PrivateO,
    PartialO,
    PartialLse,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    pub(crate) prompt_tokens: usize,
    pub(crate) call: usize,
    pub(crate) node: usize,
}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.prompt_tokens, self.call, self.node).cmp(&(
            other.prompt_tokens,
            other.call,
            other.node,
        ))
    }
}
pub(crate) struct Packet {
    pub(crate) image: Option<ProjectionImage>,
    pub(crate) roles: Vec<Role>,
    pub(crate) manifest: CubinKernelManifest,
    /// Only source/SASS-proven initialized workspace words, before this packet.
    pub(crate) clears: Vec<(Role, usize, usize)>,
}
pub(crate) struct Site {
    pub(crate) key: Key,
    pub(crate) packets: Vec<Packet>,
    /// Exact cached FA producer-to-public-output copy; source ABI supplies index.
    pub(crate) copy_output: Option<usize>,
}
pub(crate) struct Catalogue {
    sites: BTreeMap<Key, Site>,
}

impl Key {
    pub(crate) fn current(actual: &SourceCall, prompt_tokens: usize, node: usize) -> Result<Self> {
        if !matches!(prompt_tokens, 47 | 57 | 62 | 67 | 72) {
            return Err(Error::Unsupported(
                "native captured prompt length absent".into(),
            ));
        }
        let call = if actual.rows().base() == 0 {
            if actual.mode() != SourceCallMode::Prefill || actual.rows().len() != prompt_tokens {
                return Err(Error::State(
                    "native actual prefill prompt extent differs".into(),
                ));
            }
            0
        } else {
            if actual.mode() != SourceCallMode::Decode || actual.rows().len() != 1 {
                return Err(Error::State("native actual cached input differs".into()));
            }
            actual
                .rows()
                .base()
                .checked_sub(prompt_tokens)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| Error::State("native actual packet call ordinal".into()))?
        };
        if call >= 8 {
            return Err(Error::State(
                "native captured continuation extent exceeded".into(),
            ));
        }
        Ok(Self {
            prompt_tokens,
            call,
            node,
        })
    }
}
pub(crate) fn is_native(kind: SourceOperatorKind) -> bool {
    matches!(
        kind,
        SourceOperatorKind::Projection { selected_row: None }
            | SourceOperatorKind::SiluAndMul { .. }
            | SourceOperatorKind::QkRmsRotaryGated(_)
            | SourceOperatorKind::KvAppend { .. }
            | SourceOperatorKind::FlashAttention { .. }
            | SourceOperatorKind::Sigmoid { .. }
            | SourceOperatorKind::Multiply { .. }
    )
}
impl Catalogue {
    pub(crate) fn load(
        capture: TextKernelCapture,
        scopes: &[super::TextNativeGeometry],
    ) -> Result<Self> {
        let sites = super::generated::sites(capture)?;
        let mut map = BTreeMap::new();
        for site in sites {
            if site.key.call >= 8
                || site.packets.is_empty()
                || site.packets.len() > 3
                || map.insert(site.key, site).is_some()
            {
                return Err(Error::Malformed(
                    "text duplicate/incomplete literal native site".into(),
                ));
            }
        }
        let mut expected = BTreeSet::new();
        for scope in scopes.iter().filter(|g| g.scope.call().rows().base() == 0) {
            for call in 0..8 {
                for node in 0..scope.scope.record().plan().nodes().len() {
                    let Some(descriptor) = &scope.scope.record().operator(node)? else {
                        continue;
                    };
                    if is_native(descriptor.kind()) {
                        expected.insert(Key {
                            prompt_tokens: scope.scope.call().rows().len(),
                            call,
                            node,
                        });
                    }
                }
            }
        }
        if expected.len() != 5 * 8 * 168 || map.keys().copied().collect::<BTreeSet<_>>() != expected
        {
            return Err(Error::Unsupported(
                "text complete genuine five-workload/eight-call native packet catalogue unavailable".into()));
        }
        for site in map.values() {
            for packet in &site.packets {
                if packet.roles.len() != packet.manifest.bindings.len()
                    || packet.roles.is_empty()
                    || packet.roles.iter().copied().collect::<BTreeSet<_>>().len()
                        != packet.roles.len()
                    || packet.manifest.bindings.iter().any(|b| b.bytes == 0)
                    || packet.clears.iter().any(|(role, offset, n)| {
                        *n == 0
                            || !packet
                                .roles
                                .iter()
                                .zip(&packet.manifest.bindings)
                                .any(|(r, b)| {
                                    r == role
                                        && b.io != CubinIo::Read
                                        && offset
                                            .checked_add(*n)
                                            .is_some_and(|end| end as u64 <= b.bytes)
                                })
                    })
                {
                    return Err(Error::Malformed(
                        "text exact finite packet role/access/range".into(),
                    ));
                }
            }
        }
        Ok(Self { sites: map })
    }
    pub(crate) fn sites(&self) -> impl Iterator<Item = &Site> {
        self.sites.values()
    }
    pub(crate) fn site(&self, key: Key) -> Result<&Site> {
        self.sites
            .get(&key)
            .ok_or_else(|| Error::State("text current native source site missing".into()))
    }
}
