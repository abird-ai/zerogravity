//! Compiled literal packets keyed by actual source workload/call/site.
use super::super::cubin_manifests::ProjectionImage;
use super::super::short_hf_model::ShortHfModelBinding;
use cubecl::cuda::{CubinIo, CubinKernelManifest};
use std::collections::{BTreeMap, BTreeSet};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        FullModelProfile, PlanOperation, ShortHfWorkload, Vllm08Kind, Vllm08Projection,
    },
};

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
    pub(crate) workload: ShortHfWorkload,
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
        (self.workload.prompt_rows(), self.call, self.node).cmp(&(
            other.workload.prompt_rows(),
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
    pub(crate) fn current(scope: &ShortHfModelBinding, node: usize) -> Result<Self> {
        let span = scope.call().span();
        let call = if span.base() == 0 {
            0
        } else {
            span.base()
                .checked_sub(span.workload().prompt_rows())
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| Error::State("short-HF actual packet call ordinal".into()))?
        };
        if call >= 8 || (call == 0) != (span.len() == span.workload().prompt_rows()) {
            return Err(Error::State(
                "short-HF packet key is not a genuine source call".into(),
            ));
        }
        Ok(Self {
            workload: span.workload(),
            call,
            node,
        })
    }
}

pub(crate) fn is_native(kind: Vllm08Kind) -> bool {
    matches!(kind, Vllm08Kind::Projection(p) if p != Vllm08Projection::Head)
        || matches!(
            kind,
            Vllm08Kind::SiluAndMul
                | Vllm08Kind::AttentionQk
                | Vllm08Kind::KvAppend
                | Vllm08Kind::FlashAttention
                | Vllm08Kind::Sigmoid
                | Vllm08Kind::Multiply
        )
}
impl Catalogue {
    pub(crate) fn load(
        profile: FullModelProfile,
        scopes: &[super::ShortHfNativeGeometry],
    ) -> Result<Self> {
        let sites = super::generated::sites(profile)?;
        let mut map = BTreeMap::new();
        for site in sites {
            if site.key.call >= 8
                || site.packets.is_empty()
                || site.packets.len() > 3
                || map.insert(site.key, site).is_some()
            {
                return Err(Error::Malformed(
                    "short-HF duplicate/incomplete literal native site".into(),
                ));
            }
        }
        let mut expected = BTreeSet::new();
        for scope in scopes.iter().filter(|g| g.scope.call().span().base() == 0) {
            for call in 0..8 {
                for (node, n) in scope.scope.record().plan().nodes().iter().enumerate() {
                    let PlanOperation::TypedVllm08 { descriptor, .. } = &n.operation else {
                        continue;
                    };
                    if is_native(descriptor.kind()) {
                        expected.insert(Key {
                            workload: scope.scope.call().span().workload(),
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
                "short-HF complete genuine five-workload/eight-call native packet catalogue unavailable".into()));
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
                        "short-HF exact finite packet role/access/range".into(),
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
            .ok_or_else(|| Error::State("short-HF current native source site missing".into()))
    }
}
