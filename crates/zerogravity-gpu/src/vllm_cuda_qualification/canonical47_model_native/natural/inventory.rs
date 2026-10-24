//! Test-only exact factory inventory. No image loading, actor or native permit.
use super::*;
use crate::vllm_cuda_qualification::{
    activation_manifest,
    attention_cubin::canonical_natural::{self as attention_packets, Stage},
    canonical47_attention_projection,
    cubin_manifests::Projection,
    mlp_manifests::{MlpOperation, MlpProjection},
};
use cubecl::cuda::{CubinIo, CubinLaunchAttributes};
use sha2::{Digest, Sha256};

// Byte-identical algorithm to vendor attention_policy::digest; test-only copy.
fn digest(m: &CubinKernelManifest) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update((m.entry.len() as u64).to_le_bytes());
    h.update(m.entry.as_bytes());
    h.update([m.binary_sm]);
    for v in m.grid.into_iter().chain(m.block) {
        h.update(v.to_le_bytes());
    }
    h.update(m.dynamic_shared_bytes.to_le_bytes());
    h.update([match m.attributes {
        CubinLaunchAttributes::None => 0,
        CubinLaunchAttributes::Unsupported => 1,
    }]);
    h.update((m.bindings.len() as u64).to_le_bytes());
    for b in &m.bindings {
        h.update(b.bytes.to_le_bytes());
        h.update(b.alignment.to_le_bytes());
        h.update([io(b.io)]);
    }
    h.update((m.parameters.len() as u64).to_le_bytes());
    for p in &m.parameters {
        h.update((p.offset as u64).to_le_bytes());
        h.update((p.template.len() as u64).to_le_bytes());
        h.update(&p.template);
    }
    h.update((m.fixups.len() as u64).to_le_bytes());
    for f in &m.fixups {
        for n in [
            f.parameter as u64,
            f.byte_offset as u64,
            f.binding as u64,
            f.binding_offset,
            f.extent,
            f.alignment,
        ] {
            h.update(n.to_le_bytes());
        }
        h.update([io(f.io)]);
    }
    h.finalize().into()
}
fn io(v: CubinIo) -> u8 {
    match v {
        CubinIo::Read => 0,
        CubinIo::Write => 1,
        CubinIo::ReadWrite => 2,
    }
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 15) as usize] as char);
    }
    out
}
fn quoted(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn array(items: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", items.into_iter().collect::<Vec<_>>().join(","))
}
fn object(fields: &[(&str, String)]) -> String {
    format!(
        "{{{}}}",
        fields
            .iter()
            .map(|(k, v)| format!("{}:{v}", quoted(k)))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn manifest_json(m: &CubinKernelManifest) -> String {
    object(&[
        ("entry", quoted(&m.entry)),
        ("binary_sm", m.binary_sm.to_string()),
        ("grid", array(m.grid.map(|n| n.to_string()))),
        ("block", array(m.block.map(|n| n.to_string()))),
        ("dynamic_shared_bytes", m.dynamic_shared_bytes.to_string()),
        (
            "attributes",
            match m.attributes {
                CubinLaunchAttributes::None => "0",
                CubinLaunchAttributes::Unsupported => "1",
            }
            .into(),
        ),
        (
            "bindings",
            array(m.bindings.iter().map(|b| {
                object(&[
                    ("bytes", b.bytes.to_string()),
                    ("alignment", b.alignment.to_string()),
                    ("io", io(b.io).to_string()),
                ])
            })),
        ),
        (
            "parameters",
            array(m.parameters.iter().map(|p| {
                object(&[
                    ("offset", p.offset.to_string()),
                    ("template_hex", quoted(&hex(&p.template))),
                ])
            })),
        ),
        (
            "fixups",
            array(m.fixups.iter().map(|f| {
                object(&[
                    ("parameter", f.parameter.to_string()),
                    ("byte_offset", f.byte_offset.to_string()),
                    ("binding", f.binding.to_string()),
                    ("binding_offset", f.binding_offset.to_string()),
                    ("extent", f.extent.to_string()),
                    ("alignment", f.alignment.to_string()),
                    ("io", io(f.io).to_string()),
                ])
            })),
        ),
    ])
}
#[derive(Clone)]
struct Record {
    label: String,
    image: Option<ProjectionImage>,
    manifest: CubinKernelManifest,
}
impl Record {
    fn image_sha(&self) -> [u8; 32] {
        self.image
            .map_or(activation_manifest::IMAGE_SHA256, ProjectionImage::sha256)
    }
    fn image_bytes(&self) -> usize {
        self.image
            .map_or(activation_manifest::IMAGE_BYTES, ProjectionImage::bytes)
    }
    fn image_kind(&self) -> String {
        self.image
            .map_or_else(|| "Activation".into(), |kind| format!("{kind:?}"))
    }
    fn identity(&self) -> ([u8; 32], [u8; 32]) {
        (self.image_sha(), digest(&self.manifest))
    }
    fn json(&self, index: usize) -> String {
        object(&[
            ("record", index.to_string()),
            ("label", quoted(&self.label)),
            ("image_kind", quoted(&self.image_kind())),
            ("image_bytes", self.image_bytes().to_string()),
            ("image_sha256", quoted(&hex(&self.image_sha()))),
            ("abi_sha256", quoted(&hex(&digest(&self.manifest)))),
            ("manifest", manifest_json(&self.manifest)),
        ])
    }
}
#[derive(Default)]
struct Inventory {
    prepared: Vec<Record>,
    requests: Vec<(String, usize, bool)>,
    attention_reused: BTreeMap<Vec<u8>, usize>,
}
impl Inventory {
    // Every invocation creates an actual new OwnedCudaKernel record, even if a
    // separate preparation already has equal image/entry/ABI metadata.
    fn prepare(
        &mut self,
        label: String,
        image: Option<ProjectionImage>,
        manifest: CubinKernelManifest,
    ) -> usize {
        let expected_sm = image.map_or(86, ProjectionImage::binary_sm);
        assert_eq!(manifest.binary_sm, expected_sm);
        let index = self.prepared.len();
        self.prepared.push(Record {
            label: label.clone(),
            image,
            manifest,
        });
        self.requests.push((label, index, true));
        index
    }
    fn attention(&mut self, label: String, packet: attention_packets::Packet) {
        // SAME key as AttentionStorage::prepare, and a separate map from the
        // earlier projection/MLP/head preparations. No cross-owner dedup added.
        let key = format!("{:?}{:?}", packet.image, packet.manifest).into_bytes();
        if let Some(&index) = self.attention_reused.get(&key) {
            self.requests.push((label, index, false));
        } else {
            let index = self.prepare(label, Some(packet.image), packet.manifest);
            self.attention_reused.insert(key, index);
        }
    }
    fn keys(&self) -> BTreeSet<([u8; 32], [u8; 32])> {
        self.prepared.iter().map(Record::identity).collect()
    }
    fn json(&self) -> String {
        let images = self
            .prepared
            .iter()
            .map(Record::image_sha)
            .collect::<BTreeSet<_>>();
        let functions = self
            .prepared
            .iter()
            .map(|r| (r.image_sha(), r.manifest.entry.clone()))
            .collect::<BTreeSet<_>>();
        let unique = self.keys();
        object(&[
            ("status", quoted("COMPILED_FACTORY_METADATA_ONLY")),
            ("native_qualified", "false".into()),
            ("math_qualified", "false".into()),
            ("actual_prepare_calls", self.prepared.len().to_string()),
            ("factory_requests", self.requests.len().to_string()),
            ("unique_image_count", images.len().to_string()),
            ("unique_image_entry_count", functions.len().to_string()),
            ("unique_image_abi_count", unique.len().to_string()),
            (
                "records",
                array(self.prepared.iter().enumerate().map(|(i, r)| r.json(i))),
            ),
            (
                "requests",
                array(self.requests.iter().map(|(label, index, prepared)| {
                    object(&[
                        ("label", quoted(label)),
                        ("record", index.to_string()),
                        ("prepared", prepared.to_string()),
                    ])
                })),
            ),
            (
                "registry",
                array(unique.iter().map(|(image, abi)| {
                    object(&[
                        ("image_sha256", quoted(&hex(image))),
                        ("abi_sha256", quoted(&hex(abi))),
                    ])
                })),
            ),
        ])
    }
}
fn inventory() -> Result<Inventory> {
    let mut out = Inventory::default();
    // Actual CanonicalNativeStorage::prepare order, including source-bound QKV.
    for operation in Operation::ALL.into_iter().chain([Operation::AttentionQkv]) {
        let records = if operation == Operation::AttentionQkv {
            canonical47_attention_projection::packets_for_root_inventory()
        } else {
            packets(operation)?
        };
        for (i, p) in records.into_iter().enumerate() {
            out.prepare(format!("initial/{operation:?}/{i}"), p.image, p.manifest);
        }
    }
    for projection in [Projection::Qkvz1, Projection::Ba1, Projection::Out1] {
        for (i, p) in projection.launches().into_iter().enumerate() {
            let image = p.image();
            out.prepare(
                format!("one/projection/{projection:?}/{i}"),
                Some(image),
                p.into_manifest(),
            );
        }
    }
    for projection in [MlpProjection::GateUp1, MlpProjection::Down1] {
        for (i, operation) in projection.operations().into_iter().enumerate() {
            let MlpOperation::Kernel(p) = operation else {
                // These N1 factories currently require no workspace clear.
                panic!("new N1 workspace prerequisite must be reviewed before this inventory");
            };
            let image = p.image();
            out.prepare(
                format!("one/mlp/{projection:?}/{i}"),
                Some(image),
                p.into_manifest(),
            );
        }
    }
    out.prepare(
        "one/activation/Rows1".into(),
        None,
        Activation::Rows1.manifest(),
    );
    out.prepare(
        "head".into(),
        Some(ProjectionImage::Bf16Gemv2TSm86),
        Bf16VocabularyHead::manifest(),
    );
    // SAME per-base/per-stage preparation order and key as AttentionStorage.
    for base in std::iter::once(0).chain(47..=53) {
        let stages = if base == 0 {
            vec![
                Stage::Qk,
                Stage::Kv,
                Stage::Fa,
                Stage::Sigmoid,
                Stage::Multiply,
            ]
        } else {
            vec![
                Stage::Qkv,
                Stage::Qk,
                Stage::Kv,
                Stage::Fa,
                Stage::Sigmoid,
                Stage::Multiply,
                Stage::Out,
            ]
        };
        for stage in stages {
            let records = if base == 0 {
                attention_packets::prefill_packets(stage)?
            } else {
                attention_packets::cached_packets(base, stage)?
            };
            for (i, p) in records.into_iter().enumerate() {
                out.attention(format!("attention/{base}/{stage:?}/{i}"), p);
            }
        }
    }
    let images = out
        .prepared
        .iter()
        .filter_map(|r| r.image)
        .collect::<BTreeSet<_>>();
    assert_eq!(images, images::required_images());
    assert!(out.prepared.iter().any(|r| r.image.is_none()));
    Ok(out)
}
#[test]
#[ignore = "ROOT compiled metadata collector only; no native or payload reads"]
fn actual_canonical47_natural_factories_for_root_registry() {
    println!(
        "CANONICAL47_NATURAL_FACTORY_INVENTORY={}",
        inventory().unwrap().json()
    );
}
#[test]
fn same_metadata_preparations_and_actual_reuse_are_distinct() {
    let mut out = Inventory::default();
    let first = attention_packets::cached_packets(47, Stage::Fa)
        .unwrap()
        .remove(0);
    let duplicate = attention_packets::cached_packets(47, Stage::Fa)
        .unwrap()
        .remove(0);
    out.prepare("distinct_owner".into(), Some(first.image), first.manifest);
    out.attention("first_attention".into(), duplicate);
    out.attention(
        "reused_attention".into(),
        attention_packets::cached_packets(47, Stage::Fa)
            .unwrap()
            .remove(0),
    );
    assert_eq!(out.prepared.len(), 2);
    assert_eq!(out.keys().len(), 1);
    assert_eq!(out.requests.len(), 3);
    assert!(!out.requests[2].2);
}
#[test]
fn full_digest_tracks_every_abi_field_family() {
    let m = Bf16VocabularyHead::manifest();
    let expected = digest(&m);
    let mut changed = m.clone();
    changed.entry.push('_');
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.binary_sm ^= 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.grid[0] += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.block[0] += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.dynamic_shared_bytes += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.attributes = CubinLaunchAttributes::Unsupported;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.bindings[0].bytes += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.bindings[0].alignment += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.bindings[0].io = CubinIo::ReadWrite;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.parameters[0].offset += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.parameters[0].template[0] ^= 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.parameters[0].template.push(0);
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.fixups[0].parameter += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.fixups[0].byte_offset += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.fixups[0].binding += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.fixups[0].binding_offset += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.fixups[0].extent += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.fixups[0].alignment += 1;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.fixups[0].io = CubinIo::ReadWrite;
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.fixups.push(changed.fixups[0].clone());
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.bindings.push(changed.bindings[0].clone());
    assert_ne!(digest(&changed), expected);
    changed = m.clone();
    changed.parameters.push(changed.parameters[0].clone());
    assert_ne!(digest(&changed), expected);
}
