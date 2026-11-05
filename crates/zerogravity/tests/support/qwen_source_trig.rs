//! Original operator words and qualification orchestration, outside runtime code.
use super::{
    drive, reference,
    source::{Loaded, need},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek},
    path::Path,
    sync::Arc,
};
use zerogravity::{
    Result,
    gpu::{Device, ResidentModel},
    qwen35::plan::OrdinaryQwenPlan,
    resource_arbiter::ResourceArbiter,
};

#[derive(Deserialize)]
struct Pin {
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
struct Document {
    schema: String,
    status: String,
    torch: String,
    git: String,
    image: String,
    frequencies: [u32; 32],
    positions: usize,
    columns: usize,
    model_execution: bool,
    gpu_execution: bool,
    owner_retired: bool,
    replays_exact: usize,
    files: BTreeMap<String, Pin>,
}
struct Held {
    file: File,
    stamp: [u64; 7],
    data: Vec<u8>,
    hash: String,
}
impl Held {
    fn open(path: &Path, bytes: usize, hash: &str) -> Result<Self> {
        let mut file = reference::open(path)?;
        let stamp = reference::stamp(&file)?;
        need(stamp[2] == bytes as u64, "trig reference exact extent")?;
        let mut data = Vec::with_capacity(bytes);
        (&mut file).take(bytes as u64 + 1).read_to_end(&mut data)?;
        need(
            data.len() == bytes && format!("{:x}", Sha256::digest(&data)) == hash,
            "trig reference digest",
        )?;
        need(
            reference::stamp(&file)? == stamp,
            "trig reference changed while read",
        )?;
        Ok(Self {
            file,
            stamp,
            data,
            hash: hash.to_owned(),
        })
    }
    fn unchanged(&mut self) -> Result<()> {
        need(
            reference::stamp(&self.file)? == self.stamp,
            "trig reference metadata changed",
        )?;
        self.file.rewind()?;
        let mut digest = Sha256::new();
        let mut buffer = [0u8; 8192];
        loop {
            let n = self.file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
        }
        need(
            format!("{:x}", digest.finalize()) == self.hash
                && reference::stamp(&self.file)? == self.stamp,
            "trig reference bytes changed",
        )
    }
    fn word(&self, i: usize) -> u32 {
        u32::from_le_bytes(
            self.data[i * 4..i * 4 + 4]
                .try_into()
                .expect("checked word"),
        )
    }
}

pub fn run(
    loaded: Arc<Loaded>,
    plan: OrdinaryQwenPlan,
    directory: &Path,
    hash: &str,
) -> Result<()> {
    let _host = plan.source().reserve_backend_workspace(16 * 1024 * 1024)?;
    let manifest_path = directory.join("source-trig-reference-r1.json");
    let manifest_size = reference::stamp(&reference::open(&manifest_path)?)?[2] as usize;
    need(manifest_size <= 64 * 1024, "trig manifest bound")?;
    let mut manifest = Held::open(&manifest_path, manifest_size, hash)?;
    let doc: Document = serde_json::from_slice(&manifest.data)
        .map_err(|e| zerogravity::Error::Malformed(format!("trig manifest: {e}")))?;
    need(
        doc.schema == "zg.original-torch-trig.v1"
            && doc.status == "ORIGINAL_OPERATOR_REFERENCE_ONLY"
            && doc.torch == "2.13.0+cu129"
            && doc.git == "cf30153c4c131c8164ee7798e5022d810682e2cb"
            && doc.image
                == "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe"
            && !doc.model_execution
            && !doc.gpu_execution
            && doc.owner_retired
            && doc.replays_exact == 4
            && doc.positions == 32768
            && doc.columns == 32
            && doc.files.len() == 2,
        "original trig reference contract",
    )?;
    let mut files = Vec::new();
    for name in ["sin-torch.bin", "cos-torch.bin"] {
        let pin = doc
            .files
            .get(name)
            .ok_or_else(|| zerogravity::Error::Malformed("trig member absent".into()))?;
        need(
            pin.bytes == doc.positions * doc.columns * 4,
            "trig raw shape",
        )?;
        files.push(Held::open(&directory.join(name), pin.bytes, &pin.sha256)?);
    }
    for (i, word) in doc.frequencies.iter().enumerate() {
        need(
            (1.0f32 / 10_000_000f32.powf((2 * i) as f32 / 64.0)).to_bits() == *word,
            "actual source frequency bits",
        )?;
    }
    drive::block_on(async {
        let inventory = ResidentModel::ordinary_cpu_source_preflight(
            plan.source().clone(),
            plan.plan().clone(),
            2,
        )?;
        let model = ResidentModel::load_ordinary_cpu_source(
            Device::Hip(Default::default()),
            plan.source().clone(),
            loaded.clone(),
            plan.plan().clone(),
            2,
            None,
        )
        .await?;
        let probe = model.ordinary_cpu_source_native_probe().await?;
        let active = probe.observe()?;
        need(
            active.retained && active.pending_events == 0 && active.quarantined_handles == 0,
            "original trig native owner",
        )?;
        let mut words = 0;
        for base in (0..doc.positions).step_by(64) {
            let angles = (base..base + 64)
                .flat_map(|p| {
                    doc.frequencies
                        .iter()
                        .map(move |f| p as f32 * f32::from_bits(*f))
                })
                .collect::<Vec<_>>();
            let actual = model.diagnostic_trig(&angles).await?;
            need(
                actual.len() == angles.len() * 2,
                "physical trig readback extent",
            )?;
            for (i, pair) in actual.chunks_exact(2).enumerate() {
                for (operation, value) in pair.iter().enumerate() {
                    let expected = files[operation].word(base * 32 + i);
                    if value.to_bits() != expected {
                        return Err(zerogravity::Error::State(format!(
                            "original trig position={} column={} operation={} actual={:08x} expected={expected:08x}",
                            base + i / 32,
                            i % 32,
                            operation,
                            value.to_bits()
                        )));
                    }
                    words += 1;
                }
            }
        }
        let zeros = (0..32)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect::<Vec<f32>>();
        let actual = model.diagnostic_trig(&zeros).await?;
        for (input, pair) in zeros.iter().zip(actual.chunks_exact(2)) {
            need(
                pair[0].to_bits() == input.to_bits() && pair[1].to_bits() == 1.0f32.to_bits(),
                "physical source trig signed zero",
            )?;
        }
        drop(actual);
        model.shutdown_ordinary_cpu_source().await?;
        let terminal = probe.observe()?;
        need(
            !terminal.retained
                && terminal.modules == 0
                && terminal.pending_events == 0
                && terminal.quarantined_handles == 0
                && terminal.completion_owners == 0
                && terminal.resident_bytes == 0,
            "original trig native retirement",
        )?;
        let device = ResourceArbiter::process_shared().snapshot(&active.physical)?;
        need(
            device.reserved == inventory.context_device_bytes as u64 && !device.fenced,
            "trig persistent context only",
        )?;
        loaded.verify_after()?;
        manifest.unchanged()?;
        for file in &mut files {
            file.unchanged()?;
        }
        println!(
            "SOURCE_TRIG_NUMERICS angles={} f32_words={words} signed_zero_words=64 mismatches=0 model_forward=false",
            doc.positions * doc.columns
        );
        println!(
            "SOURCE_TRIG_TERMINAL native_retained=false resident_bytes=0 device_context_bytes={} fenced=false",
            device.reserved
        );
        Ok(())
    })
}
