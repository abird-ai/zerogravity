//! Authenticated original HF GDN transport; bit widening only, test-only.
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};
use zerogravity_core::{Error, Result};
pub const STAGES: [&str; 20] = [
    "scaled_query",
    "normalized_key",
    "k_beta",
    "v_beta",
    "cum_decay",
    "pairwise_decay",
    "ut_system",
    "intra_chunk_attn",
    "decayed_k_beta",
    "new_values",
    "k_cumdecay",
    "decayed_query",
    "decayed_key",
    "chunk_decay",
    "v_new",
    "inter_chunk_attn",
    "core_f32",
    "final_state",
    "output",
    "returned_state",
];
const BYTES: u64 = 299_795_904;
const SHA256: &str = "0b314ec242346caec8d7f2ed2d0e45647cf8eed37649f97c66f907c5a657a071";
pub struct Case {
    pub rows: usize,
    pub heads: usize,
    pub initialized: bool,
    pub inputs: [Vec<u8>; 6],
    pub expected: [Vec<u8>; 20],
}
struct Header {
    rows: usize,
    heads: usize,
    initialized: bool,
    offset: u64,
}
pub struct Archive {
    raw: File,
    headers: Vec<Header>,
    stamp: std::fs::Metadata,
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Malformed(message.into()))
    }
}
fn u32le(file: &mut File) -> Result<usize> {
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes) as usize)
}
impl Archive {
    pub fn open() -> Result<Self> {
        let path = PathBuf::from(std::env::var("ZG_TORCH_CPU_GDN_TRANSPORT").map_err(|_| {
            Error::Unsupported("independent original GDN transport required".into())
        })?);
        let mut raw = File::open(path)?;
        let stamp = raw.metadata()?;
        require(stamp.len() == BYTES, "original GDN transport extent")?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        let mut total = 0u64;
        loop {
            let n = raw.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            total += n as u64;
            require(total <= BYTES, "growing GDN transport")?;
            hash.update(&buffer[..n]);
        }
        require(
            total == BYTES && format!("{:x}", hash.finalize()) == SHA256,
            "original GDN transport hash/EOF",
        )?;
        raw.rewind()?;
        let mut magic = [0; 8];
        raw.read_exact(&mut magic)?;
        require(
            &magic == b"ZGGD0001" && u32le(&mut raw)? == 19,
            "original GDN transport header",
        )?;
        let mut headers = Vec::with_capacity(19);
        let mut records = 0;
        for _ in 0..19 {
            let rows = u32le(&mut raw)?;
            let heads = u32le(&mut raw)?;
            let initialized = u32le(&mut raw)?;
            require(
                (1..=64).contains(&rows)
                    && heads > 0
                    && heads <= 48
                    && heads.is_multiple_of(16)
                    && initialized <= 1,
                "GDN original operator geometry",
            )?;
            let offset = raw.stream_position()?;
            let inputs = input_words(rows, heads);
            for words in inputs[..if initialized == 1 { 6 } else { 5 }]
                .iter()
                .copied()
                .chain(output_words(rows, heads))
            {
                require(u32le(&mut raw)? == words, "GDN transported field extent")?;
                let end = raw
                    .stream_position()?
                    .checked_add((words * 4) as u64)
                    .ok_or_else(|| Error::Malformed("GDN transport overflow".into()))?;
                require(end <= BYTES, "GDN bounded field")?;
                raw.seek(SeekFrom::Start(end))?;
                records += 1;
            }
            headers.push(Header {
                rows,
                heads,
                initialized: initialized == 1,
                offset,
            });
        }
        require(
            records == 484 && raw.stream_position()? == BYTES,
            "complete GDN original transport",
        )?;
        let out = Self {
            raw,
            headers,
            stamp,
        };
        out.require_stable()?;
        Ok(out)
    }
    pub fn len(&self) -> usize {
        self.headers.len()
    }
    pub fn heads(&self) -> impl Iterator<Item = usize> + '_ {
        self.headers.iter().map(|h| h.heads)
    }
    pub fn case(&mut self, index: usize) -> Result<Case> {
        let h = &self.headers[index];
        let (rows, heads, initialized) = (h.rows, h.heads, h.initialized);
        self.raw.seek(SeekFrom::Start(h.offset))?;
        let mut take = |words: usize| -> Result<Vec<u8>> {
            require(
                u32le(&mut self.raw)? == words && words <= 4 * 1024 * 1024,
                "bounded GDN field read",
            )?;
            let mut bytes = vec![0; words * 4];
            self.raw.read_exact(&mut bytes)?;
            Ok(bytes)
        };
        let sizes = input_words(rows, heads);
        let mut inputs = [
            take(sizes[0])?,
            take(sizes[1])?,
            take(sizes[2])?,
            take(sizes[3])?,
            take(sizes[4])?,
            Vec::new(),
        ];
        // Mathematical empty state, explicitly not an original populated record.
        inputs[5] = if initialized {
            take(sizes[5])?
        } else {
            vec![0; sizes[5] * 4]
        };
        let expected = output_words(rows, heads)
            .into_iter()
            .map(take)
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("GDN stage arity".into()))?;
        let case = Case {
            rows,
            heads,
            initialized,
            inputs,
            expected,
        };
        require(
            case.expected[17] == case.expected[19],
            "original returned state identity",
        )?;
        self.require_stable()?;
        Ok(case)
    }
    pub fn require_stable(&self) -> Result<()> {
        use std::os::unix::fs::MetadataExt;
        let after = self.raw.metadata()?;
        let stamp = |m: &std::fs::Metadata| {
            (
                m.dev(),
                m.ino(),
                m.len(),
                m.mtime(),
                m.mtime_nsec(),
                m.ctime(),
                m.ctime_nsec(),
            )
        };
        require(
            stamp(&self.stamp) == stamp(&after),
            "unchanged GDN reference descriptor",
        )
    }
}
fn input_words(rows: usize, heads: usize) -> [usize; 6] {
    let n = rows * heads;
    [n * 128, n * 128, n * 128, n, n, heads * 16384]
}
pub fn output_words(rows: usize, heads: usize) -> [usize; 20] {
    let v = heads * 8192;
    let c = heads * 64;
    let m = heads * 4096;
    let s = heads * 16384;
    [
        v,
        v,
        v,
        v,
        c,
        m,
        m,
        m,
        v,
        v,
        v,
        v,
        v,
        heads,
        v,
        v,
        v,
        s,
        rows * heads * 128,
        s,
    ]
}
