//! Packed-BF16 KV transition for independent private roots.
//!
//! This is a model-neutral CUDA capability primitive, not a selected model path.
//! It owns no allocation, session/root authority, model geometry or publication.

#[cfg(feature = "cuda-cublas")]
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result};

const PRIVATE_B4_ROWS: usize = 4;
const PRIVATE_B8_ROWS: usize = 8;
#[cfg(feature = "cuda-cublas")]
const PRIVATE_THREADS: u32 = 32;
#[cfg(feature = "cuda-cublas")]
const PRIVATE_B8_THREADS: u32 = 256;

/// Model-free packed-BF16 transition for independent private KV roots.
///
/// Units are BF16 elements at this boundary and packed U32 words only inside
/// the launch. Every candidate bit is written: each committed prefix is copied
/// unchanged, the current token/chunk is appended from supplied activation
/// views, and each future candidate tail is zeroed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bf16PrivateKvGeometry<const ROWS: usize> {
    pub bases: [usize; ROWS],
    pub tokens: usize,
    pub capacity: usize,
    pub slab: usize,
    pub value_offset: usize,
    pub packed_row_stride: usize,
}

#[derive(Clone, Copy, Debug)]
struct Bf16PrivateKvWords {
    state: usize,
    key_input: usize,
    packed_input: usize,
    slab: usize,
    value_offset: usize,
    packed_row_stride: usize,
    total: usize,
}

impl<const ROWS: usize> Bf16PrivateKvGeometry<ROWS> {
    fn words(self) -> Result<Bf16PrivateKvWords> {
        if !matches!(ROWS, PRIVATE_B4_ROWS | PRIVATE_B8_ROWS)
            || self.tokens == 0
            || self.slab == 0
            || !self.slab.is_multiple_of(2)
            || !self.value_offset.is_multiple_of(2)
            || !self.packed_row_stride.is_multiple_of(2)
            || self
                .value_offset
                .checked_add(self.slab)
                .is_none_or(|end| end > self.packed_row_stride)
            || self.bases.iter().any(|base| {
                base.checked_add(self.tokens)
                    .is_none_or(|end| end > self.capacity)
            })
        {
            return Err(Error::Malformed(
                "packed BF16 private KV transition geometry".into(),
            ));
        }
        let slab = self.slab / 2;
        let state = self
            .capacity
            .checked_mul(slab)
            .ok_or_else(|| Error::Limit("packed BF16 private KV state words".into()))?;
        let append = self
            .tokens
            .checked_mul(slab)
            .ok_or_else(|| Error::Limit("packed BF16 private KV append words".into()))?;
        let key_input = ROWS
            .checked_mul(append)
            .ok_or_else(|| Error::Limit("packed BF16 private KV key words".into()))?;
        let packed_row_stride = self.packed_row_stride / 2;
        let packed_input = ROWS
            .checked_mul(self.tokens)
            .and_then(|n| n.checked_mul(packed_row_stride))
            .ok_or_else(|| Error::Limit("packed BF16 private KV packed words".into()))?;
        let total = ROWS
            .checked_mul(2)
            .and_then(|n| n.checked_mul(state))
            .filter(|&n| n > 0 && u32::try_from(n).is_ok())
            .ok_or_else(|| Error::Limit("packed BF16 private KV launch words".into()))?;
        for value in self.bases.iter().copied().chain([
            self.tokens,
            self.capacity,
            slab,
            self.value_offset / 2,
            packed_row_stride,
            state,
        ]) {
            if u32::try_from(value).is_err() {
                return Err(Error::Limit(
                    "packed BF16 private KV shader index domain".into(),
                ));
            }
        }
        Ok(Bf16PrivateKvWords {
            state,
            key_input,
            packed_input,
            slab,
            value_offset: self.value_offset / 2,
            packed_row_stride,
            total,
        })
    }
}

#[cfg(feature = "cuda-cublas")]
pub(crate) struct Bf16PrivateKvRow<'a> {
    pub prior_key: &'a Handle,
    pub prior_value: &'a Handle,
    pub candidate_key: &'a Handle,
    pub candidate_value: &'a Handle,
}

#[cfg(feature = "cuda-cublas")]
pub(crate) struct PreparedBf16PrivateKv<const ROWS: usize> {
    service: cubecl_common::device::ServiceId,
    prior_key: [Handle; ROWS],
    prior_value: [Handle; ROWS],
    candidate_key: [Handle; ROWS],
    candidate_value: [Handle; ROWS],
    key: Handle,
    packed: Handle,
    status: Handle,
    geometry: Bf16PrivateKvGeometry<ROWS>,
    words: Bf16PrivateKvWords,
}

#[cfg(feature = "cuda-cublas")]
fn exact_words(client: &Client, handle: &Handle, words: usize, role: &str) -> Result<()> {
    let bytes = words
        .checked_mul(4)
        .ok_or_else(|| Error::Limit(format!("{role} byte extent")))?;
    if handle.service != client.service_id() || handle.size_in_used() != bytes as u64 {
        return Err(Error::State(format!(
            "packed BF16 private KV {role} service/extent"
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda-cublas")]
fn reject_overlap(a: &Handle, b: &Handle, role: &str) -> Result<()> {
    if crate::resident_program::BufferRange::new(a)?
        .overlaps(&crate::resident_program::BufferRange::new(b)?)
    {
        return Err(Error::State(format!(
            "packed BF16 private KV aliased {role}"
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda-cublas")]
impl<const ROWS: usize> PreparedBf16PrivateKv<ROWS> {
    pub(crate) fn prepare(
        client: &Client,
        rows: &[Bf16PrivateKvRow<'_>],
        key: &Handle,
        packed: &Handle,
        status: &Handle,
        geometry: Bf16PrivateKvGeometry<ROWS>,
    ) -> Result<Self> {
        if !matches!(ROWS, PRIVATE_B4_ROWS | PRIVATE_B8_ROWS) || rows.len() != ROWS {
            return Err(Error::Unsupported(
                "packed BF16 private KV transition requires four or eight rows".into(),
            ));
        }
        let words = geometry.words()?;
        for row in rows {
            for (handle, role) in [
                (row.prior_key, "prior key"),
                (row.prior_value, "prior value"),
                (row.candidate_key, "candidate key"),
                (row.candidate_value, "candidate value"),
            ] {
                exact_words(client, handle, words.state, role)?;
            }
        }
        exact_words(client, key, words.key_input, "key input")?;
        exact_words(client, packed, words.packed_input, "packed input")?;
        exact_words(client, status, 1, "status")?;

        // Every root plane is private. Read/read activation aliases are allowed,
        // but every candidate and status write remains disjoint from all roots
        // and activation inputs.
        let state = rows
            .iter()
            .flat_map(|row| {
                [
                    row.prior_key,
                    row.prior_value,
                    row.candidate_key,
                    row.candidate_value,
                ]
            })
            .collect::<Vec<_>>();
        for (index, handle) in state.iter().enumerate() {
            for prior in &state[..index] {
                reject_overlap(handle, prior, "private root planes")?;
            }
        }
        for row in rows {
            for candidate in [row.candidate_key, row.candidate_value] {
                reject_overlap(candidate, key, "candidate/key input")?;
                reject_overlap(candidate, packed, "candidate/packed input")?;
                reject_overlap(candidate, status, "candidate/status")?;
            }
            reject_overlap(status, row.prior_key, "status/prior key")?;
            reject_overlap(status, row.prior_value, "status/prior value")?;
        }
        reject_overlap(status, key, "status/key input")?;
        reject_overlap(status, packed, "status/packed input")?;

        let prior_key = rows
            .iter()
            .map(|row| row.prior_key.clone())
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| Error::State("packed BF16 private KV prior-key coverage".into()))?;
        let prior_value = rows
            .iter()
            .map(|row| row.prior_value.clone())
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| Error::State("packed BF16 private KV prior-value coverage".into()))?;
        let candidate_key = rows
            .iter()
            .map(|row| row.candidate_key.clone())
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| Error::State("packed BF16 private KV candidate-key coverage".into()))?;
        let candidate_value = rows
            .iter()
            .map(|row| row.candidate_value.clone())
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| Error::State("packed BF16 private KV candidate-value coverage".into()))?;
        Ok(Self {
            service: client.service_id(),
            prior_key,
            prior_value,
            candidate_key,
            candidate_value,
            key: key.clone(),
            packed: packed.clone(),
            status: status.clone(),
            geometry,
            words,
        })
    }

    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
        if client.service_id() != self.service {
            return Err(Error::State(
                "packed BF16 private KV service changed after prepare".into(),
            ));
        }
        let arg =
            |handle: &Handle, words| unsafe { BufferArg::from_raw_parts(handle.clone(), words) };
        let grid = CubeCount::Static((self.words.total as u32).div_ceil(PRIVATE_THREADS), 1, 1);
        let threads = CubeDim::new_1d(PRIVATE_THREADS);
        match ROWS {
            PRIVATE_B4_ROWS => unsafe {
                append_private_b4_bf16::launch_unchecked(
                    client,
                    grid,
                    threads,
                    arg(&self.prior_key[0], self.words.state),
                    arg(&self.prior_key[1], self.words.state),
                    arg(&self.prior_key[2], self.words.state),
                    arg(&self.prior_key[3], self.words.state),
                    arg(&self.prior_value[0], self.words.state),
                    arg(&self.prior_value[1], self.words.state),
                    arg(&self.prior_value[2], self.words.state),
                    arg(&self.prior_value[3], self.words.state),
                    arg(&self.key, self.words.key_input),
                    arg(&self.packed, self.words.packed_input),
                    arg(&self.candidate_key[0], self.words.state),
                    arg(&self.candidate_key[1], self.words.state),
                    arg(&self.candidate_key[2], self.words.state),
                    arg(&self.candidate_key[3], self.words.state),
                    arg(&self.candidate_value[0], self.words.state),
                    arg(&self.candidate_value[1], self.words.state),
                    arg(&self.candidate_value[2], self.words.state),
                    arg(&self.candidate_value[3], self.words.state),
                    arg(&self.status, 1),
                    self.geometry.bases[0] as u32,
                    self.geometry.bases[1] as u32,
                    self.geometry.bases[2] as u32,
                    self.geometry.bases[3] as u32,
                    self.geometry.tokens as u32,
                    self.words.slab as u32,
                    self.words.value_offset as u32,
                    self.words.packed_row_stride as u32,
                    self.words.state as u32,
                );
            },
            PRIVATE_B8_ROWS => unsafe {
                let plane_total = PRIVATE_B8_ROWS
                    .checked_mul(self.words.state)
                    .ok_or_else(|| Error::Limit("packed BF16 private KV B8 plane words".into()))?;
                let plane_grid =
                    CubeCount::Static((plane_total as u32).div_ceil(PRIVATE_B8_THREADS), 1, 1);
                let plane_threads = CubeDim::new_1d(PRIVATE_B8_THREADS);
                append_private_b8_plane_bf16::launch_unchecked(
                    client,
                    plane_grid.clone(),
                    plane_threads,
                    arg(&self.prior_key[0], self.words.state),
                    arg(&self.prior_key[1], self.words.state),
                    arg(&self.prior_key[2], self.words.state),
                    arg(&self.prior_key[3], self.words.state),
                    arg(&self.prior_key[4], self.words.state),
                    arg(&self.prior_key[5], self.words.state),
                    arg(&self.prior_key[6], self.words.state),
                    arg(&self.prior_key[7], self.words.state),
                    arg(&self.key, self.words.key_input),
                    arg(&self.candidate_key[0], self.words.state),
                    arg(&self.candidate_key[1], self.words.state),
                    arg(&self.candidate_key[2], self.words.state),
                    arg(&self.candidate_key[3], self.words.state),
                    arg(&self.candidate_key[4], self.words.state),
                    arg(&self.candidate_key[5], self.words.state),
                    arg(&self.candidate_key[6], self.words.state),
                    arg(&self.candidate_key[7], self.words.state),
                    arg(&self.status, 1),
                    self.geometry.bases[0] as u32,
                    self.geometry.bases[1] as u32,
                    self.geometry.bases[2] as u32,
                    self.geometry.bases[3] as u32,
                    self.geometry.bases[4] as u32,
                    self.geometry.bases[5] as u32,
                    self.geometry.bases[6] as u32,
                    self.geometry.bases[7] as u32,
                    self.geometry.tokens as u32,
                    self.words.slab as u32,
                    0,
                    self.words.slab as u32,
                    self.words.state as u32,
                );
                append_private_b8_plane_bf16::launch_unchecked(
                    client,
                    plane_grid,
                    plane_threads,
                    arg(&self.prior_value[0], self.words.state),
                    arg(&self.prior_value[1], self.words.state),
                    arg(&self.prior_value[2], self.words.state),
                    arg(&self.prior_value[3], self.words.state),
                    arg(&self.prior_value[4], self.words.state),
                    arg(&self.prior_value[5], self.words.state),
                    arg(&self.prior_value[6], self.words.state),
                    arg(&self.prior_value[7], self.words.state),
                    arg(&self.packed, self.words.packed_input),
                    arg(&self.candidate_value[0], self.words.state),
                    arg(&self.candidate_value[1], self.words.state),
                    arg(&self.candidate_value[2], self.words.state),
                    arg(&self.candidate_value[3], self.words.state),
                    arg(&self.candidate_value[4], self.words.state),
                    arg(&self.candidate_value[5], self.words.state),
                    arg(&self.candidate_value[6], self.words.state),
                    arg(&self.candidate_value[7], self.words.state),
                    arg(&self.status, 1),
                    self.geometry.bases[0] as u32,
                    self.geometry.bases[1] as u32,
                    self.geometry.bases[2] as u32,
                    self.geometry.bases[3] as u32,
                    self.geometry.bases[4] as u32,
                    self.geometry.bases[5] as u32,
                    self.geometry.bases[6] as u32,
                    self.geometry.bases[7] as u32,
                    self.geometry.tokens as u32,
                    self.words.slab as u32,
                    self.words.value_offset as u32,
                    self.words.packed_row_stride as u32,
                    self.words.state as u32,
                );
                return Ok(2);
            },
            _ => {
                return Err(Error::Unsupported(
                    "packed BF16 private KV transition width".into(),
                ));
            }
        }
        Ok(1)
    }
}

#[cfg(feature = "cuda-cublas")]
#[cube]
fn bf16_pair_is_nonfinite(word: u32) -> bool {
    let low = word & 0xffffu32;
    let high = word >> 16u32;
    (low & 0x7f80u32) == 0x7f80u32 || (high & 0x7f80u32) == 0x7f80u32
}

#[cfg(feature = "cuda-cublas")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn append_private_b4_bf16(
    prior_k0: &[u32],
    prior_k1: &[u32],
    prior_k2: &[u32],
    prior_k3: &[u32],
    prior_v0: &[u32],
    prior_v1: &[u32],
    prior_v2: &[u32],
    prior_v3: &[u32],
    key: &[u32],
    packed: &[u32],
    candidate_k0: &mut [u32],
    candidate_k1: &mut [u32],
    candidate_k2: &mut [u32],
    candidate_k3: &mut [u32],
    candidate_v0: &mut [u32],
    candidate_v1: &mut [u32],
    candidate_v2: &mut [u32],
    candidate_v3: &mut [u32],
    status: &[Atomic<u32>],
    base0: u32,
    base1: u32,
    base2: u32,
    base3: u32,
    tokens: u32,
    slab_words: u32,
    value_offset_words: u32,
    packed_row_stride_words: u32,
    state_words: u32,
) {
    let at = ABSOLUTE_POS;
    let state = usize::cast_from(state_words);
    if state > 0usize && at < PRIVATE_B4_ROWS * 2usize * state {
        let request = at / (2usize * state);
        let plane = (at / state) % 2usize;
        let local = at % state;
        let slab = usize::cast_from(slab_words);
        let base = if request == 0usize {
            base0
        } else if request == 1usize {
            base1
        } else if request == 2usize {
            base2
        } else {
            base3
        };
        let base_words = usize::cast_from(base) * slab;
        let append_words = usize::cast_from(tokens) * slab;
        let mut value = 0u32;
        if local < base_words {
            if plane == 0usize {
                value = if request == 0usize {
                    prior_k0[local]
                } else if request == 1usize {
                    prior_k1[local]
                } else if request == 2usize {
                    prior_k2[local]
                } else {
                    prior_k3[local]
                };
            } else {
                value = if request == 0usize {
                    prior_v0[local]
                } else if request == 1usize {
                    prior_v1[local]
                } else if request == 2usize {
                    prior_v2[local]
                } else {
                    prior_v3[local]
                };
            }
        } else if local < base_words + append_words {
            let appended = local - base_words;
            let token = appended / slab;
            let lane = appended % slab;
            if plane == 0usize {
                value = key[(request * usize::cast_from(tokens) + token) * slab + lane];
            } else {
                value = packed[(request * usize::cast_from(tokens) + token)
                    * usize::cast_from(packed_row_stride_words)
                    + usize::cast_from(value_offset_words)
                    + lane];
            }
            if bf16_pair_is_nonfinite(value) {
                status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
            }
        }
        if plane == 0usize {
            if request == 0usize {
                candidate_k0[local] = value;
            } else if request == 1usize {
                candidate_k1[local] = value;
            } else if request == 2usize {
                candidate_k2[local] = value;
            } else {
                candidate_k3[local] = value;
            }
        } else if request == 0usize {
            candidate_v0[local] = value;
        } else if request == 1usize {
            candidate_v1[local] = value;
        } else if request == 2usize {
            candidate_v2[local] = value;
        } else {
            candidate_v3[local] = value;
        }
    }
}

#[cfg(feature = "cuda-cublas")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn append_private_b8_plane_bf16(
    prior0: &[u32],
    prior1: &[u32],
    prior2: &[u32],
    prior3: &[u32],
    prior4: &[u32],
    prior5: &[u32],
    prior6: &[u32],
    prior7: &[u32],
    input: &[u32],
    candidate0: &mut [u32],
    candidate1: &mut [u32],
    candidate2: &mut [u32],
    candidate3: &mut [u32],
    candidate4: &mut [u32],
    candidate5: &mut [u32],
    candidate6: &mut [u32],
    candidate7: &mut [u32],
    status: &[Atomic<u32>],
    base0: u32,
    base1: u32,
    base2: u32,
    base3: u32,
    base4: u32,
    base5: u32,
    base6: u32,
    base7: u32,
    tokens: u32,
    slab_words: u32,
    input_offset_words: u32,
    input_row_stride_words: u32,
    state_words: u32,
) {
    let at = ABSOLUTE_POS;
    let state = usize::cast_from(state_words);
    if state > 0usize && at < PRIVATE_B8_ROWS * state {
        let request = at / state;
        let local = at % state;
        let slab = usize::cast_from(slab_words);
        let base = if request == 0usize {
            base0
        } else if request == 1usize {
            base1
        } else if request == 2usize {
            base2
        } else if request == 3usize {
            base3
        } else if request == 4usize {
            base4
        } else if request == 5usize {
            base5
        } else if request == 6usize {
            base6
        } else {
            base7
        };
        let base_words = usize::cast_from(base) * slab;
        let append_words = usize::cast_from(tokens) * slab;
        let mut value = 0u32;
        if local < base_words {
            value = if request == 0usize {
                prior0[local]
            } else if request == 1usize {
                prior1[local]
            } else if request == 2usize {
                prior2[local]
            } else if request == 3usize {
                prior3[local]
            } else if request == 4usize {
                prior4[local]
            } else if request == 5usize {
                prior5[local]
            } else if request == 6usize {
                prior6[local]
            } else {
                prior7[local]
            };
        } else if local < base_words + append_words {
            let appended = local - base_words;
            let token = appended / slab;
            let lane = appended % slab;
            value = input[(request * usize::cast_from(tokens) + token)
                * usize::cast_from(input_row_stride_words)
                + usize::cast_from(input_offset_words)
                + lane];
            if bf16_pair_is_nonfinite(value) {
                status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
            }
        }
        if request == 0usize {
            candidate0[local] = value;
        } else if request == 1usize {
            candidate1[local] = value;
        } else if request == 2usize {
            candidate2[local] = value;
        } else if request == 3usize {
            candidate3[local] = value;
        } else if request == 4usize {
            candidate4[local] = value;
        } else if request == 5usize {
            candidate5[local] = value;
        } else if request == 6usize {
            candidate6[local] = value;
        } else {
            candidate7[local] = value;
        }
    }
}

#[cfg(all(feature = "qualification", feature = "cuda-cublas"))]
fn qualify_case<const ROWS: usize>(
    client: &Client,
    bases: [usize; ROWS],
    tokens: usize,
    capacity: usize,
    inject_nonfinite: bool,
) -> Result<()> {
    const BF16_SLAB: usize = 512;
    const PACKED_STRIDE: usize = 5_120;
    const VALUE_OFFSET: usize = 4_608;
    const GUARD: u32 = 0xd15e_a5e5;

    fn finite_pair(seed: usize) -> u32 {
        let lo = 0x3f00u32 | (seed as u32 & 0x007f);
        let hi = 0x4000u32 | ((seed as u32 >> 7) & 0x007f);
        lo | (hi << 16)
    }
    fn guarded(client: &Client, words: &[u32]) -> (Handle, Handle) {
        let mut storage = Vec::with_capacity(words.len() + 2);
        storage.push(GUARD);
        storage.extend_from_slice(words);
        storage.push(GUARD);
        let raw = client.create_from_slice(u32::as_bytes(&storage));
        let view = raw.clone().offset_start(4).offset_end(4);
        (raw, view)
    }
    fn read_words(client: &Client, handle: Handle, role: &str) -> Result<Vec<u32>> {
        client
            .read_one(handle)
            .map(|bytes| u32::from_bytes(&bytes).to_vec())
            .map_err(|e| Error::State(format!("private BF16 KV {role} readback: {e}")))
    }

    let geometry = Bf16PrivateKvGeometry {
        bases,
        tokens,
        capacity,
        slab: BF16_SLAB,
        value_offset: VALUE_OFFSET,
        packed_row_stride: PACKED_STRIDE,
    };
    let words = geometry.words()?;
    let mut prior_key_raw = Vec::new();
    let mut prior_key = Vec::new();
    let mut prior_value_raw = Vec::new();
    let mut prior_value = Vec::new();
    let mut candidate_key_raw = Vec::new();
    let mut candidate_key = Vec::new();
    let mut candidate_value_raw = Vec::new();
    let mut candidate_value = Vec::new();
    let mut expected_key = Vec::new();
    let mut expected_value = Vec::new();

    let mut key_words = (0..words.key_input)
        .map(|i| finite_pair(0x10000 + i))
        .collect::<Vec<_>>();
    if inject_nonfinite {
        key_words[0] = 0x7f80_3f80;
    }
    let packed_words = (0..words.packed_input)
        .map(|i| finite_pair(0x20000 + i))
        .collect::<Vec<_>>();
    let (key_raw, key) = guarded(client, &key_words);
    let (packed_raw, packed) = guarded(client, &packed_words);

    for (row, base) in bases.iter().copied().enumerate() {
        let pk = (0..words.state)
            .map(|i| finite_pair((row + 1) * 0x100000 + i))
            .collect::<Vec<_>>();
        let pv = (0..words.state)
            .map(|i| finite_pair((row + 9) * 0x100000 + i))
            .collect::<Vec<_>>();
        let (pk_raw, pk_view) = guarded(client, &pk);
        let (pv_raw, pv_view) = guarded(client, &pv);
        let poison = vec![0x7fc0_7fc0u32; words.state];
        let (ck_raw, ck_view) = guarded(client, &poison);
        let (cv_raw, cv_view) = guarded(client, &poison);

        let mut ek = vec![0u32; words.state];
        let mut ev = vec![0u32; words.state];
        let prefix = base * words.slab;
        ek[..prefix].copy_from_slice(&pk[..prefix]);
        ev[..prefix].copy_from_slice(&pv[..prefix]);
        for token in 0..tokens {
            let target = prefix + token * words.slab;
            let key_source = (row * tokens + token) * words.slab;
            let value_source =
                (row * tokens + token) * words.packed_row_stride + words.value_offset;
            ek[target..target + words.slab]
                .copy_from_slice(&key_words[key_source..key_source + words.slab]);
            ev[target..target + words.slab]
                .copy_from_slice(&packed_words[value_source..value_source + words.slab]);
        }
        expected_key.push(ek);
        expected_value.push(ev);
        prior_key_raw.push(pk_raw);
        prior_key.push(pk_view);
        prior_value_raw.push(pv_raw);
        prior_value.push(pv_view);
        candidate_key_raw.push(ck_raw);
        candidate_key.push(ck_view);
        candidate_value_raw.push(cv_raw);
        candidate_value.push(cv_view);
    }

    let (status_raw, status) = guarded(client, &[0]);
    let rows = (0..ROWS)
        .map(|row| Bf16PrivateKvRow {
            prior_key: &prior_key[row],
            prior_value: &prior_value[row],
            candidate_key: &candidate_key[row],
            candidate_value: &candidate_value[row],
        })
        .collect::<Vec<_>>();
    let launches =
        PreparedBf16PrivateKv::<ROWS>::prepare(client, &rows, &key, &packed, &status, geometry)?
            .launch(client)?;

    for row in 0..ROWS {
        let actual_key = read_words(client, candidate_key_raw[row].clone(), "candidate key")?;
        let actual_value = read_words(client, candidate_value_raw[row].clone(), "candidate value")?;
        if actual_key.first() != Some(&GUARD)
            || actual_key.last() != Some(&GUARD)
            || actual_key[1..actual_key.len() - 1] != expected_key[row]
            || actual_value.first() != Some(&GUARD)
            || actual_value.last() != Some(&GUARD)
            || actual_value[1..actual_value.len() - 1] != expected_value[row]
        {
            return Err(Error::State(format!(
                "private BF16 B{ROWS} KV transition differs bases={bases:?} tokens={tokens} row={row}"
            )));
        }
    }

    let status_words = read_words(client, status_raw, "status")?;
    let expected_status = if inject_nonfinite {
        crate::program_kernels::NONFINITE_VALUE
    } else {
        0
    };
    if status_words.as_slice() != [GUARD, expected_status, GUARD] {
        return Err(Error::State(format!(
            "private BF16 B{ROWS} KV status/guard differs bases={bases:?} tokens={tokens}"
        )));
    }
    for (raw, role) in [(&key_raw, "key"), (&packed_raw, "packed")]
        .into_iter()
        .chain(prior_key_raw.iter().map(|h| (h, "prior-key")))
        .chain(prior_value_raw.iter().map(|h| (h, "prior-value")))
    {
        let all = read_words(client, raw.clone(), role)?;
        if all.first() != Some(&GUARD) || all.last() != Some(&GUARD) {
            return Err(Error::State(format!(
                "private BF16 B{ROWS} KV input guard differs role={role}"
            )));
        }
    }
    println!(
        "BF16_PRIVATE_B{ROWS}_KV_PASS bases={bases:?} tokens={tokens} capacity={capacity}          exact_words={} guards=true status={expected_status} launches={launches}",
        words.total,
    );
    Ok(())
}

#[cfg(all(feature = "qualification", feature = "cuda-cublas"))]
pub(super) fn qualify(client: &Client) -> Result<()> {
    for (base, tokens, capacity, inject_nonfinite) in [
        (0usize, 1usize, 64usize, false),
        (7, 1, 64, true),
        (31, 8, 64, false),
        (511, 1, 1024, false),
        (960, 64, 1024, false),
    ] {
        qualify_case(
            client,
            [base; PRIVATE_B4_ROWS],
            tokens,
            capacity,
            inject_nonfinite,
        )?;
    }
    qualify_case(client, [0, 17, 31, 63], 1, 128, false)?;
    qualify_case(client, [0, 1, 7, 17, 31, 63, 95, 127], 1, 256, false)?;
    qualify_case(client, [7; PRIVATE_B8_ROWS], 1, 64, true)?;
    qualify_case(client, [31; PRIVATE_B8_ROWS], 8, 64, false)?;
    qualify_case(client, [511; PRIVATE_B8_ROWS], 1, 1024, false)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_is_model_neutral_and_bounded() {
        let words = Bf16PrivateKvGeometry::<PRIVATE_B4_ROWS> {
            bases: [511; PRIVATE_B4_ROWS],
            tokens: 64,
            capacity: 32_768,
            slab: 512,
            value_offset: 4_608,
            packed_row_stride: 5_120,
        }
        .words()
        .unwrap();
        assert_eq!(words.state, 32_768 * 256);
        assert_eq!(words.key_input, 4 * 64 * 256);
        assert_eq!(words.packed_input, 4 * 64 * 2_560);
        assert_eq!(words.slab, 256);
        assert_eq!(words.value_offset, 2_304);
        assert_eq!(words.packed_row_stride, 2_560);
        assert_eq!(words.total, 4 * 2 * 32_768 * 256);

        let b8 = Bf16PrivateKvGeometry::<PRIVATE_B8_ROWS> {
            bases: [0, 1, 7, 17, 31, 63, 95, 127],
            tokens: 1,
            capacity: 256,
            slab: 512,
            value_offset: 4_608,
            packed_row_stride: 5_120,
        }
        .words()
        .unwrap();
        assert_eq!(b8.state, 256 * 256);
        assert_eq!(b8.key_input, 8 * 256);
        assert_eq!(b8.packed_input, 8 * 2_560);
        assert_eq!(b8.total, 8 * 2 * 256 * 256);

        let base = Bf16PrivateKvGeometry::<PRIVATE_B4_ROWS> {
            bases: [0; PRIVATE_B4_ROWS],
            tokens: 1,
            capacity: 512,
            slab: 512,
            value_offset: 4_608,
            packed_row_stride: 5_120,
        };
        for geometry in [
            Bf16PrivateKvGeometry::<PRIVATE_B4_ROWS> { slab: 511, ..base },
            Bf16PrivateKvGeometry::<PRIVATE_B4_ROWS> {
                value_offset: 4_609,
                ..base
            },
            Bf16PrivateKvGeometry::<PRIVATE_B4_ROWS> {
                bases: [500; PRIVATE_B4_ROWS],
                tokens: 13,
                ..base
            },
        ] {
            assert!(geometry.words().is_err());
        }
        assert!(
            Bf16PrivateKvGeometry::<PRIVATE_B8_ROWS> {
                bases: [0, 1, 2, 3, 4, 5, 6, 512],
                tokens: 1,
                capacity: 512,
                slab: 512,
                value_offset: 4_608,
                packed_row_stride: 5_120,
            }
            .words()
            .is_err()
        );
    }

    #[cfg(all(
        feature = "qualification",
        feature = "cuda-cublas",
        target_os = "linux"
    ))]
    #[test]
    #[ignore = "physical CUDA qualification"]
    fn matches_guarded_bit_oracle_on_cuda() {
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("bf16-private-kv")
            .expect("CUDA diagnostic");
        let service = diagnostic.service.clone();
        qualify(service.client()).unwrap();
        service.client().memory_cleanup().expect("CUDA cleanup");
        assert_eq!(service.resident_bytes().unwrap(), 0);
        service
            .close_model_namespace()
            .expect("close CUDA namespace");
    }
}
