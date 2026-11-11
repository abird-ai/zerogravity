//! Qualified packed-BF16 KV transition for four independent private roots.
//!
//! This is a model-neutral capability primitive, not a selected Qwen path.
//! A model/backend may adopt it only after its own positive serving evidence.
#[cfg(feature = "qualification")]
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result};

const PRIVATE_B4_ROWS: usize = 4;
#[cfg(feature = "qualification")]
const PRIVATE_B4_THREADS: u32 = 32;

/// Model-free packed-BF16 transition for four independent private KV roots.
///
/// Units are BF16 elements at this boundary and packed U32 words only inside
/// the launch. Every candidate bit is written: committed prefix is copied
/// unchanged, the current token/chunk is appended from supplied activation
/// views, and future capacity is zeroed. This owns no allocation, session/root
/// authority, model geometry or publication capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bf16PrivateB4Geometry {
    pub base: usize,
    pub tokens: usize,
    pub capacity: usize,
    pub slab: usize,
    pub value_offset: usize,
    pub packed_row_stride: usize,
}

#[derive(Clone, Copy, Debug)]
struct Bf16PrivateB4Words {
    state: usize,
    key_input: usize,
    packed_input: usize,
    slab: usize,
    value_offset: usize,
    packed_row_stride: usize,
    total: usize,
}

impl Bf16PrivateB4Geometry {
    fn words(self) -> Result<Bf16PrivateB4Words> {
        if self.tokens == 0
            || self.slab == 0
            || !self.slab.is_multiple_of(2)
            || !self.value_offset.is_multiple_of(2)
            || !self.packed_row_stride.is_multiple_of(2)
            || self
                .value_offset
                .checked_add(self.slab)
                .is_none_or(|end| end > self.packed_row_stride)
            || self
                .base
                .checked_add(self.tokens)
                .is_none_or(|end| end > self.capacity)
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
        let key_input = PRIVATE_B4_ROWS
            .checked_mul(append)
            .ok_or_else(|| Error::Limit("packed BF16 private KV key words".into()))?;
        let packed_row_stride = self.packed_row_stride / 2;
        let packed_input = PRIVATE_B4_ROWS
            .checked_mul(self.tokens)
            .and_then(|n| n.checked_mul(packed_row_stride))
            .ok_or_else(|| Error::Limit("packed BF16 private KV packed words".into()))?;
        let total = PRIVATE_B4_ROWS
            .checked_mul(2)
            .and_then(|n| n.checked_mul(state))
            .filter(|&n| n > 0 && u32::try_from(n).is_ok())
            .ok_or_else(|| Error::Limit("packed BF16 private KV launch words".into()))?;
        for value in [
            self.base,
            self.tokens,
            self.capacity,
            slab,
            self.value_offset / 2,
            packed_row_stride,
            state,
        ] {
            if u32::try_from(value).is_err() {
                return Err(Error::Limit(
                    "packed BF16 private KV shader index domain".into(),
                ));
            }
        }
        Ok(Bf16PrivateB4Words {
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

/// Four independent original/candidate root planes. Handles are borrowed only
/// for validation; the prepared value stores checked handle clones, never a
/// lease or publication capability.
#[cfg(feature = "qualification")]
pub(crate) struct Bf16PrivateKvRow<'a> {
    pub prior_key: &'a Handle,
    pub prior_value: &'a Handle,
    pub candidate_key: &'a Handle,
    pub candidate_value: &'a Handle,
}

#[cfg(feature = "qualification")]
pub(crate) struct PreparedBf16PrivateB4 {
    service: cubecl_common::device::ServiceId,
    prior_key: [Handle; PRIVATE_B4_ROWS],
    prior_value: [Handle; PRIVATE_B4_ROWS],
    candidate_key: [Handle; PRIVATE_B4_ROWS],
    candidate_value: [Handle; PRIVATE_B4_ROWS],
    key: Handle,
    packed: Handle,
    status: Handle,
    geometry: Bf16PrivateB4Geometry,
    words: Bf16PrivateB4Words,
}

#[cfg(feature = "qualification")]
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

#[cfg(feature = "qualification")]
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

#[cfg(feature = "qualification")]
impl PreparedBf16PrivateB4 {
    pub(crate) fn prepare(
        client: &Client,
        rows: &[Bf16PrivateKvRow<'_>],
        key: &Handle,
        packed: &Handle,
        status: &Handle,
        geometry: Bf16PrivateB4Geometry,
    ) -> Result<Self> {
        if rows.len() != PRIVATE_B4_ROWS {
            return Err(Error::Unsupported(
                "packed BF16 private KV transition requires four rows".into(),
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

        Ok(Self {
            service: client.service_id(),
            prior_key: rows
                .iter()
                .map(|row| row.prior_key.clone())
                .collect::<Vec<_>>()
                .try_into()
                .expect("four validated rows"),
            prior_value: rows
                .iter()
                .map(|row| row.prior_value.clone())
                .collect::<Vec<_>>()
                .try_into()
                .expect("four validated rows"),
            candidate_key: rows
                .iter()
                .map(|row| row.candidate_key.clone())
                .collect::<Vec<_>>()
                .try_into()
                .expect("four validated rows"),
            candidate_value: rows
                .iter()
                .map(|row| row.candidate_value.clone())
                .collect::<Vec<_>>()
                .try_into()
                .expect("four validated rows"),
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
        let grid = CubeCount::Static((self.words.total as u32).div_ceil(PRIVATE_B4_THREADS), 1, 1);
        let threads = CubeDim::new_1d(PRIVATE_B4_THREADS);
        unsafe {
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
                self.geometry.base as u32,
                self.geometry.tokens as u32,
                self.words.slab as u32,
                self.words.value_offset as u32,
                self.words.packed_row_stride as u32,
                self.words.state as u32,
            );
        }
        Ok(1)
    }
}

#[cfg(feature = "qualification")]
#[cube]
fn bf16_pair_is_nonfinite(word: u32) -> bool {
    let low = word & 0xffffu32;
    let high = word >> 16u32;
    (low & 0x7f80u32) == 0x7f80u32 || (high & 0x7f80u32) == 0x7f80u32
}

#[cfg(feature = "qualification")]
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
    base: u32,
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
        let base_words = usize::cast_from(base) * slab;
        let append_words = usize::cast_from(tokens) * slab;
        let mut value = 0u32;
        if local < base_words {
            if plane == 0usize {
                if request == 0usize {
                    value = prior_k0[local];
                } else if request == 1usize {
                    value = prior_k1[local];
                } else if request == 2usize {
                    value = prior_k2[local];
                } else {
                    value = prior_k3[local];
                }
            } else if request == 0usize {
                value = prior_v0[local];
            } else if request == 1usize {
                value = prior_v1[local];
            } else if request == 2usize {
                value = prior_v2[local];
            } else {
                value = prior_v3[local];
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

#[cfg(feature = "qualification")]
pub(super) fn qualify(client: &Client) -> Result<()> {
    // The production private-root BF16 form is a pure bit transition. Qualify
    // exact prefix/append/future bytes on guarded physical views, including a
    // frontier beyond the legacy 512-token fixture.
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

    for (base, tokens, capacity, inject_nonfinite) in [
        (0usize, 1usize, 64usize, false),
        (7, 1, 64, true),
        (31, 8, 64, false),
        (511, 1, 1024, false),
        (960, 64, 1024, false),
    ] {
        let geometry = Bf16PrivateB4Geometry {
            base,
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
            // One appended BF16 pair carries +inf in its high half. The exact
            // payload still lands in the candidate while certification fails
            // closed through the shared status word.
            key_words[0] = 0x7f80_3f80;
        }
        let packed_words = (0..words.packed_input)
            .map(|i| finite_pair(0x20000 + i))
            .collect::<Vec<_>>();
        let (key_raw, key) = guarded(client, &key_words);
        let (packed_raw, packed) = guarded(client, &packed_words);
        for row in 0..PRIVATE_B4_ROWS {
            let pk = (0..words.state)
                .map(|i| finite_pair((row + 1) * 0x100000 + i))
                .collect::<Vec<_>>();
            let pv = (0..words.state)
                .map(|i| finite_pair((row + 5) * 0x100000 + i))
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
        let rows = (0..PRIVATE_B4_ROWS)
            .map(|row| Bf16PrivateKvRow {
                prior_key: &prior_key[row],
                prior_value: &prior_value[row],
                candidate_key: &candidate_key[row],
                candidate_value: &candidate_value[row],
            })
            .collect::<Vec<_>>();
        PreparedBf16PrivateB4::prepare(client, &rows, &key, &packed, &status, geometry)?
            .launch(client)?;

        for row in 0..PRIVATE_B4_ROWS {
            let actual_key = read_words(client, candidate_key_raw[row].clone(), "candidate key")?;
            let actual_value =
                read_words(client, candidate_value_raw[row].clone(), "candidate value")?;
            if actual_key.first() != Some(&GUARD)
                || actual_key.last() != Some(&GUARD)
                || actual_key[1..actual_key.len() - 1] != expected_key[row]
                || actual_value.first() != Some(&GUARD)
                || actual_value.last() != Some(&GUARD)
                || actual_value[1..actual_value.len() - 1] != expected_value[row]
            {
                return Err(Error::State(format!(
                    "private BF16 B4 KV transition differs base={base} tokens={tokens} row={row}"
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
                "private BF16 B4 KV status/guard differs base={base} tokens={tokens}"
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
                    "private BF16 B4 KV input guard differs role={role}"
                )));
            }
        }
        println!(
            "BF16_PRIVATE_B4_KV_PASS base={base} tokens={tokens} capacity={capacity} \
             exact_words={} guards=true status={expected_status} launches=1",
            words.total,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_is_model_neutral_and_bounded() {
        let words = Bf16PrivateB4Geometry {
            base: 511,
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

        let base = Bf16PrivateB4Geometry {
            base: 0,
            tokens: 1,
            capacity: 512,
            slab: 512,
            value_offset: 4_608,
            packed_row_stride: 5_120,
        };
        for geometry in [
            Bf16PrivateB4Geometry { slab: 511, ..base },
            Bf16PrivateB4Geometry {
                value_offset: 4_609,
                ..base
            },
            Bf16PrivateB4Geometry {
                base: 500,
                tokens: 13,
                ..base
            },
        ] {
            assert!(geometry.words().is_err());
        }
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
