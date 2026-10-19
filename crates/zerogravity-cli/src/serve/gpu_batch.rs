//! Finite, deterministic source-aware GPU request wave formation.
//!
//! A wave is scheduling, NOT a fused physical dispatch. ResidentModel's
//! current one-token workspace is serialized. Later batch-native GPU
//! lowering must consume the same checked members in ONE graph and prove
//! per-request StateRoot/Ready parity before claiming fused batching.

use super::engine::ServiceError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Prefill,
    Decode,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Ready {
    pub source: usize,
    pub phase: Phase,
    pub frontier: usize,
    pub tokens: usize,
    pub scratch_bytes: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Wave {
    pub indices: Vec<usize>,
    pub tokens: usize,
    pub scratch_bytes: usize,
}
pub(super) struct Limits {
    pub max_members: usize,
    pub max_tokens: usize,
    pub max_scratch: usize,
}
/// Every caller's ready work is included exactly once, using first-seen
/// source/phase/frontier grouping and finite quota-split waves. No sorting
/// across requests or silent dropping of a queue tail.
pub(super) fn form(slots: &[Ready], bounds: Limits) -> Result<Vec<Wave>, ServiceError> {
    if slots.is_empty()
        || slots.len() > 64
        || bounds.max_members == 0
        || bounds.max_members > 64
        || bounds.max_tokens == 0
        || bounds.max_tokens > 512
        || bounds.max_scratch == 0
        || bounds.max_scratch > 512 * 1024 * 1024
    {
        return Err(ServiceError::new(
            400,
            "invalid_wave",
            "batch admission bounds",
        ));
    }
    let mut waves = Vec::<Wave>::new();
    let mut heads = Vec::<(usize, Phase, usize)>::new();
    for (index, ready) in slots.iter().enumerate() {
        if ready.frontier > 512
            || (ready.phase == Phase::Prefill && ready.frontier >= 512)
            || ready.tokens == 0
            || ready.tokens > 32
            || ready.scratch_bytes == 0
            || ready.tokens > bounds.max_tokens
            || ready.scratch_bytes > bounds.max_scratch
            || (ready.phase == Phase::Decode && ready.tokens != 1)
        {
            return Err(ServiceError::new(
                400,
                "invalid_wave_member",
                "GPU work is outside source/budget",
            ));
        }
        let key = (ready.source, ready.phase, ready.frontier);
        let mut chosen = None;
        // A compatible wave can accept at most N requests/T tokens/B bytes.
        for (wave_idx, head) in heads.iter().enumerate() {
            let wave = &waves[wave_idx];
            if *head == key
                && wave.indices.len() < bounds.max_members
                && wave
                    .tokens
                    .checked_add(ready.tokens)
                    .is_some_and(|v| v <= bounds.max_tokens)
                && wave
                    .scratch_bytes
                    .checked_add(ready.scratch_bytes)
                    .is_some_and(|v| v <= bounds.max_scratch)
            {
                chosen = Some(wave_idx);
                break;
            }
        }
        if let Some(wave_idx) = chosen {
            let wave = &mut waves[wave_idx];
            wave.indices.push(index);
            wave.tokens += ready.tokens;
            wave.scratch_bytes += ready.scratch_bytes;
        } else {
            heads.push(key);
            waves.push(Wave {
                indices: vec![index],
                tokens: ready.tokens,
                scratch_bytes: ready.scratch_bytes,
            });
        }
    }
    Ok(waves)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn r(source: usize, phase: Phase, frontier: usize, tokens: usize) -> Ready {
        Ready {
            source,
            phase,
            frontier,
            tokens,
            scratch_bytes: 1024,
        }
    }
    #[test]
    fn group_matching_source_phase_frontier_and_preserve_full_coverage() {
        let ready = [
            r(1, Phase::Decode, 4, 1),
            r(2, Phase::Decode, 4, 1),
            r(1, Phase::Decode, 4, 1),
            r(1, Phase::Decode, 5, 1),
            r(1, Phase::Prefill, 4, 16),
            r(2, Phase::Decode, 4, 1),
            r(1, Phase::Decode, 4, 1),
        ];
        let waves = form(
            &ready,
            Limits {
                max_members: 2,
                max_tokens: 32,
                max_scratch: 2048,
            },
        )
        .unwrap();
        assert_eq!(waves[0].indices, [0, 2]);
        assert_eq!(waves[1].indices, [1, 5]);
        assert_eq!(waves[2].indices, [3]);
        assert_eq!(waves[3].indices, [4]);
        assert_eq!(waves[4].indices, [6]);
        let mut seen = waves
            .iter()
            .flat_map(|wave| wave.indices.clone())
            .collect::<Vec<_>>();
        seen.sort_unstable();
        assert_eq!(seen, (0..ready.len()).collect::<Vec<_>>());
        assert_eq!(waves.iter().map(|w| w.tokens).sum::<usize>(), 22);
        assert!(waves.iter().all(|w| w.indices.len() <= 2));
    }
    #[test]
    fn wave_admission_rejects_overflow_unsupported_steps_and_invalid_limits() {
        let valid = r(1, Phase::Decode, 0, 1);
        assert!(
            form(
                &[valid],
                Limits {
                    max_members: 2,
                    max_tokens: 32,
                    max_scratch: 2048
                }
            )
            .is_ok()
        );
        for bad in [
            r(0, Phase::Decode, 0, 2),
            r(0, Phase::Decode, 513, 1),
            r(0, Phase::Prefill, 1, 33),
            r(0, Phase::Prefill, 1, 0),
            Ready {
                scratch_bytes: 0,
                ..valid
            },
        ] {
            assert!(
                form(
                    &[bad],
                    Limits {
                        max_members: 2,
                        max_tokens: 32,
                        max_scratch: 2048
                    }
                )
                .is_err()
            );
        }
        assert!(
            form(
                &[valid],
                Limits {
                    max_members: 0,
                    max_tokens: 32,
                    max_scratch: 2048
                }
            )
            .is_err()
        );
        assert!(
            form(
                &[valid],
                Limits {
                    max_members: 2,
                    max_tokens: 0,
                    max_scratch: 2048
                }
            )
            .is_err()
        );
        assert!(
            form(
                &[valid],
                Limits {
                    max_members: 2,
                    max_tokens: 32,
                    max_scratch: 1
                }
            )
            .is_err()
        );
    }
}
