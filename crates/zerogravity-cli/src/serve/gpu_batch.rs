//! Finite, deterministic source-aware GPU request wave formation.
//!
//! A wave is scheduler membership, not physical completion or memory credit.
//! The GPU worker separately selects qualified native bands or original B1
//! execution and reports actual Ready/dispatch/root receipts.

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

/// Contiguous, unpadded native bands; B1 carries an unmatched survivor.
/// Ceiling 1 is the final prelaunch capacity fallback.
pub(super) fn native_bands(count: usize, ceiling: usize) -> Result<Vec<usize>, ServiceError> {
    if !matches!(ceiling, 1 | 2 | 4 | 8) {
        return Err(ServiceError::new(
            400,
            "invalid_native_bands",
            "native ceiling1/2/4/8",
        ));
    }
    let widths = [8, 4, 2, 1]
        .into_iter()
        .filter(|&n| n <= ceiling)
        .collect::<Vec<_>>();
    qualified_bands(count, &widths)
}

/// Use only the actual prepared provider's finite widths. In particular a
/// qualified B4/B8 cohort does not imply B2; unmatched survivors stay B1.
/// This plans membership only. A submitted/unknown error is never fallback.
pub(super) fn qualified_bands(count: usize, widths: &[usize]) -> Result<Vec<usize>, ServiceError> {
    if count > 64
        || widths.is_empty()
        || widths.len() > 4
        || widths.last() != Some(&1)
        || widths.iter().any(|n| !matches!(n, 1 | 2 | 4 | 8))
        || widths.windows(2).any(|p| p[0] <= p[1])
    {
        return Err(ServiceError::new(
            400,
            "invalid_native_bands",
            "exact descending qualified widths ending in B1",
        ));
    }
    let mut left = count;
    let mut result = Vec::new();
    for &width in widths {
        while left >= width {
            result.push(width);
            left -= width;
        }
    }
    Ok(result)
}

/// BF16/HF source bands never inherit the GGUF B2 qualification.
pub(super) fn bf16_bands(count: usize, ceiling: usize) -> Result<Vec<usize>, ServiceError> {
    if count > 64 || !matches!(ceiling, 1 | 4 | 8) {
        return Err(ServiceError::new(
            400,
            "invalid_bf16_bands",
            "BF16 bands require width8/4/1",
        ));
    }
    let mut remaining = count;
    let mut bands = Vec::new();
    for width in [8, 4, 1] {
        if width <= ceiling {
            while remaining >= width {
                bands.push(width);
                remaining -= width;
            }
        }
    }
    Ok(bands)
}
/// Every caller's ready work is included exactly once, using first-seen
/// source/phase/frontier grouping and finite quota-split waves. No sorting
/// across requests or silent dropping of a queue tail.
pub(super) fn form(slots: &[Ready], bounds: Limits) -> Result<Vec<Wave>, ServiceError> {
    form_with_frontiers(slots, bounds, false, false, 32)
}

/// Atomic source prefill consumes its actual whole quantum; quotas never split
/// a member or turn wave membership into native/completion authority.
/// Caller already validated each source's prompt and execution route.
pub(super) fn form_with_prefill_quantum(
    slots: &[Ready],
    bounds: Limits,
    prefill_quantum: usize,
) -> Result<Vec<Wave>, ServiceError> {
    if !(1..=512).contains(&prefill_quantum)
        || slots.len() > 64
        || slots.iter().any(|row| {
            row.phase == Phase::Prefill
                && row
                    .frontier
                    .checked_add(row.tokens)
                    .is_none_or(|end| end > 512)
        })
    {
        return Err(ServiceError::new(
            400,
            "invalid_wave",
            "atomic prefill quantum/span exceeds bounded context",
        ));
    }
    form_with_frontiers(slots, bounds, false, false, prefill_quantum)
}

/// Whole original prompts retain their atomic quantum; cached requests keep
/// individual frontiers in their row bindings while sharing a fair wave.
pub(super) fn form_cached_with_prefill_quantum(
    slots: &[Ready],
    bounds: Limits,
    prefill_quantum: usize,
) -> Result<Vec<Wave>, ServiceError> {
    // Reuse the existing source/span validation before choosing wider grouping.
    form_with_prefill_quantum(
        slots,
        Limits {
            max_members: bounds.max_members,
            max_tokens: bounds.max_tokens,
            max_scratch: bounds.max_scratch,
        },
        prefill_quantum,
    )?;
    form_with_frontiers(slots, bounds, true, false, prefill_quantum)
}

/// Native decode binds every row's own causal position on the GPU. Prefill
/// grouping remains exact-frontier until its separate input path is qualified.
pub(super) fn form_decode(slots: &[Ready], bounds: Limits) -> Result<Vec<Wave>, ServiceError> {
    form_with_frontiers(slots, bounds, true, false, 32)
}

/// Every native row consumes exactly one teacher/device token at its own
/// committed frontier. Prompt readout barriers are partitioned by the owner.
pub(super) fn form_native(slots: &[Ready], bounds: Limits) -> Result<Vec<Wave>, ServiceError> {
    if slots.iter().all(|row| row.phase == Phase::Decode) {
        form_decode(slots, bounds)
    } else {
        form_with_frontiers(slots, bounds, true, true, 32)
    }
}

fn form_with_frontiers(
    slots: &[Ready],
    bounds: Limits,
    unequal_decode: bool,
    unequal_prefill: bool,
    prefill_quantum: usize,
) -> Result<Vec<Wave>, ServiceError> {
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
    let mut heads = Vec::<(usize, Phase, Option<usize>)>::new();
    for (index, ready) in slots.iter().enumerate() {
        if ready.frontier > 512
            || (ready.phase == Phase::Prefill && ready.frontier >= 512)
            || ready.tokens == 0
            || ready.tokens > prefill_quantum
            || ready.scratch_bytes == 0
            || ready.tokens > bounds.max_tokens
            || ready.scratch_bytes > bounds.max_scratch
            || ((ready.phase == Phase::Decode || unequal_prefill) && ready.tokens != 1)
        {
            return Err(ServiceError::new(
                400,
                "invalid_wave_member",
                "GPU work is outside source/budget",
            ));
        }
        let frontier = if (unequal_decode && ready.phase == Phase::Decode)
            || (unequal_prefill && ready.phase == Phase::Prefill)
        {
            None
        } else {
            Some(ready.frontier)
        };
        let key = (ready.source, ready.phase, frontier);
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

    #[test]
    fn cached_whole_prompt_round_preserves_staggered_refill_and_source() {
        let ready = [
            r(1, Phase::Decode, 47, 1),
            r(1, Phase::Prefill, 0, 47),
            r(1, Phase::Decode, 50, 1),
            r(1, Phase::Decode, 48, 1),
            r(2, Phase::Decode, 51, 1),
            r(1, Phase::Decode, 53, 1),
            r(1, Phase::Prefill, 0, 57),
            r(1, Phase::Prefill, 0, 72),
        ];
        let waves = form_cached_with_prefill_quantum(
            &ready,
            Limits {
                max_members: 4,
                max_tokens: 288,
                max_scratch: 256 * 1024 * 1024,
            },
            72,
        )
        .unwrap();
        assert_eq!(
            waves.iter().map(|w| w.indices.clone()).collect::<Vec<_>>(),
            [vec![0, 2, 3, 5], vec![1, 6, 7], vec![4]]
        );
        let mut all = waves
            .iter()
            .flat_map(|w| w.indices.iter().copied())
            .collect::<Vec<_>>();
        all.sort_unstable();
        assert_eq!(all, (0..8).collect::<Vec<_>>());
        assert_eq!(waves[1].tokens, 47 + 57 + 72);
        for n in 1..=3 {
            let partial = form_cached_with_prefill_quantum(
                &ready[..n],
                Limits {
                    max_members: 4,
                    max_tokens: 288,
                    max_scratch: 256 * 1024 * 1024,
                },
                72,
            )
            .unwrap();
            assert_eq!(partial.iter().map(|w| w.indices.len()).sum::<usize>(), n);
        }
    }

    #[test]
    fn native_teacher_waves_bind_independent_frontiers_without_losing_phase_or_source() {
        let ready = [
            r(1, Phase::Prefill, 0, 1),
            r(1, Phase::Decode, 12, 1),
            r(1, Phase::Prefill, 7, 1),
            r(2, Phase::Prefill, 7, 1),
            r(1, Phase::Decode, 22, 1),
            r(1, Phase::Prefill, 511, 1),
        ];
        let bounds = || Limits {
            max_members: 4,
            max_tokens: 4,
            max_scratch: 4096,
        };
        let waves = form_native(&ready, bounds()).unwrap();
        assert_eq!(
            waves
                .iter()
                .map(|wave| wave.indices.clone())
                .collect::<Vec<_>>(),
            [vec![0, 2, 5], vec![1, 4], vec![3]]
        );
        assert_eq!(
            waves.iter().map(|wave| wave.tokens).sum::<usize>(),
            ready.len()
        );
        // Previously accepted planners retain their exact-frontier policies.
        assert_eq!(form_decode(&ready, bounds()).unwrap().len(), 5);
        assert_eq!(form(&ready, bounds()).unwrap().len(), 6);
        for phase in [Phase::Prefill, Phase::Decode] {
            assert!(form_native(&[r(1, phase, 0, 2)], bounds()).is_err());
        }
    }

    #[test]
    fn hf_partial_batches_never_invent_an_unqualified_band_or_pad_survivors() {
        for count in 0..=64 {
            let bands = qualified_bands(count, &[8, 4, 1]).unwrap();
            assert_eq!(bands.iter().sum::<usize>(), count);
            assert!(bands.iter().all(|n| [8, 4, 1].contains(n)));
        }
        assert_eq!(qualified_bands(7, &[8, 4, 1]).unwrap(), [4, 1, 1, 1]);
        assert_eq!(qualified_bands(8, &[4, 1]).unwrap(), [4, 4]);
        assert_eq!(qualified_bands(3, &[1]).unwrap(), [1, 1, 1]);
        for bad in [
            &[][..],
            &[8, 4][..],
            &[8, 8, 1][..],
            &[1, 4][..],
            &[3, 1][..],
        ] {
            assert!(qualified_bands(8, bad).is_err());
        }
    }
    #[test]
    fn native_bands_cover_every_row_in_order_without_padding() {
        for ceiling in [1, 2, 4, 8] {
            for count in 0..=64 {
                let bands = native_bands(count, ceiling).unwrap();
                assert_eq!(bands.iter().sum::<usize>(), count);
                assert!(bands.windows(2).all(|pair| pair[0] >= pair[1]));
                let mut frontier = 0;
                let mut rows = Vec::new();
                for width in bands {
                    assert!(matches!(width, 1 | 2 | 4 | 8));
                    assert!(width <= ceiling);
                    assert!(frontier + width <= count);
                    for larger in [2, 4, 8] {
                        assert!(larger <= width || larger > ceiling || larger > count - frontier);
                    }
                    rows.extend(frontier..frontier + width);
                    frontier += width;
                }
                assert_eq!(rows, (0..count).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn native_bands_repartition_survivors_and_lower_capacity_ceilings() {
        assert_eq!(native_bands(8, 8).unwrap(), [8]);
        assert_eq!(native_bands(7, 8).unwrap(), [4, 2, 1]);
        assert_eq!(native_bands(6, 8).unwrap(), [4, 2]);
        assert_eq!(native_bands(5, 8).unwrap(), [4, 1]);
        assert_eq!(native_bands(3, 4).unwrap(), [2, 1]);
        assert_eq!(native_bands(8, 4).unwrap(), [4, 4]);
        assert_eq!(native_bands(4, 2).unwrap(), [2, 2]);
        assert_eq!(native_bands(2, 1).unwrap(), [1, 1]);
    }

    #[test]
    fn native_bands_reject_unbounded_or_unsupported_shapes() {
        for ceiling in [0, 3, 5, 6, 7, 9, usize::MAX] {
            assert!(native_bands(8, ceiling).is_err());
        }
        for count in [65, usize::MAX] {
            assert!(native_bands(count, 8).is_err());
        }
    }

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

    #[test]
    fn decode_refill_keeps_source_and_phase_but_preserves_each_frontier() {
        let ready = [
            r(1, Phase::Decode, 255, 1),
            r(2, Phase::Decode, 0, 1),
            r(1, Phase::Decode, 0, 1),
            r(1, Phase::Prefill, 7, 1),
            r(1, Phase::Prefill, 8, 1),
            r(1, Phase::Decode, 256, 1),
            r(1, Phase::Decode, 511, 1),
        ];
        let waves = form_decode(
            &ready,
            Limits {
                max_members: 4,
                max_tokens: 64,
                max_scratch: 4096,
            },
        )
        .unwrap();
        assert_eq!(
            waves
                .iter()
                .map(|wave| wave.indices.clone())
                .collect::<Vec<_>>(),
            [vec![0, 2, 5, 6], vec![1], vec![3], vec![4]]
        );
        assert_eq!(
            waves[0]
                .indices
                .iter()
                .map(|&index| ready[index].frontier)
                .collect::<Vec<_>>(),
            [255, 0, 256, 511]
        );
        assert_eq!(
            waves.iter().map(|wave| wave.tokens).sum::<usize>(),
            ready.len()
        );
    }

    #[test]
    fn eight_atomic47_prefills_keep_real_credit_and_decode_uses_one_b1_wave() {
        let bounds = || Limits {
            max_members: 8,
            max_tokens: 64,
            max_scratch: 256 * 1024 * 1024,
        };
        let prefill = (0..8)
            .map(|_| r(1, Phase::Prefill, 0, 47))
            .collect::<Vec<_>>();
        let waves = form_with_prefill_quantum(&prefill, bounds(), 47).unwrap();
        assert_eq!(waves.len(), 8);
        for (index, wave) in waves.iter().enumerate() {
            assert_eq!(wave.indices, vec![index]);
            assert_eq!(wave.tokens, 47);
            assert_eq!(wave.scratch_bytes, 1024);
        }
        assert_eq!(waves.iter().map(|w| w.tokens).sum::<usize>(), 376);
        let decode = (0..8)
            .map(|_| r(1, Phase::Decode, 47, 1))
            .collect::<Vec<_>>();
        let waves = form_with_prefill_quantum(&decode, bounds(), 47).unwrap();
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].indices, (0..8).collect::<Vec<_>>());
        assert_eq!(waves[0].tokens, 8);
        assert!(form(&prefill, bounds()).is_err());
        assert!(form_native(&prefill, bounds()).is_err());
    }

    fn atomic_bounds(tokens: usize) -> Limits {
        Limits {
            max_members: 2,
            max_tokens: tokens,
            max_scratch: 2048,
        }
    }

    #[test]
    fn atomic_prefill_keeps_two_actual47_members_in_separate64_token_waves() {
        let ready = [r(1, Phase::Prefill, 0, 47), r(1, Phase::Prefill, 0, 47)];
        let waves = form_with_prefill_quantum(&ready, atomic_bounds(64), 47).unwrap();
        assert_eq!(
            waves.iter().map(|w| w.indices.clone()).collect::<Vec<_>>(),
            [vec![0], vec![1]]
        );
        assert_eq!(waves.iter().map(|w| w.tokens).collect::<Vec<_>>(), [47, 47]);
        assert_eq!(waves.iter().map(|w| w.scratch_bytes).sum::<usize>(), 2048);
        assert_eq!(waves.iter().map(|w| w.tokens).sum::<usize>(), 94);
        assert!(form(&ready, atomic_bounds(64)).is_err());
        assert!(form_decode(&ready, atomic_bounds(64)).is_err());
        assert!(form_native(&ready, atomic_bounds(64)).is_err());
    }

    #[test]
    fn atomic_prefill_preserves_source_phase_frontier_and_complete_membership() {
        let ready = [
            r(1, Phase::Prefill, 0, 47),
            r(1, Phase::Prefill, 1, 47),
            r(2, Phase::Prefill, 0, 47),
            r(1, Phase::Decode, 47, 1),
            r(1, Phase::Decode, 47, 1),
            r(1, Phase::Decode, 48, 1),
            r(1, Phase::Prefill, 0, 47),
        ];
        let waves = form_with_prefill_quantum(&ready, atomic_bounds(128), 47).unwrap();
        assert_eq!(
            waves.iter().map(|w| w.indices.clone()).collect::<Vec<_>>(),
            [vec![0, 6], vec![1], vec![2], vec![3, 4], vec![5]]
        );
        let mut covered = waves
            .iter()
            .flat_map(|w| w.indices.iter().copied())
            .collect::<Vec<_>>();
        covered.sort_unstable();
        assert_eq!(covered, (0..ready.len()).collect::<Vec<_>>());
        for wave in &waves {
            assert_eq!(
                wave.tokens,
                wave.indices.iter().map(|&i| ready[i].tokens).sum::<usize>()
            );
            assert_eq!(
                wave.scratch_bytes,
                wave.indices
                    .iter()
                    .map(|&i| ready[i].scratch_bytes)
                    .sum::<usize>()
            );
            assert!(wave.tokens <= 128 && wave.scratch_bytes <= 2048);
        }
    }

    #[test]
    fn atomic_prefill_rejects_span_quantum_and_real_credit_overflow() {
        let valid = r(1, Phase::Prefill, 0, 47);
        for quantum in [0, 513, usize::MAX] {
            assert!(form_with_prefill_quantum(&[valid], atomic_bounds(64), quantum).is_err());
        }
        for bad in [
            r(1, Phase::Prefill, 0, 0),
            r(1, Phase::Prefill, 0, 48),
            r(1, Phase::Prefill, 466, 47),
            r(1, Phase::Prefill, usize::MAX, 47),
            r(1, Phase::Prefill, 0, usize::MAX),
            r(1, Phase::Decode, 47, 2),
        ] {
            assert!(form_with_prefill_quantum(&[bad], atomic_bounds(64), 47).is_err());
        }
        assert!(form_with_prefill_quantum(&[valid], atomic_bounds(46), 47).is_err());
        let insufficient_scratch = Limits {
            max_scratch: 1023,
            ..atomic_bounds(64)
        };
        assert!(form_with_prefill_quantum(&[valid], insufficient_scratch, 47).is_err());
        let edge = r(1, Phase::Prefill, 465, 47);
        assert!(form_with_prefill_quantum(&[edge], atomic_bounds(64), 47).is_ok());
        // Existing32-token and native1-token APIs retain their own contracts.
        assert!(form(&[r(1, Phase::Prefill, 0, 32)], atomic_bounds(64)).is_ok());
        assert!(form(&[r(1, Phase::Prefill, 0, 33)], atomic_bounds(64)).is_err());
        assert!(form_native(&[r(1, Phase::Prefill, 0, 1)], atomic_bounds(64)).is_ok());
        assert!(form_native(&[r(1, Phase::Prefill, 0, 2)], atomic_bounds(64)).is_err());
    }
}

#[cfg(test)]
mod short_hf_atomic_tests {
    use super::*;
    fn bounds(tokens: usize) -> Limits {
        Limits {
            max_members: 8,
            max_tokens: tokens,
            max_scratch: 256 * 1024 * 1024,
        }
    }
    #[test]
    fn genuine_ragged_atomic_prompts_use_exact_rows_and_original_source_groups() {
        let rows = [57, 62, 67, 72].map(|tokens| Ready {
            source: 3,
            phase: Phase::Prefill,
            frontier: 0,
            tokens,
            scratch_bytes: 32 * 1024 * 1024,
        });
        assert!(form(&rows, bounds(72)).is_err()); // Default32 stays strict.
        assert!(form_with_prefill_quantum(&rows, bounds(64), 72).is_err());
        let waves = form_with_prefill_quantum(&rows, bounds(72), 72).unwrap();
        assert_eq!(
            waves.iter().map(|w| w.tokens).collect::<Vec<_>>(),
            vec![57, 62, 67, 72]
        );
        assert_eq!(
            waves.iter().map(|w| w.indices.clone()).collect::<Vec<_>>(),
            vec![vec![0], vec![1], vec![2], vec![3]]
        );
        let cached = [
            Ready {
                source: 3,
                phase: Phase::Decode,
                frontier: 72,
                tokens: 1,
                scratch_bytes: 32 * 1024 * 1024,
            },
            Ready {
                source: 4,
                phase: Phase::Decode,
                frontier: 72,
                tokens: 1,
                scratch_bytes: 32 * 1024 * 1024,
            },
            Ready {
                source: 3,
                phase: Phase::Decode,
                frontier: 73,
                tokens: 1,
                scratch_bytes: 32 * 1024 * 1024,
            },
        ];
        assert_eq!(
            form_with_prefill_quantum(&cached, bounds(72), 72)
                .unwrap()
                .len(),
            3
        );
    }
}
