//! Finite, deterministic source-aware GPU request wave formation.
//!
//! A wave is scheduler membership, not physical completion or memory credit.
//! The GPU worker separately selects qualified native bands or original B1
//! execution and reports actual Ready/dispatch/root receipts.

use super::ServiceError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}
#[derive(Clone, Copy, Debug)]
pub struct Ready {
    pub source: usize,
    pub phase: Phase,
    pub frontier: usize,
    /// Exact configured context bound for this row. Wave formation owns no
    /// model-specific context constant and never widens this admission.
    pub capacity: usize,
    pub tokens: usize,
    pub scratch_bytes: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wave {
    pub indices: Vec<usize>,
    pub tokens: usize,
    pub scratch_bytes: usize,
}
pub struct Limits {
    pub max_members: usize,
    pub max_tokens: usize,
    pub max_scratch: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fairness {
    /// Aggregate logical tokens allowed across all waves selected this round.
    pub round_tokens: usize,
    /// Decode may defer an eligible prefill this many complete rounds.
    pub prefill_age_rounds: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FairRound {
    pub waves: Vec<Wave>,
    pub deferred: Vec<usize>,
}

/// Decode rows always receive their eligible one-token quantum. Prefill runs
/// normally when decode is absent; while decode is live it enters only after
/// finite aging. Deferred rows are returned separately so the worker can retain
/// their original owners without manufacturing completion or retry authority.
pub(crate) fn fair_round(
    slots: &[Ready],
    ages: &[usize],
    waves: Vec<Wave>,
    fairness: Fairness,
) -> Result<FairRound, ServiceError> {
    if slots.is_empty()
        || slots.len() != ages.len()
        || fairness.round_tokens == 0
        || fairness.prefill_age_rounds == 0
    {
        return Err(ServiceError::new(
            400,
            "invalid_fair_round",
            "bounded round tokens and finite prefill aging required",
        ));
    }
    let mut seen = vec![false; slots.len()];
    let mut classified = Vec::with_capacity(waves.len());
    for (order, wave) in waves.into_iter().enumerate() {
        if wave.indices.is_empty() || wave.tokens == 0 || wave.tokens > fairness.round_tokens {
            return Err(ServiceError::new(
                400,
                "invalid_fair_round",
                "wave exceeds bounded fair-round token budget",
            ));
        }
        let first = *wave.indices.first().expect("nonempty wave checked above");
        let phase = slots
            .get(first)
            .ok_or_else(|| {
                ServiceError::new(
                    400,
                    "invalid_fair_round",
                    "wave row index outside ready set",
                )
            })?
            .phase;
        let mut age = 0usize;
        for &index in &wave.indices {
            let slot = slots.get(index).ok_or_else(|| {
                ServiceError::new(
                    400,
                    "invalid_fair_round",
                    "wave row index outside ready set",
                )
            })?;
            if slot.phase != phase || seen[index] {
                return Err(ServiceError::new(
                    400,
                    "invalid_fair_round",
                    "wave rows must be unique and phase-homogeneous",
                ));
            }
            seen[index] = true;
            age = age.max(ages[index]);
        }
        classified.push((order, phase, age, wave));
    }
    if seen.iter().any(|&included| !included) {
        return Err(ServiceError::new(
            400,
            "invalid_fair_round",
            "fair round must account every ready row exactly once",
        ));
    }

    let decode_live = classified
        .iter()
        .any(|(_, phase, _, _)| *phase == Phase::Decode);
    let mut priority = classified
        .iter()
        .enumerate()
        .filter_map(|(index, (order, phase, age, _))| {
            let class = match phase {
                Phase::Decode => 0usize,
                Phase::Prefill if !decode_live || *age >= fairness.prefill_age_rounds => 1,
                Phase::Prefill => return None,
            };
            Some((class, usize::MAX - *age, *order, index))
        })
        .collect::<Vec<_>>();
    priority.sort_unstable();

    let mut selected = vec![false; classified.len()];
    let mut selected_order = Vec::new();
    let mut used = 0usize;
    for (_, _, _, index) in priority {
        let wave = &classified[index].3;
        if used
            .checked_add(wave.tokens)
            .is_some_and(|next| next <= fairness.round_tokens)
        {
            used += wave.tokens;
            selected[index] = true;
            selected_order.push(index);
        }
    }
    if decode_live
        && classified
            .iter()
            .enumerate()
            .any(|(index, (_, phase, _, _))| *phase == Phase::Decode && !selected[index])
    {
        return Err(ServiceError::new(
            400,
            "invalid_fair_round",
            "round budget cannot protect every active decode row",
        ));
    }
    let aged_prefill = classified
        .iter()
        .any(|(_, phase, age, _)| *phase == Phase::Prefill && *age >= fairness.prefill_age_rounds);
    if aged_prefill
        && !classified
            .iter()
            .enumerate()
            .any(|(index, (_, phase, age, _))| {
                selected[index] && *phase == Phase::Prefill && *age >= fairness.prefill_age_rounds
            })
    {
        return Err(ServiceError::new(
            400,
            "invalid_fair_round",
            "round budget cannot service the oldest aged prefill wave",
        ));
    }

    let mut remaining = classified
        .into_iter()
        .map(|(_, _, _, wave)| Some(wave))
        .collect::<Vec<_>>();
    let mut chosen = Vec::with_capacity(selected_order.len());
    for index in selected_order {
        chosen.push(remaining[index].take().expect("selected wave is unique"));
    }
    let mut deferred = Vec::new();
    for wave in remaining.into_iter().flatten() {
        deferred.extend(wave.indices);
    }
    Ok(FairRound {
        waves: chosen,
        deferred,
    })
}

/// Contiguous, unpadded native bands; B1 carries an unmatched survivor.
/// Ceiling 1 is the final prelaunch capacity fallback.
pub fn native_bands(count: usize, ceiling: usize) -> Result<Vec<usize>, ServiceError> {
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
pub fn qualified_bands(count: usize, widths: &[usize]) -> Result<Vec<usize>, ServiceError> {
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
#[cfg(test)]
pub fn bf16_bands(count: usize, ceiling: usize) -> Result<Vec<usize>, ServiceError> {
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
pub fn form(slots: &[Ready], bounds: Limits) -> Result<Vec<Wave>, ServiceError> {
    form_with_frontiers(slots, bounds, false, false, 32)
}

/// Atomic source prefill consumes its actual whole quantum; quotas never split
/// a member or turn wave membership into native/completion authority.
/// Caller already validated each source's prompt and execution route.
pub fn form_with_prefill_quantum(
    slots: &[Ready],
    bounds: Limits,
    prefill_quantum: usize,
) -> Result<Vec<Wave>, ServiceError> {
    if prefill_quantum == 0
        || slots.len() > 64
        || slots
            .iter()
            .any(|row| row.capacity == 0 || prefill_quantum > row.capacity)
    {
        return Err(ServiceError::new(
            400,
            "invalid_wave",
            "atomic prefill quantum exceeds admitted row context",
        ));
    }
    form_with_frontiers(slots, bounds, false, false, prefill_quantum)
}

/// Whole original prompts retain their atomic quantum; cached requests keep
/// individual frontiers in their row bindings while sharing a fair wave.
pub fn form_cached_with_prefill_quantum(
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
pub fn form_decode(slots: &[Ready], bounds: Limits) -> Result<Vec<Wave>, ServiceError> {
    form_with_frontiers(slots, bounds, true, false, 32)
}

/// Every native row consumes exactly one teacher/device token at its own
/// committed frontier. Prompt readout barriers are partitioned by the owner.
pub fn form_native(slots: &[Ready], bounds: Limits) -> Result<Vec<Wave>, ServiceError> {
    if slots.iter().all(|row| row.phase == Phase::Decode) {
        form_decode(slots, bounds)
    } else {
        form_with_frontiers(slots, bounds, true, true, 32)
    }
}

pub(crate) fn form_with_frontiers(
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
        if ready.capacity == 0
            || ready
                .frontier
                .checked_add(ready.tokens)
                .is_none_or(|end| end > ready.capacity)
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
            capacity: 512,
            tokens,
            scratch_bytes: 1024,
        }
    }

    fn fair(slots: &[Ready], ages: &[usize], round_tokens: usize) -> FairRound {
        let waves = form_with_frontiers(
            slots,
            Limits {
                max_members: 8,
                max_tokens: 64,
                max_scratch: 64 * 1024,
            },
            true,
            false,
            32,
        )
        .unwrap();
        fair_round(
            slots,
            ages,
            waves,
            Fairness {
                round_tokens,
                prefill_age_rounds: 2,
            },
        )
        .unwrap()
    }

    #[test]
    fn decode_is_protected_until_prefill_reaches_finite_age() {
        // Wave formation sees prefill first. Fair selection must still execute
        // decode first and defer the young prompt without dropping ownership.
        let slots = [
            r(1, Phase::Prefill, 0, 32),
            r(1, Phase::Decode, 40, 1),
            r(1, Phase::Decode, 41, 1),
            r(1, Phase::Prefill, 0, 32),
        ];
        let young = fair(&slots, &[0, 0, 0, 0], 66);
        assert_eq!(
            young
                .waves
                .iter()
                .map(|wave| wave.indices.clone())
                .collect::<Vec<_>>(),
            [vec![1, 2]]
        );
        assert_eq!(young.deferred, [0, 3]);

        let aged = fair(&slots, &[2, 0, 0, 2], 66);
        assert_eq!(
            aged.waves
                .iter()
                .map(|wave| wave.indices.clone())
                .collect::<Vec<_>>(),
            [vec![1, 2], vec![0, 3]]
        );
        assert!(aged.deferred.is_empty());
    }

    #[test]
    fn oldest_prefill_wins_a_bounded_prefill_only_round() {
        let slots = [
            r(1, Phase::Prefill, 0, 32),
            r(2, Phase::Prefill, 0, 32),
            r(3, Phase::Prefill, 0, 32),
            r(4, Phase::Prefill, 0, 32),
        ];
        let round = fair(&slots, &[0, 4, 1, 2], 32);
        assert_eq!(round.waves.len(), 1);
        assert_eq!(round.waves[0].indices, [1]);
        assert_eq!(round.deferred, [0, 2, 3]);
    }

    #[test]
    fn multiple_aged_prefills_rotate_behind_complete_decode_service() {
        let slots = [
            r(1, Phase::Decode, 20, 1),
            r(2, Phase::Decode, 21, 1),
            r(3, Phase::Prefill, 0, 32),
            r(4, Phase::Prefill, 0, 32),
            r(5, Phase::Prefill, 0, 32),
        ];
        let round = fair(&slots, &[0, 0, 4, 3, 2], 34);
        assert_eq!(
            round
                .waves
                .iter()
                .map(|wave| wave.indices.clone())
                .collect::<Vec<_>>(),
            [vec![0], vec![1], vec![2]]
        );
        assert_eq!(round.deferred, [3, 4]);
    }

    #[test]
    fn fair_round_refuses_budget_that_cannot_protect_decode_or_aged_prefill() {
        let slots = [
            r(1, Phase::Decode, 20, 1),
            r(2, Phase::Decode, 21, 1),
            r(3, Phase::Prefill, 0, 32),
        ];
        let waves = form_with_frontiers(
            &slots,
            Limits {
                max_members: 8,
                max_tokens: 64,
                max_scratch: 64 * 1024,
            },
            true,
            false,
            32,
        )
        .unwrap();
        assert!(
            fair_round(
                &slots,
                &[0, 0, 2],
                waves,
                Fairness {
                    round_tokens: 33,
                    prefill_age_rounds: 2,
                },
            )
            .is_err()
        );
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
    fn every_phase_respects_each_rows_admitted_context() {
        for phase in [Phase::Prefill, Phase::Decode] {
            let last = r(1, phase, 511, 1);
            assert!(form_native(&[last], atomic_bounds(64)).is_ok());
            for frontier in [512, usize::MAX] {
                let invalid = r(1, phase, frontier, 1);
                assert!(form_native(&[invalid], atomic_bounds(64)).is_err());
                assert!(form_with_prefill_quantum(&[invalid], atomic_bounds(64), 47).is_err());
            }
        }
        let invalid_decode = [r(1, Phase::Decode, 512, 1)];
        assert!(form(&invalid_decode, atomic_bounds(64)).is_err());
        assert!(form_decode(&invalid_decode, atomic_bounds(64)).is_err());

        let long = Ready {
            source: 1,
            phase: Phase::Prefill,
            frontier: 16_384,
            capacity: 32_768,
            tokens: 64,
            scratch_bytes: 1024,
        };
        assert!(form_with_prefill_quantum(&[long], atomic_bounds(64), 64).is_ok());
        let exhausted = Ready {
            frontier: 32_705,
            ..long
        };
        assert!(form_with_prefill_quantum(&[exhausted], atomic_bounds(64), 64).is_err());
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
mod qwen35_cuda_atomic_tests {
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
            capacity: 512,
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
                capacity: 512,
                tokens: 1,
                scratch_bytes: 32 * 1024 * 1024,
            },
            Ready {
                source: 4,
                phase: Phase::Decode,
                frontier: 72,
                capacity: 512,
                tokens: 1,
                scratch_bytes: 32 * 1024 * 1024,
            },
            Ready {
                source: 3,
                phase: Phase::Decode,
                frontier: 73,
                capacity: 512,
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
