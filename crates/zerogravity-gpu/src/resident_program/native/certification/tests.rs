use super::*;

#[test]
fn cohort_predicate_preserves_all_bf16_exponent_classes() {
    for bits in 0..=u16::MAX {
        let original = bits & 0x7f80 == 0x7f80;
        let exponent = (bits >> 7) & 0xff;
        assert_eq!(original, exponent == 255);
    }
}

#[test]
fn generated_cohort_covers_exact_whole_ranges_without_padding_reads() -> Result<()> {
    for lengths in [vec![1, 129], vec![127, 128, 129, 1024, 31, 32, 33, 248320]] {
        let (text, grid) = source(&lengths)?;
        assert!(text.contains("atomicOr(status,1u)"));
        assert_eq!(
            text.matches("const unsigned short* x").count(),
            lengths.len()
        );
        let mut count = vec![0usize; lengths.len()];
        let mut start = 0;
        for (member, &n) in lengths.iter().enumerate() {
            let end = start + n.div_ceil(128);
            assert!(text.contains(&format!(
                "(b<{end}u){{i=(b-{start}u)*128u+threadIdx.x;if(i<{n}u"
            )));
            for block in start..end {
                for thread in 0..128 {
                    let i = (block - start) * 128 + thread;
                    if i < n {
                        count[member] += 1;
                    }
                }
            }
            start = end;
        }
        assert_eq!(start, grid);
        assert_eq!(
            count,
            lengths.into_iter().map(|n| n as usize).collect::<Vec<_>>()
        );
    }
    for lengths in [vec![], vec![1], vec![1, 0], vec![1; 9]] {
        assert!(source(&lengths).is_err());
    }
    Ok(())
}

#[test]
fn wrong_missing_duplicate_and_replayed_producers_cannot_flush() -> Result<()> {
    let observed = [AtomicU8::new(0)];
    let started = AtomicBool::new(false);
    assert!(observe(&observed[0], 1, 7, 6).is_err());
    assert_eq!(observed[0].load(Ordering::Acquire), 0);
    assert!(begin_flush(&observed, &[3], &started).is_err());
    observe(&observed[0], 1, 7, 7)?;
    assert!(observe(&observed[0], 1, 7, 7).is_err());
    assert!(begin_flush(&observed, &[3], &started).is_err());
    observe(&observed[0], 2, 8, 8)?;
    begin_flush(&observed, &[3], &started)?;
    assert!(begin_flush(&observed, &[3], &started).is_err());
    assert!(begin_flush(&observed, &[], &AtomicBool::new(false)).is_err());
    Ok(())
}

#[test]
fn compiler_context_all_retained_subsets_use_only_exact_prepared_groups_or_individuals()
-> Result<()> {
    use crate::vllm_cuda_qualification::model_values::bounded::finite_code_groups;
    use cubecl_runtime::kernel::KernelMetadata;
    // A non-dedicated hole and singleton tail must never shift runtime groups.
    let mut dedicated = vec![true; 18];
    dedicated[3] = false;
    let groups = finite_code_groups(&dedicated)?;
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0], [0, 1, 2, 4, 5, 6, 7, 8].map(ValueId));
    assert_eq!(groups[1], [9, 10, 11, 12, 13, 14, 15, 16].map(ValueId));
    for rows in [1, 47, 57, 62, 67, 72] {
        for hidden in [1024, 2048] {
            // Exact ordered physical extents are retained in source and ABI.
            let sizes = (0..18)
                .map(|id| (rows * (hidden + id * 128)) as u32)
                .collect::<Vec<_>>();
            let prepared = groups
                .iter()
                .map(|g| {
                    definition(&g.iter().map(|id| sizes[id.0 as usize]).collect::<Vec<_>>())
                        .map(|(k, _)| k.id())
                })
                .collect::<Result<Vec<_>>>()?;
            let individuals = sizes
                .iter()
                .map(|&n| super::super::finite_definition(n as usize).map(|k| k.id()))
                .collect::<Result<Vec<_>>>()?;
            for mask in 0..(1usize << WIDTH) {
                let eligible = (0..18u32)
                    .filter(|id| {
                        groups[0]
                            .iter()
                            .position(|member| member.0 == *id)
                            .is_none_or(|bit| mask & (1 << bit) != 0)
                    })
                    .map(|id| (id, id as usize))
                    .collect::<BTreeMap<_, _>>();
                let mut deferred = BTreeMap::new();
                for (index, group) in groups.iter().enumerate() {
                    let members = group
                        .iter()
                        .map(|&id| (id, id.0 as usize))
                        .collect::<Vec<_>>();
                    if complete_group(&members, &eligible) {
                        let actual = definition(
                            &group
                                .iter()
                                .map(|id| sizes[id.0 as usize])
                                .collect::<Vec<_>>(),
                        )?
                        .0
                        .id();
                        assert_eq!(actual, prepared[index]);
                        for id in group {
                            assert!(deferred.insert(id.0, index).is_none());
                        }
                    }
                }
                assert_eq!(deferred.len(), if mask == 255 { 16 } else { 8 });
                // Every exclusion, non-dedicated value and singleton still has
                // the original exact individual producer-side certificate.
                for id in 0..18 {
                    if !deferred.contains_key(&(id as u32)) {
                        assert_eq!(
                            super::super::finite_definition(sizes[id] as usize)?.id(),
                            individuals[id]
                        );
                    }
                }
            }
            let original = definition(&[sizes[0], sizes[1]])?.0.id();
            assert_ne!(original, definition(&[sizes[1], sizes[0]])?.0.id());
        }
    }
    assert!(finite_code_groups(&[true; 513]).is_err());
    assert!(finite_code_groups(&[])?.is_empty());
    assert!(finite_code_groups(&[true])?.is_empty());
    Ok(())
}
