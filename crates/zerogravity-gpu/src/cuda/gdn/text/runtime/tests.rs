use super::*;

fn runtime_config() -> Result<GdnRuntimeConfig> {
    Ok(GdnRuntimeConfig::new(
        GdnKernelConfig::from_stages(
            GdnKktKernel::Bk32W8S3,
            GdnRecomputeKernel::Bk64Bv64W8S3,
            GdnStateKernel::Bv64W4S2,
            GdnOutputKernel::Bk32Bv64W4S3,
        )?,
        GdnRuntimeKernelGeometry::new(6144, 2048, 32, 4, 16, 128, 64, 5)?,
    ))
}

fn control(plan: &GdnRuntimePlan, role: GdnRuntimeBuffer) -> &[u8] {
    plan.controls()
        .iter()
        .find(|control| control.role() == role)
        .unwrap()
        .bytes()
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

#[test]
fn chunk64_and_tail_share_one_workspace_and_advance_explicit_state() -> Result<()> {
    let slot = GdnRuntimeCacheSlot::new(3)?;
    let prefill = GdnRuntimePrefill::new(runtime_config()?, slot, 0, 69, false)?;
    assert_eq!(prefill.chunk_count(), 2);
    assert_eq!(prefill.resulting_frontier(), 69);

    let first = prefill.chunk(0)?;
    let tail = prefill.chunk(1)?;
    assert_eq!(first.geometry().rows(), 64);
    assert_eq!(first.geometry().frontier(), 0);
    assert!(!first.geometry().has_initial());
    assert_eq!(tail.geometry().rows(), 5);
    assert_eq!(tail.geometry().frontier(), 64);
    assert!(tail.geometry().has_initial());
    assert!(Arc::ptr_eq(&first.workspace, &tail.workspace));
    assert_eq!(control(&first, GdnRuntimeBuffer::HasInitial), [0]);
    assert_eq!(control(&tail, GdnRuntimeBuffer::HasInitial), [1]);
    assert_eq!(i32s(control(&tail, GdnRuntimeBuffer::QueryStarts)), [0, 5]);
    assert_eq!(i32s(control(&tail, GdnRuntimeBuffer::ChunkIndices)), [0, 0]);
    assert_eq!(
        control(&tail, GdnRuntimeBuffer::CacheIndices),
        3_i32.to_le_bytes()
    );
    assert_eq!(
        first.initial_recurrent_state(),
        GdnRuntimeInitialState::Zero {
            destination: local_state_view(
                GdnRuntimeBuffer::InitialState,
                GdnRuntimeStateKind::RecurrentInitial,
            ),
        }
    );
    assert_eq!(
        tail.initial_recurrent_state(),
        GdnRuntimeInitialState::Copy {
            source: cache_recurrent_view(slot),
            destination: local_state_view(
                GdnRuntimeBuffer::InitialState,
                GdnRuntimeStateKind::RecurrentInitial,
            ),
        }
    );
    assert_eq!(
        tail.final_recurrent_state(),
        GdnRuntimeFinalState::Scatter {
            source: local_state_view(
                GdnRuntimeBuffer::FinalState,
                GdnRuntimeStateKind::RecurrentFinal,
            ),
            destination: cache_recurrent_view(slot),
        }
    );
    assert!(prefill.chunk(2).is_err());
    Ok(())
}

#[test]
fn nonzero_prior_is_explicit_and_cached_decode_accepts_arbitrary_frontier() -> Result<()> {
    let config = runtime_config()?;
    let slot = GdnRuntimeCacheSlot::new(4)?;
    let prefill = GdnRuntimePrefill::new(config, slot, 137, 70, true)?;
    let first = prefill.chunk(0)?;
    let tail = prefill.chunk(1)?;
    assert_eq!(
        (first.geometry().frontier(), first.geometry().rows()),
        (137, 64)
    );
    assert_eq!(
        (tail.geometry().frontier(), tail.geometry().rows()),
        (201, 6)
    );
    assert_eq!(control(&first, GdnRuntimeBuffer::HasInitial), [1]);
    assert_eq!(control(&tail, GdnRuntimeBuffer::HasInitial), [1]);

    for frontier in [1, 2, 63, 64, 207, 8192, 1_000_003] {
        let decode = prefill.cached_decode(frontier)?;
        assert_eq!(decode.geometry().mode(), GdnRuntimeMode::CachedDecode);
        assert_eq!(decode.geometry().frontier(), frontier);
        assert_eq!(decode.geometry().resulting_frontier(), frontier + 1);
        assert_eq!(decode.stages().len(), 3);
        assert!(decode.stage(GdnRuntimeStageKind::Decode).is_some());
        assert!(decode.stage(GdnRuntimeStageKind::Prep).is_none());
        assert_eq!(
            decode.initial_recurrent_state(),
            GdnRuntimeInitialState::InPlace {
                state: cache_recurrent_view(slot),
            }
        );
        assert_eq!(
            decode.final_recurrent_state(),
            GdnRuntimeFinalState::InPlace {
                state: cache_recurrent_view(slot),
            }
        );
        assert_eq!(
            control(&decode, GdnRuntimeBuffer::CacheIndices),
            4_i32.to_le_bytes()
        );
        assert!(Arc::ptr_eq(&prefill.workspace, &decode.workspace));
    }
    Ok(())
}

#[test]
fn arbitrary_rows_have_checked_grids_abis_and_workspace_extents() -> Result<()> {
    let config = runtime_config()?;
    let slot = GdnRuntimeCacheSlot::new(1)?;
    for rows in [1, 2, 7, 8, 9, 16, 31, 48, 63, 64] {
        let plan = GdnRuntimePlan::prefill_chunk(config, slot, 0, rows, false)?;
        let grid = |stage| plan.stage(stage).unwrap().identity().grid();
        assert_eq!(
            grid(GdnRuntimeStageKind::Conv),
            [rows.div_ceil(8) as u32, 24, 1]
        );
        assert_eq!(
            grid(GdnRuntimeStageKind::Prep),
            [rows.div_ceil(16) as u32, 32, 1]
        );
        for stage in [
            GdnRuntimeStageKind::Cumsum,
            GdnRuntimeStageKind::Kkt,
            GdnRuntimeStageKind::Solve,
            GdnRuntimeStageKind::Recompute,
        ] {
            assert_eq!(grid(stage), [1, 16, 1]);
        }
        assert_eq!(grid(GdnRuntimeStageKind::State), [2, 16, 1]);
        assert_eq!(grid(GdnRuntimeStageKind::Output), [2, 1, 16]);
        assert_eq!(grid(GdnRuntimeStageKind::Gated), [(rows * 4) as u32, 1, 1]);

        let prep = plan.stage(GdnRuntimeStageKind::Prep).unwrap().identity();
        let q = prep
            .bindings()
            .iter()
            .find(|binding| binding.role() == GdnRuntimeBuffer::Q)
            .unwrap();
        assert_eq!(q.bytes(), rows * 2048 * 2);
        let solve = plan.stage(GdnRuntimeStageKind::Solve).unwrap().identity();
        let inverse = solve
            .bindings()
            .iter()
            .find(|binding| binding.role() == GdnRuntimeBuffer::Inverse)
            .unwrap();
        assert_eq!(inverse.bytes(), rows * 16 * 64 * 2);
        assert_eq!(
            plan.workspace().bytes(GdnRuntimeBuffer::Q),
            Some(64 * 2048 * 2)
        );
        assert_eq!(
            plan.workspace().bytes(GdnRuntimeBuffer::Inverse),
            Some(64 * 16 * 64 * 2)
        );
        assert_eq!(plan.workspace().cache_slots(), 5);
        assert_eq!(
            plan.workspace().bytes(GdnRuntimeBuffer::CacheArena),
            Some(5 * CACHE_SLOT_BYTES)
        );
    }

    for tokens in [1, 63, 64, 65, 127, 128, 129, 1000, 65_537, usize::MAX] {
        let prefill = GdnRuntimePrefill::new(config, slot, 0, tokens, false)?;
        assert_eq!(prefill.chunk_count(), tokens.div_ceil(64));
        let last = prefill.chunk(prefill.chunk_count() - 1)?;
        assert_eq!(
            last.geometry().rows(),
            tokens - (prefill.chunk_count() - 1) * 64
        );
        assert_eq!(last.geometry().resulting_frontier(), tokens);
    }
    Ok(())
}

#[test]
fn source_abi_config_and_geometry_form_execution_identity() -> Result<()> {
    let base = runtime_config()?;
    let alternate = GdnRuntimeConfig::new(
        GdnKernelConfig::from_stages(
            GdnKktKernel::Bk32W4S3,
            GdnRecomputeKernel::Bk64Bv64W8S4,
            GdnStateKernel::Bv32W4S2,
            GdnOutputKernel::Bk32Bv32W4S2,
        )?,
        base.geometry(),
    );
    let slot = GdnRuntimeCacheSlot::new(1)?;
    let a = GdnRuntimePlan::prefill_chunk(base, slot, 0, 17, false)?;
    let b = GdnRuntimePlan::prefill_chunk(alternate, slot, 0, 17, false)?;
    let c = GdnRuntimePlan::prefill_chunk(base, slot, 64, 17, true)?;
    let a_kkt = a.stage(GdnRuntimeStageKind::Kkt).unwrap().identity();
    let b_kkt = b.stage(GdnRuntimeStageKind::Kkt).unwrap().identity();
    assert_ne!(a_kkt, b_kkt);
    assert_ne!(a_kkt.source(), b_kkt.source());
    assert_ne!(a_kkt, c.stage(GdnRuntimeStageKind::Kkt).unwrap().identity());
    assert_eq!(
        a_kkt.source(),
        c.stage(GdnRuntimeStageKind::Kkt)
            .unwrap()
            .identity()
            .source()
    );
    assert_eq!(
        a_kkt.bindings(),
        c.stage(GdnRuntimeStageKind::Kkt)
            .unwrap()
            .identity()
            .bindings()
    );

    let d1 = GdnRuntimePlan::cached_decode(base, slot, 17)?;
    let d2 = GdnRuntimePlan::cached_decode(base, slot, 999)?;
    assert_ne!(
        d1.stage(GdnRuntimeStageKind::Decode).unwrap().identity(),
        d2.stage(GdnRuntimeStageKind::Decode).unwrap().identity()
    );
    assert_eq!(
        d1.stage(GdnRuntimeStageKind::Decode)
            .unwrap()
            .identity()
            .source(),
        d2.stage(GdnRuntimeStageKind::Decode)
            .unwrap()
            .identity()
            .source()
    );
    Ok(())
}

#[test]
fn invalid_configuration_state_geometry_and_indices_refuse() -> Result<()> {
    assert!(GdnRuntimeKernelGeometry::new(6145, 2048, 32, 4, 16, 128, 64, 5).is_err());
    assert!(GdnRuntimeKernelGeometry::new(6144, 2048, 32, 4, 16, 128, 63, 5).is_err());
    assert!(GdnRuntimeKernelGeometry::new(6144, 2048, 32, 4, 16, 128, 64, 1).is_err());
    assert!(GdnRuntimeCacheSlot::new(0).is_err());
    assert!(GdnRuntimeCacheSlot::new(5).is_err());
    for index in 1..5 {
        assert_eq!(GdnRuntimeCacheSlot::new(index)?.index(), index);
    }

    let config = runtime_config()?;
    let slot = GdnRuntimeCacheSlot::new(1)?;
    assert!(GdnRuntimePlan::prefill_chunk(config, slot, 0, 0, false).is_err());
    assert!(GdnRuntimePlan::prefill_chunk(config, slot, 0, 65, false).is_err());
    assert!(GdnRuntimePlan::prefill_chunk(config, slot, 0, 1, true).is_err());
    assert!(GdnRuntimePlan::prefill_chunk(config, slot, 9, 1, false).is_err());
    assert!(GdnRuntimePlan::prefill_chunk(config, slot, usize::MAX, 1, true).is_err());
    assert!(GdnRuntimePrefill::new(config, slot, 0, 0, false).is_err());
    assert!(GdnRuntimePrefill::new(config, slot, 0, 2, true).is_err());
    assert!(GdnRuntimePrefill::new(config, slot, 2, 2, false).is_err());
    assert!(GdnRuntimePrefill::new(config, slot, usize::MAX, 2, true).is_err());
    assert!(GdnRuntimePlan::cached_decode(config, slot, 0).is_err());
    assert!(GdnRuntimePlan::cached_decode(config, slot, usize::MAX).is_err());
    Ok(())
}

fn kinds(plan: &GdnRuntimePlan) -> Vec<GdnRuntimeScheduleKind> {
    plan.schedule().iter().map(|step| step.kind()).collect()
}

fn schedule_binding(
    plan: &GdnRuntimePlan,
    kind: GdnRuntimeScheduleKind,
    role: GdnRuntimeBuffer,
) -> GdnRuntimeBinding {
    *plan
        .schedule()
        .iter()
        .find(|step| step.kind() == kind)
        .unwrap()
        .bindings()
        .iter()
        .find(|binding| binding.role() == role)
        .unwrap()
}

#[test]
fn schedules_include_every_copy_in_dependency_order_with_exact_access() -> Result<()> {
    use GdnRuntimeScheduleKind::*;
    let config = runtime_config()?;
    let slot = GdnRuntimeCacheSlot::new(2)?;
    let first = GdnRuntimePlan::prefill_chunk(config, slot, 0, 9, false)?;
    assert_eq!(
        kinds(&first),
        [
            PackA,
            PackB,
            PackZ,
            Conv,
            Prep,
            Cumsum,
            Kkt,
            ZeroInverse,
            Solve,
            Recompute,
            ZeroInitialState,
            State,
            ScatterFinalState,
            Output,
            Gated,
        ]
    );
    assert_eq!(
        schedule_binding(&first, PackA, GdnRuntimeBuffer::PackedBa),
        GdnRuntimeBinding {
            role: GdnRuntimeBuffer::PackedBa,
            offset: 8 * 4,
            bytes: ((9 - 1) * 16 + 8) * 4,
            access: GdnRuntimeAccess::ReadOnly,
        }
    );
    assert_eq!(
        schedule_binding(&first, PackA, GdnRuntimeBuffer::AContiguous).access(),
        GdnRuntimeAccess::WriteOnly
    );
    assert_eq!(
        schedule_binding(&first, PackB, GdnRuntimeBuffer::PackedBa).offset(),
        0
    );
    assert_eq!(
        schedule_binding(&first, PackZ, GdnRuntimeBuffer::PackedQkvz).offset(),
        3072 * 4
    );
    assert_eq!(
        schedule_binding(&first, ZeroInverse, GdnRuntimeBuffer::Inverse).access(),
        GdnRuntimeAccess::WriteOnly
    );
    assert_eq!(
        schedule_binding(&first, ZeroInitialState, GdnRuntimeBuffer::InitialState).access(),
        GdnRuntimeAccess::WriteOnly
    );
    assert_eq!(
        schedule_binding(&first, ScatterFinalState, GdnRuntimeBuffer::FinalState).access(),
        GdnRuntimeAccess::ReadOnly
    );
    assert_eq!(
        schedule_binding(&first, ScatterFinalState, GdnRuntimeBuffer::CacheArena),
        GdnRuntimeBinding {
            role: GdnRuntimeBuffer::CacheArena,
            offset: slot.recurrent_offset(),
            bytes: RECURRENT_STATE_BYTES,
            access: GdnRuntimeAccess::WriteOnly,
        }
    );

    let continued = GdnRuntimePlan::prefill_chunk(config, slot, 9, 1, true)?;
    assert_eq!(
        kinds(&continued),
        [
            PackZ,
            Conv,
            Prep,
            Cumsum,
            Kkt,
            ZeroInverse,
            Solve,
            Recompute,
            GatherInitialState,
            State,
            ScatterFinalState,
            Output,
            Gated,
        ]
    );
    assert_eq!(
        schedule_binding(&continued, GatherInitialState, GdnRuntimeBuffer::CacheArena),
        GdnRuntimeBinding {
            role: GdnRuntimeBuffer::CacheArena,
            offset: slot.recurrent_offset(),
            bytes: RECURRENT_STATE_BYTES,
            access: GdnRuntimeAccess::ReadOnly,
        }
    );

    let cached = GdnRuntimePlan::cached_decode(config, slot, 10)?;
    assert_eq!(kinds(&cached), [CopyQkvPrefix, PackZ, Conv, Decode, Gated]);
    assert_eq!(
        schedule_binding(&cached, CopyQkvPrefix, GdnRuntimeBuffer::PackedQkvz).bytes(),
        3072 * 4
    );
    assert_eq!(
        schedule_binding(&cached, CopyQkvPrefix, GdnRuntimeBuffer::ConvOutput).access(),
        GdnRuntimeAccess::WriteOnly
    );
    Ok(())
}

#[test]
fn recurrent_views_have_distinct_roles_and_explicit_ownership() -> Result<()> {
    let config = runtime_config()?;
    let slot = GdnRuntimeCacheSlot::new(4)?;
    let plan = GdnRuntimePlan::prefill_chunk(config, slot, 64, 2, true)?;
    let ownership = |role| {
        plan.workspace()
            .buffers()
            .iter()
            .find(|layout| layout.role() == role)
            .unwrap()
            .ownership()
    };
    assert_eq!(
        ownership(GdnRuntimeBuffer::CacheArena),
        GdnRuntimeBufferOwnership::CacheState
    );
    assert_eq!(
        ownership(GdnRuntimeBuffer::InitialState),
        GdnRuntimeBufferOwnership::RecurrentState
    );
    assert_eq!(
        ownership(GdnRuntimeBuffer::FinalState),
        GdnRuntimeBufferOwnership::RecurrentState
    );
    let GdnRuntimeInitialState::Copy {
        source: initial_source,
        destination: initial_destination,
    } = plan.initial_recurrent_state()
    else {
        panic!("continued prefill must gather recurrent state");
    };
    let GdnRuntimeFinalState::Scatter {
        source: final_source,
        destination: final_destination,
    } = plan.final_recurrent_state()
    else {
        panic!("prefill must scatter recurrent state");
    };
    assert_ne!(initial_source.role(), initial_destination.role());
    assert_ne!(initial_destination.role(), final_source.role());
    assert_ne!(final_source.role(), final_destination.role());
    assert_eq!(initial_source, final_destination);
    assert_eq!(
        initial_destination.kind(),
        GdnRuntimeStateKind::RecurrentInitial
    );
    assert_eq!(final_source.kind(), GdnRuntimeStateKind::RecurrentFinal);
    Ok(())
}

#[test]
fn caller_handle_binding_rejects_physical_state_aliases_and_slices_exact_views() -> Result<()> {
    use cubecl::server::Handle;
    use cubecl_common::device::{DeviceId, ServiceId};
    use cubecl_core::environment::stream::StreamId;

    struct RuntimeHandleTest;
    let plan = GdnRuntimePlan::prefill_chunk(
        runtime_config()?,
        GdnRuntimeCacheSlot::new(2)?,
        64,
        2,
        true,
    )?;
    let service = ServiceId::of::<RuntimeHandleTest>(DeviceId::new(0, 0));
    let stream = StreamId::current();
    let mut handles = plan
        .workspace()
        .buffers()
        .iter()
        .map(|layout| {
            (
                layout.role(),
                Handle::new(service, stream, layout.bytes() as u64),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let bound = plan.bind_schedule_handles(&handles)?;
    assert_eq!(bound.len(), plan.schedule().len());
    for (step, operation) in plan.schedule().iter().zip(&bound) {
        assert_eq!(operation.len(), step.bindings().len());
        for (handle, binding) in operation.iter().zip(step.bindings()) {
            assert_eq!(handle.size_in_used(), binding.bytes() as u64);
        }
    }

    handles.insert(
        GdnRuntimeBuffer::FinalState,
        handles[&GdnRuntimeBuffer::InitialState].clone(),
    );
    assert!(plan.bind_schedule_handles(&handles).is_err());

    handles.insert(
        GdnRuntimeBuffer::FinalState,
        Handle::new(service, stream, RECURRENT_STATE_BYTES as u64),
    );
    handles.insert(
        GdnRuntimeBuffer::InitialState,
        handles[&GdnRuntimeBuffer::CacheArena].clone(),
    );
    assert!(plan.bind_schedule_handles(&handles).is_err());

    let cached =
        GdnRuntimePlan::cached_decode(runtime_config()?, GdnRuntimeCacheSlot::new(2)?, 66)?;
    let mut required = BTreeMap::<GdnRuntimeBuffer, usize>::new();
    for binding in cached.schedule().iter().flat_map(|step| step.bindings()) {
        let end = binding.offset() + binding.bytes();
        required
            .entry(binding.role())
            .and_modify(|bytes| *bytes = (*bytes).max(end))
            .or_insert(end);
    }
    let cached_handles = required
        .into_iter()
        .map(|(role, bytes)| (role, Handle::new(service, stream, bytes as u64)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        cached.bind_schedule_handles(&cached_handles)?.len(),
        cached.schedule().len()
    );
    assert!(!cached_handles.contains_key(&GdnRuntimeBuffer::InitialState));
    assert!(!cached_handles.contains_key(&GdnRuntimeBuffer::FinalState));
    Ok(())
}

#[test]
fn schedule_order_access_and_state_alias_mutations_refuse() -> Result<()> {
    let plan = GdnRuntimePlan::prefill_chunk(
        runtime_config()?,
        GdnRuntimeCacheSlot::new(3)?,
        64,
        2,
        true,
    )?;
    for index in 0..plan.schedule.len() - 1 {
        let mut changed = plan.clone();
        changed.schedule.swap(index, index + 1);
        assert!(
            changed.require_schedule().is_err(),
            "order mutation {index}"
        );
    }
    for step_index in 0..plan.schedule.len() {
        for binding_index in 0..plan.schedule[step_index].bindings.len() {
            let mut changed = plan.clone();
            let access = &mut changed.schedule[step_index].bindings[binding_index].access;
            *access = match *access {
                GdnRuntimeAccess::ReadOnly => GdnRuntimeAccess::WriteOnly,
                _ => GdnRuntimeAccess::ReadOnly,
            };
            assert!(
                changed.require_schedule().is_err(),
                "access mutation {step_index}/{binding_index}"
            );
        }
    }

    let mut changed = plan.clone();
    let GdnRuntimeInitialState::Copy {
        ref mut destination,
        ..
    } = changed.initial_state
    else {
        panic!("continued prefill state");
    };
    destination.role = GdnRuntimeBuffer::FinalState;
    assert!(changed.require_schedule().is_err());

    let mut changed = plan.clone();
    let GdnRuntimeFinalState::Scatter { ref mut source, .. } = changed.final_state else {
        panic!("prefill final state");
    };
    source.role = GdnRuntimeBuffer::InitialState;
    assert!(changed.require_schedule().is_err());

    let mut changed = plan;
    let workspace = Arc::make_mut(&mut changed.workspace);
    workspace
        .buffers
        .iter_mut()
        .find(|layout| layout.role == GdnRuntimeBuffer::InitialState)
        .unwrap()
        .ownership = GdnRuntimeBufferOwnership::Scratch;
    assert!(changed.require_schedule().is_err());
    Ok(())
}

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
#[test]
#[ignore = "SM86 CUDA compile/reflection diagnostic; no numerical assertion"]
fn ordinary_runtime_sources_compile_and_reflect_exact_abis() -> Result<()> {
    use cubecl::cuda::{CudaRuntime, OwnedCudaQueue};
    use cubecl_runtime::runtime::Runtime;
    use std::sync::atomic::{AtomicBool, Ordering};

    let client = CudaRuntime::client(&Default::default());
    if client.properties().identity.fingerprint != "ptx_sm86" {
        return Err(Error::Unsupported(
            "ordinary GDN runtime ABI diagnostic requires SM86".into(),
        ));
    }
    let (_client, queue) = unsafe { OwnedCudaQueue::pin_cold(client) }
        .map_err(|error| Error::State(format!("pin GDN runtime ABI queue: {error}")))?;
    let adopted = Arc::new(AtomicBool::new(false));
    unsafe { queue.install_compiler_context(Arc::new(()), adopted.clone()) }
        .map_err(|error| Error::State(format!("admit GDN runtime ABI compiler: {error}")))?;
    if !adopted.load(Ordering::Acquire) {
        return Err(Error::State(
            "GDN runtime ABI compiler owner was not adopted".into(),
        ));
    }

    let prefill = GdnRuntimePrefill::new(
        runtime_config()?,
        GdnRuntimeCacheSlot::new(2)?,
        0,
        69,
        false,
    )?;
    for plan in [
        prefill.chunk(0)?,
        prefill.chunk(1)?,
        prefill.cached_decode(69)?,
    ] {
        for stage in plan.stages() {
            let pointers = stage.spec.bindings.len();
            let scalar_bytes = stage.spec.bindings[0].static_scalars.schema().byte_size();
            let parameter_count = pointers + usize::from(scalar_bytes != 0);
            let actual = unsafe {
                queue.diagnose_kernel_parameters(
                    Box::new(stage.spec.kernel.clone()),
                    parameter_count,
                )
            }
            .map_err(|error| {
                Error::State(format!(
                    "{} GDN runtime CUDA reflection: {error}",
                    stage.identity().entry()
                ))
            })?;
            let mut expected = (0..pointers)
                .map(|index| (index * 8, 8))
                .collect::<Vec<_>>();
            if scalar_bytes != 0 {
                expected.push((pointers * 8, scalar_bytes));
            }
            assert_eq!(actual, expected, "{}", stage.identity().entry());
        }
    }
    Ok(())
}
