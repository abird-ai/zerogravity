use super::*;

#[test]
fn private_cached_decode_specialization_is_bounded_and_keeps_base_abi() -> Result<()> {
    use sha2::{Digest, Sha256};

    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_str!("text/cached_decode_parameterized.cu").as_bytes())
        ),
        "e0bd77f07a1ead8b9ef25ce26769aabccd3ac2839fef8eeaee0386c19906ee05"
    );
    for (rows, grid_y, last_arg) in [(4, 64, "zg_arg14"), (8, 128, "zg_arg22")] {
        let source = text::cached_decode_private_batch(rows)?;
        assert_eq!(source.entry, "zg_gdn_decode_cached_parameterized_v1");
        assert_eq!(source.threads, 32);
        assert_eq!(source.grid, [4, grid_y, 1]);
        assert_eq!(
            source.ptx_sha256,
            "e15170544e78ee3ab6b19670b4cc80e9a74db549314bb57f8a897ac855283fd5"
        );
        assert_eq!(source.scalars.packed_bytes().len(), 8);
        assert!(source.source.contains(last_arg));
        assert!(source.source.contains("zg_private_prior"));
        assert!(source.source.contains("zg_private_candidate"));
        assert!(source.source.contains("mov.u64 \\t%%zg_orig_rd2, 0"));
        assert!(!source.source.contains("zg_scalars.words[1]"));
    }
    for rows in [0, 1, 2, 3, 5, 6, 7, 9, 16] {
        assert!(text::cached_decode_private_batch(rows).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "physical SM86 private-state cached GDN decode parity/timing"]
fn physical_cached_decode_private_state_matches_transport_pipeline() -> Result<()> {
    futures_lite::future::block_on(async {
        use crate::cuda::gdn::state_transport::{
            RECURRENT_ELEMENTS, StateCopy, StateTransport, plane_handle,
        };
        use zerogravity_core::device_events::BufferRole;

        const ROWS: usize = 4;
        const REPEATS: usize = 120;
        let diagnostic =
            crate::cuda::test_support::CudaDiagnostic::new("cached-gdn-private-state")?;
        let service = diagnostic.service.clone();
        let client = service.client().clone();
        let baseline = CachedDecodeBatch::prepare(ROWS)?;
        let direct = CachedDecodePrivateBatch::prepare(ROWS)?;

        let bf16_bytes = |words: usize, seed: usize| {
            (0..words)
                .flat_map(|index| {
                    let bits = match (index * 17 + seed) % 8 {
                        0 => 0x3d80u16,
                        1 => 0xbd00,
                        2 => 0x3e00,
                        3 => 0xbe00,
                        4 => 0x3e80,
                        5 => 0xbe80,
                        6 => 0x3f00,
                        _ => 0xbf00,
                    };
                    bits.to_ne_bytes()
                })
                .collect::<Vec<_>>()
        };
        let recurrent_bytes = |lane: usize| {
            (0..RECURRENT_ELEMENTS)
                .flat_map(|index| {
                    let base = match (index * 19 + lane * 7) % 8 {
                        0 => 0x3d80_0000u32,
                        1 => 0xbd80_0000,
                        2 => 0x3e00_0000,
                        3 => 0xbe00_0000,
                        4 => 0x3e80_0000,
                        5 => 0xbe80_0000,
                        6 => 0x3f00_0000,
                        _ => 0xbf00_0000,
                    };
                    (base | u32::try_from((index + lane) % 251).unwrap()).to_ne_bytes()
                })
                .collect::<Vec<_>>()
        };
        let f32_bytes = |words: usize, seed: usize| {
            (0..words)
                .flat_map(|index| ((((index * 13 + seed) % 17) as f32 - 8.0) / 64.0).to_ne_bytes())
                .collect::<Vec<_>>()
        };

        let conv_bytes = GdnBuffer::PackedQkvz.bytes(GdnPhase::CachedDecode) * ROWS;
        let packed_ba_bytes = GdnBuffer::PackedBa.bytes(GdnPhase::CachedDecode) * ROWS;
        let ba_view_bytes = packed_ba_bytes - 32;
        let core_bytes = GdnBuffer::Core.bytes(GdnPhase::CachedDecode) * ROWS;
        let cache_bytes = GdnBuffer::CacheArena.bytes(GdnPhase::CachedDecode);
        let state_bytes = RECURRENT_ELEMENTS * 4;
        let (
            conv,
            packed_ba,
            a_log,
            dt_bias,
            indices,
            cache,
            status,
            canonical_prior,
            baseline_canonical,
            direct_canonical,
            private_prior,
            baseline_core,
            direct_core,
        ) = {
            let mut init = service.begin_initialization(0)?;
            let conv = init.upload_bytes(&bf16_bytes(conv_bytes / 2, 1), BufferRole::Scratch)?;
            let packed_ba =
                init.upload_bytes(&bf16_bytes(packed_ba_bytes / 2, 3), BufferRole::Scratch)?;
            let a_log = init.upload_bytes(
                &f32_bytes(GdnBuffer::ALog.bytes(GdnPhase::CachedDecode) / 4, 5),
                BufferRole::Scratch,
            )?;
            let dt_bias = init.upload_bytes(
                &bf16_bytes(GdnBuffer::DtBias.bytes(GdnPhase::CachedDecode) / 2, 7),
                BufferRole::Scratch,
            )?;
            let indices = init.upload_bytes(
                &[1i32, 2, 3, 4]
                    .into_iter()
                    .flat_map(i32::to_ne_bytes)
                    .collect::<Vec<_>>(),
                BufferRole::Scratch,
            )?;
            let cache = init.upload_zeroed(cache_bytes, BufferRole::Scratch)?;
            let status = init.upload_zeroed(4, BufferRole::Scratch)?;
            let mut canonical_prior = Vec::with_capacity(ROWS);
            let mut baseline_canonical = Vec::with_capacity(ROWS);
            let mut direct_canonical = Vec::with_capacity(ROWS);
            let mut private_prior = Vec::with_capacity(ROWS);
            for lane in 0..ROWS {
                canonical_prior
                    .push(init.upload_bytes(&recurrent_bytes(lane), BufferRole::RecurrentState)?);
                baseline_canonical
                    .push(init.allocate_empty(state_bytes, BufferRole::RecurrentState)?);
                direct_canonical
                    .push(init.allocate_empty(state_bytes, BufferRole::RecurrentState)?);
                private_prior.push(init.allocate_empty(state_bytes, BufferRole::Scratch)?);
            }
            let baseline_core = init.allocate_empty(core_bytes, BufferRole::Scratch)?;
            let direct_core = init.allocate_empty(core_bytes, BufferRole::Scratch)?;
            drop(init.complete_sync().await?);
            (
                conv,
                packed_ba,
                a_log,
                dt_bias,
                indices,
                cache,
                status,
                canonical_prior,
                baseline_canonical,
                direct_canonical,
                private_prior,
                baseline_core,
                direct_core,
            )
        };
        let a = packed_ba
            .handle()
            .offset_start(32)
            .offset_end((packed_ba_bytes - 32 - ba_view_bytes) as u64);
        let beta = packed_ba.handle().offset_end(32);
        let recurrent_cache = cache.handle().offset_start(CONV_STATE_BYTES as u64);
        let prior_handles: [cubecl::server::Handle; 4] = private_prior
            .iter()
            .map(|lease| lease.handle())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let candidate_handles = prior_handles.clone();

        let import_baseline = || -> Result<()> {
            for lane in 0..ROWS {
                let slot = text::GdnRuntimeCacheSlot::new(lane + 1)?;
                let native = plane_handle(&cache, slot.recurrent_offset(), state_bytes)?;
                StateTransport::prepare(
                    &service,
                    StateCopy::RecurrentIn,
                    canonical_prior[lane].handle(),
                    native,
                    &status,
                )?
                .launch(&service)?;
            }
            Ok(())
        };
        let export_baseline = || -> Result<()> {
            for lane in 0..ROWS {
                let slot = text::GdnRuntimeCacheSlot::new(lane + 1)?;
                let native = plane_handle(&cache, slot.recurrent_offset(), state_bytes)?;
                StateTransport::prepare(
                    &service,
                    StateCopy::RecurrentOut,
                    native,
                    baseline_canonical[lane].handle(),
                    &status,
                )?
                .launch(&service)?;
            }
            Ok(())
        };
        let import_private = || -> Result<()> {
            for lane in 0..ROWS {
                StateTransport::prepare(
                    &service,
                    StateCopy::RecurrentIn,
                    canonical_prior[lane].handle(),
                    private_prior[lane].handle(),
                    &status,
                )?
                .launch(&service)?;
            }
            Ok(())
        };
        let export_private = || -> Result<()> {
            for lane in 0..ROWS {
                StateTransport::prepare(
                    &service,
                    StateCopy::RecurrentOut,
                    private_prior[lane].handle(),
                    direct_canonical[lane].handle(),
                    &status,
                )?
                .launch(&service)?;
            }
            Ok(())
        };
        let launch_baseline = || -> Result<()> {
            baseline.launch(
                &client,
                &conv.handle(),
                &a,
                &beta,
                &a_log.handle(),
                &dt_bias.handle(),
                &baseline_core.handle(),
                &recurrent_cache,
                &indices.handle(),
            )
        };
        let launch_direct = || -> Result<()> {
            direct.launch(
                &client,
                &conv.handle(),
                &a,
                &beta,
                &a_log.handle(),
                &dt_bias.handle(),
                &direct_core.handle(),
                &prior_handles,
                &candidate_handles,
                &indices.handle(),
            )
        };

        import_baseline()?;
        import_private()?;
        launch_baseline()?;
        launch_direct()?;
        export_baseline()?;
        export_private()?;
        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("private GDN decode parity sync: {error}")))?;
        let baseline_core_bytes = client
            .read_one(baseline_core.handle())
            .map_err(|error| Error::State(format!("private GDN baseline core read: {error}")))?;
        let direct_core_bytes = client
            .read_one(direct_core.handle())
            .map_err(|error| Error::State(format!("private GDN direct core read: {error}")))?;
        if baseline_core_bytes != direct_core_bytes {
            return Err(Error::State(
                "private GDN decode core differs from transport pipeline".into(),
            ));
        }
        for lane in 0..ROWS {
            let baseline_state =
                client
                    .read_one(baseline_canonical[lane].handle())
                    .map_err(|error| {
                        Error::State(format!("private GDN baseline state read: {error}"))
                    })?;
            let direct_state = client
                .read_one(direct_canonical[lane].handle())
                .map_err(|error| Error::State(format!("private GDN direct state read: {error}")))?;
            if baseline_state != direct_state {
                return Err(Error::State(format!(
                    "private GDN decode canonical state differs at lane {lane}"
                )));
            }
        }

        for _ in 0..12 {
            import_baseline()?;
            launch_baseline()?;
            export_baseline()?;
            launch_direct()?;
        }
        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("private GDN decode timing warmup: {error}")))?;
        let measure_baseline = || -> Result<f64> {
            let started = std::time::Instant::now();
            for _ in 0..REPEATS {
                import_baseline()?;
                launch_baseline()?;
                export_baseline()?;
            }
            futures_lite::future::block_on(client.sync()).map_err(|error| {
                Error::State(format!("private GDN baseline timing sync: {error}"))
            })?;
            Ok(started.elapsed().as_secs_f64() * 1e6 / REPEATS as f64)
        };
        let measure_direct = || -> Result<f64> {
            let started = std::time::Instant::now();
            for _ in 0..REPEATS {
                launch_direct()?;
            }
            futures_lite::future::block_on(client.sync()).map_err(|error| {
                Error::State(format!("private GDN direct timing sync: {error}"))
            })?;
            Ok(started.elapsed().as_secs_f64() * 1e6 / REPEATS as f64)
        };
        let baseline_fwd = measure_baseline()?;
        let direct_fwd = measure_direct()?;
        let direct_rev = measure_direct()?;
        let baseline_rev = measure_baseline()?;
        println!(
            "PRIVATE_GDN_DECODE_PASS exact=true baseline_us_fwd={baseline_fwd:.3} direct_us_fwd={direct_fwd:.3} direct_us_rev={direct_rev:.3} baseline_us_rev={baseline_rev:.3}"
        );

        drop((
            conv,
            packed_ba,
            a_log,
            dt_bias,
            indices,
            cache,
            status,
            canonical_prior,
            baseline_canonical,
            direct_canonical,
            private_prior,
            baseline_core,
            direct_core,
        ));
        client
            .memory_cleanup()
            .map_err(|error| Error::State(format!("private GDN decode cleanup: {error}")))?;
        service.close_model_namespace()?;
        Ok(())
    })
}

#[test]
#[ignore = "physical SM86 private-state cached GDN B8 parity/timing"]
fn physical_cached_decode_private_b8_matches_two_b4() -> Result<()> {
    futures_lite::future::block_on(async {
        use zerogravity_core::device_events::BufferRole;
        const ROWS: usize = 8;
        const REPEATS: usize = 180;
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("cached-gdn-private-b8")?;
        let service = diagnostic.service.clone();
        let client = service.client().clone();
        let b4 = CachedDecodePrivateBatch::prepare(4)?;
        let b8 = CachedDecodePrivateBatch::prepare(8)?;
        let bf16 = |words: usize, seed: usize| {
            (0..words)
                .flat_map(|index| {
                    [
                        0x3d80u16, 0xbd00, 0x3e00, 0xbe00, 0x3e80, 0xbe80, 0x3f00, 0xbf00,
                    ][(index * 17 + seed) % 8]
                        .to_ne_bytes()
                })
                .collect::<Vec<_>>()
        };
        let f32s = |words: usize, seed: usize| {
            (0..words)
                .flat_map(|index| ((((index * 13 + seed) % 17) as f32 - 8.0) / 64.0).to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let row_conv = GdnBuffer::PackedQkvz.bytes(GdnPhase::CachedDecode);
        let row_ba = GdnBuffer::PackedBa.bytes(GdnPhase::CachedDecode);
        let row_core = GdnBuffer::Core.bytes(GdnPhase::CachedDecode);
        let state_bytes = RECURRENT_STATE_BYTES;
        let (conv, ba, a_log, dt, indices, prior, candidate_b4, core_b4, core_b8) = {
            let mut init = service.begin_initialization(0)?;
            let conv = init.upload_bytes(&bf16(row_conv * ROWS / 2, 1), BufferRole::Scratch)?;
            let ba = init.upload_bytes(&bf16(row_ba * ROWS / 2, 3), BufferRole::Scratch)?;
            let a_log = init.upload_bytes(
                &f32s(GdnBuffer::ALog.bytes(GdnPhase::CachedDecode) / 4, 5),
                BufferRole::Scratch,
            )?;
            let dt = init.upload_bytes(
                &bf16(GdnBuffer::DtBias.bytes(GdnPhase::CachedDecode) / 2, 7),
                BufferRole::Scratch,
            )?;
            let indices = init.upload_bytes(
                &(1i32..=8).flat_map(i32::to_ne_bytes).collect::<Vec<_>>(),
                BufferRole::Scratch,
            )?;
            let mut prior = Vec::with_capacity(ROWS);
            let mut candidate_b4 = Vec::with_capacity(ROWS);
            for lane in 0..ROWS {
                prior.push(
                    init.upload_bytes(&f32s(state_bytes / 4, 11 + lane), BufferRole::Scratch)?,
                );
                candidate_b4.push(init.allocate_empty(state_bytes, BufferRole::Scratch)?);
            }
            let core_b4 = init.allocate_empty(row_core * ROWS, BufferRole::Scratch)?;
            let core_b8 = init.allocate_empty(row_core * ROWS, BufferRole::Scratch)?;
            drop(init.complete_sync().await?);
            (
                conv,
                ba,
                a_log,
                dt,
                indices,
                prior,
                candidate_b4,
                core_b4,
                core_b8,
            )
        };
        let launch_b4 = || -> Result<()> {
            for group in 0..2 {
                let first = group * 4;
                let remaining = ROWS - first - 4;
                let conv_view = conv
                    .handle()
                    .offset_start((first * row_conv) as u64)
                    .offset_end((remaining * row_conv) as u64);
                let a_view = ba
                    .handle()
                    .offset_start((first * row_ba + 32) as u64)
                    .offset_end((remaining * row_ba) as u64);
                let beta_view = ba
                    .handle()
                    .offset_start((first * row_ba) as u64)
                    .offset_end((remaining * row_ba + 32) as u64);
                let core_view = core_b4
                    .handle()
                    .offset_start((first * row_core) as u64)
                    .offset_end((remaining * row_core) as u64);
                let index_view = indices
                    .handle()
                    .offset_start((first * 4) as u64)
                    .offset_end((remaining * 4) as u64);
                let prior_handles = (first..first + 4)
                    .map(|lane| prior[lane].handle())
                    .collect::<Vec<_>>();
                let candidate_handles = (first..first + 4)
                    .map(|lane| candidate_b4[lane].handle())
                    .collect::<Vec<_>>();
                b4.launch(
                    &client,
                    &conv_view,
                    &a_view,
                    &beta_view,
                    &a_log.handle(),
                    &dt.handle(),
                    &core_view,
                    &prior_handles,
                    &candidate_handles,
                    &index_view,
                )?;
            }
            Ok(())
        };
        let prior_b8 = prior.iter().map(|lease| lease.handle()).collect::<Vec<_>>();
        let candidate_b8_handles = prior_b8.clone();
        let launch_b8 = || {
            b8.launch(
                &client,
                &conv.handle(),
                &ba.handle().offset_start(32),
                &ba.handle().offset_end(32),
                &a_log.handle(),
                &dt.handle(),
                &core_b8.handle(),
                &prior_b8,
                &candidate_b8_handles,
                &indices.handle(),
            )
        };
        launch_b4()?;
        launch_b8()?;
        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("private GDN B8 parity sync: {error}")))?;
        if client
            .read_one(core_b4.handle())
            .map_err(|error| Error::State(format!("private GDN B4 core read: {error}")))?
            != client
                .read_one(core_b8.handle())
                .map_err(|error| Error::State(format!("private GDN B8 core read: {error}")))?
        {
            return Err(Error::State(
                "private GDN B8 core differs from two B4 launches".into(),
            ));
        }
        for lane in 0..ROWS {
            if client
                .read_one(candidate_b4[lane].handle())
                .map_err(|error| Error::State(format!("private GDN B4 state read: {error}")))?
                != client.read_one(prior[lane].handle()).map_err(|error| {
                    Error::State(format!("private GDN B8 in-place state read: {error}"))
                })?
            {
                return Err(Error::State(format!(
                    "private GDN B8 state differs at lane {lane}"
                )));
            }
        }
        for _ in 0..20 {
            launch_b4()?;
            launch_b8()?;
        }
        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("private GDN B8 warmup: {error}")))?;
        let measure = |launch: &dyn Fn() -> Result<()>| -> Result<f64> {
            let started = std::time::Instant::now();
            for _ in 0..REPEATS {
                launch()?;
            }
            futures_lite::future::block_on(client.sync())
                .map_err(|error| Error::State(format!("private GDN B8 timing sync: {error}")))?;
            Ok(started.elapsed().as_secs_f64() * 1e6 / REPEATS as f64)
        };
        let b4_forward = measure(&launch_b4)?;
        let b8_forward = measure(&launch_b8)?;
        let b8_reverse = measure(&launch_b8)?;
        let b4_reverse = measure(&launch_b4)?;
        println!(
            "PRIVATE_GDN_B8_PASS exact=true two_b4_us_fwd={b4_forward:.3} b8_us_fwd={b8_forward:.3} b8_us_rev={b8_reverse:.3} two_b4_us_rev={b4_reverse:.3}"
        );
        drop((
            conv,
            ba,
            a_log,
            dt,
            indices,
            prior,
            candidate_b4,
            core_b4,
            core_b8,
        ));
        client
            .memory_cleanup()
            .map_err(|error| Error::State(format!("private GDN B8 cleanup: {error}")))?;
        service.close_model_namespace()?;
        Ok(())
    })
}
