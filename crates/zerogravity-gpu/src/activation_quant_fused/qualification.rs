//! Physical differential operator fixtures, never model or speed evidence.
use super::*;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use std::sync::Arc;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};

const ROWS: usize = 8;
const CASES: [&str; 17] = [
    "signed_zero",
    "first_negative_tie",
    "first_positive_tie",
    "integer_ties",
    "subnormal",
    "minimum_normal",
    "half_subnormal_edge",
    "half_scale_edge",
    "half_scale_overflow_edge",
    "half_midpoint",
    "unused_sum_overflow",
    "scale_overflow",
    "dense_finite",
    "nonfinite_input",
    "nonfinite_static",
    "irregular_blocks",
    "single_block",
];

fn inputs(case: &str, width: usize, batch: usize, blocks: usize) -> Vec<f32> {
    (0..batch * blocks * width)
        .map(|i| {
            let at = i % width;
            let request = i / (blocks * width);
            let bits = match case {
                "signed_zero" => {
                    if (at + request).is_multiple_of(2) {
                        0
                    } else {
                        0x80000000
                    }
                }
                "first_negative_tie" => {
                    if at == 0 {
                        (-127.0f32).to_bits()
                    } else {
                        127.0f32.to_bits()
                    }
                }
                "first_positive_tie" => {
                    if at == 0 {
                        127.0f32.to_bits()
                    } else {
                        (-127.0f32).to_bits()
                    }
                }
                "integer_ties" => [127.0f32, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
                    [(at + request) % 8]
                    .to_bits(),
                "subnormal" => {
                    [1, 0x80000001, 0, 0x80000000, 0x007fffff, 0x807fffff][(at + request) % 6]
                }
                "minimum_normal" => {
                    [0x00800000, 0x80800000, 0x00800001, 0x80800001][(at + request) % 4]
                }
                "half_subnormal_edge" => {
                    let edge = (127.0f32 * 2.0f32.powi(-24)).to_bits();
                    [edge - 1, edge, edge + 1, edge | 0x80000000][(at + request) % 4]
                }
                "half_scale_edge" => {
                    let edge = (65504.0f32 * 127.0).to_bits();
                    [edge - 1, edge, edge + 1, edge | 0x80000000][(at + request) % 4]
                }
                "half_scale_overflow_edge" => {
                    let edge = (65520.0f32 * 127.0).to_bits();
                    [edge - 1, edge, edge + 1, edge | 0x80000000][(at + request) % 4]
                }
                "half_midpoint" => {
                    let edge = ((1.0f32 + 2.0f32.powi(-11)) * 127.0).to_bits();
                    [edge - 1, edge, edge + 1, edge | 0x80000000][(at + request) % 4]
                }
                "unused_sum_overflow" => 65504.0f32.to_bits(),
                "scale_overflow" => f32::MAX.to_bits(),
                "nonfinite_input" => [
                    0x7fc12345, 0x7f800001, 0x7f800000, 0xff800000, 0x3f800000, 0xbf800000,
                ][at % 6],
                "dense_finite" | "nonfinite_static" | "irregular_blocks" | "single_block" => {
                    let mut word = (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x5135791d);
                    word ^= word >> 13;
                    if word & 0x7f800000 == 0x7f800000 {
                        word ^= 0x00800000;
                    }
                    // Broad finite bit patterns, including both signs and tiny words.
                    word
                }
                _ => unreachable!("fixture case"),
            };
            f32::from_bits(bits)
        })
        .collect()
}

struct Workspace {
    parameters: ResidentLease,
    scaled: ResidentLease,
    codes: ResidentLease,
    scales: ResidentLease,
    offsets: ResidentLease,
    block_status: ResidentLease,
    row_status: ResidentLease,
}

impl Workspace {
    fn allocate(
        init: &mut ResidentInitialization<'_>,
        l: &ActivationLayout,
        batch: usize,
    ) -> Result<Self> {
        let empty = |init: &mut ResidentInitialization<'_>, words: usize| {
            init.allocate_empty(words * 4, BufferRole::Scratch)
        };
        Ok(Self {
            parameters: empty(init, batch * l.blocks * 4)?,
            scaled: empty(init, batch * l.columns)?,
            codes: empty(init, batch * l.columns)?,
            scales: empty(init, batch * l.composite_count)?,
            // The unused dummy is not a source result. Initialize both copies
            // with the same sentinel solely to detect an unintended write.
            offsets: if l.with_bias {
                empty(init, batch * l.composite_count)?
            } else {
                init.upload_bytes(&0xa5a55a5au32.to_ne_bytes(), BufferRole::Scratch)?
            },
            block_status: empty(init, batch * l.blocks)?,
            row_status: empty(init, batch * l.rows)?,
        })
    }
    fn leases(&self) -> [ResidentLease; 7] {
        [
            self.parameters.clone(),
            self.scaled.clone(),
            self.codes.clone(),
            self.scales.clone(),
            self.offsets.clone(),
            self.block_status.clone(),
            self.row_status.clone(),
        ]
    }
    fn bindings<'a>(input: &'a Handle, handles: &'a [Handle; 4]) -> BlockBindings<'a> {
        BlockBindings {
            input,
            parameters: &handles[0],
            scaled: &handles[1],
            codes: &handles[2],
            block_status: &handles[3],
        }
    }
    fn block_handles(&self) -> [Handle; 4] {
        [
            self.parameters.handle(),
            self.scaled.handle(),
            self.codes.handle(),
            self.block_status.handle(),
        ]
    }
}

fn view(handle: &Handle, start: usize, words: usize) -> Handle {
    handle
        .clone()
        .offset_start((start * 4) as u64)
        .offset_end(handle.size_in_used() - ((start + words) * 4) as u64)
}

/// Reference block stages are the original physical implementation unchanged.
fn launch_reference(
    client: &Client,
    l: &ActivationLayout,
    batch: usize,
    input: &Handle,
    w: &Workspace,
) -> usize {
    let blocks = batch * l.blocks;
    let columns = batch * l.columns;
    let parameters = blocks * 4;
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    let threads = CubeDim::new_1d(32);
    // SAFETY: the same canonical geometry is preflighted before submission;
    // original owner pins every source and workspace throughout completion.
    unsafe {
        activation_quant::block_parameters::launch_unchecked(
            client,
            grid(blocks),
            threads,
            BufferArg::from_raw_parts(input.clone(), columns),
            BufferArg::from_raw_parts(w.parameters.handle(), parameters),
            BufferArg::from_raw_parts(w.block_status.handle(), blocks),
            blocks,
            l.block_width,
            l.signed_scale,
        );
        activation_quant::block_scales::launch_unchecked(
            client,
            grid(blocks),
            threads,
            BufferArg::from_raw_parts(w.parameters.handle(), parameters),
            BufferArg::from_raw_parts(w.block_status.handle(), blocks),
            blocks,
            l.signed_scale,
        );
        activation_quant::scaled_inputs::launch_unchecked(
            client,
            grid(columns),
            threads,
            BufferArg::from_raw_parts(input.clone(), columns),
            BufferArg::from_raw_parts(w.parameters.handle(), parameters),
            BufferArg::from_raw_parts(w.scaled.handle(), columns),
            columns,
            l.block_width,
        );
        activation_quant::codes::launch_unchecked(
            client,
            grid(columns),
            threads,
            BufferArg::from_raw_parts(w.scaled.handle(), columns),
            BufferArg::from_raw_parts(w.codes.handle(), columns),
            columns,
        );
        if !l.signed_scale {
            activation_quant::sum_products::launch_unchecked(
                client,
                grid(blocks),
                threads,
                BufferArg::from_raw_parts(w.codes.handle(), columns),
                BufferArg::from_raw_parts(w.parameters.handle(), parameters),
                blocks,
            );
            activation_quant::sum_halves::launch_unchecked(
                client,
                grid(blocks),
                threads,
                BufferArg::from_raw_parts(w.parameters.handle(), parameters),
                BufferArg::from_raw_parts(w.block_status.handle(), blocks),
                blocks,
                l.with_bias,
            );
        }
    }
    if l.signed_scale { 4 } else { 6 }
}

/// Reuse accepted B1 composites for each independent original row. Both
/// preparation variants feed exactly the same physical consumer kernel.
fn launch_composites(
    client: &Client,
    l: &ActivationLayout,
    batch: usize,
    scales: &Handle,
    biases: &Handle,
    w: &Workspace,
) -> usize {
    let offset_count = if l.with_bias { l.composite_count } else { 1 };
    for request in 0..batch {
        let parameters = view(&w.parameters.handle(), request * l.blocks * 4, l.blocks * 4);
        let output = view(
            &w.scales.handle(),
            request * l.composite_count,
            l.composite_count,
        );
        let offsets = if l.with_bias {
            view(
                &w.offsets.handle(),
                request * l.composite_count,
                l.composite_count,
            )
        } else {
            w.offsets.handle()
        };
        let status = view(&w.row_status.handle(), request * l.rows, l.rows);
        // SAFETY: exact original per-request views, shared immutable source
        // metadata, all whole leases retained by the one armed submission.
        unsafe {
            activation_quant::composites::launch_unchecked(
                client,
                CubeCount::Static((l.rows as u32).div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(scales.clone(), l.composite_count),
                BufferArg::from_raw_parts(biases.clone(), offset_count),
                BufferArg::from_raw_parts(parameters, l.blocks * 4),
                BufferArg::from_raw_parts(output, l.composite_count),
                BufferArg::from_raw_parts(offsets, offset_count),
                BufferArg::from_raw_parts(status, l.rows),
                l.blocks,
                l.rows,
                l.with_bias,
            );
        }
    }
    batch
}

async fn compare_case(
    service: &Arc<ResidentService>,
    kind: QuantKind,
    batch: usize,
    case: &str,
) -> Result<usize> {
    let width = if kind == QuantKind::Q6K { 256 } else { 32 };
    let blocks = match case {
        "irregular_blocks" => 33,
        "single_block" => 1,
        _ => 2,
    };
    let l = ActivationLayout::new(kind, blocks * width, ROWS)?;
    let values = inputs(case, width, batch, blocks);
    let source_scales = (0..l.composite_count)
        .map(|i| {
            if case == "nonfinite_static" {
                f32::INFINITY
            } else {
                [0.5f32, -0.125, 65504.0, 0.0, -0.0][i % 5]
            }
        })
        .collect::<Vec<_>>();
    let source_biases = if l.with_bias {
        (0..l.composite_count)
            .map(|i| [0.25f32, -0.5, 0.0, -0.0][i % 4])
            .collect::<Vec<_>>()
    } else {
        vec![0.0]
    };
    let input_bytes = (values.len() + source_scales.len() + source_biases.len()) * 4;
    let output_words = batch
        * (l.blocks * 4 + l.columns * 2 + l.composite_count + l.blocks + l.rows)
        + if l.with_bias {
            batch * l.composite_count
        } else {
            1
        };
    let output_bytes = output_words * 4 * 2;
    let required = input_bytes + output_bytes;
    if required > service.remaining_capacity()? {
        return Err(Error::Limit("fused preparation fixture capacity".into()));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[
            AllocationRequest {
                kind: AllocationClass::ImmutableWeights,
                bytes: input_bytes as u64,
            },
            AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: output_bytes as u64,
            },
        ],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    if required > service.remaining_capacity()? {
        return Err(Error::Limit(
            "fused preparation queued capacity changed".into(),
        ));
    }
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let input = init.upload_bytes(f32::as_bytes(&values), BufferRole::ImmutableWeights)?;
    let static_scales =
        init.upload_bytes(f32::as_bytes(&source_scales), BufferRole::ImmutableWeights)?;
    let biases = init.upload_bytes(f32::as_bytes(&source_biases), BufferRole::ImmutableWeights)?;
    let reference = Workspace::allocate(&mut init, &l, batch)?;
    let candidate = Workspace::allocate(&mut init, &l, batch)?;
    let initialized = init.complete_sync().await?;
    if initialized.outputs().len() != 17 {
        return Err(Error::State(
            "fused preparation initialization coverage".into(),
        ));
    }
    drop(initialized);
    let client = service.client();
    let input_handle = input.handle();
    let handles = candidate.block_handles();
    let binding = Workspace::bindings(&input_handle, &handles);
    validate(client, &l, batch, &binding)?;
    // Negative bindings are rejected before this fixture's GPU math begins.
    let alias = BlockBindings {
        codes: &input_handle,
        ..Workspace::bindings(&input_handle, &handles)
    };
    let short = candidate.codes.handle().offset_end(4);
    let short_binding = BlockBindings {
        codes: &short,
        ..Workspace::bindings(&input_handle, &handles)
    };
    if validate(client, &l, batch, &alias).is_ok()
        || validate(client, &l, batch, &short_binding).is_ok()
        || validate(client, &l, 3, &binding).is_ok()
    {
        return Err(Error::State(
            "fused preparation accepted invalid preflight".into(),
        ));
    }
    let workspaces = reference
        .leases()
        .into_iter()
        .chain(candidate.leases())
        .collect::<Vec<_>>();
    let outputs = workspaces
        .iter()
        .enumerate()
        .filter(|(index, _)| l.with_bias || index % 7 != 4)
        .map(|(_, lease)| lease.clone())
        .collect::<Vec<_>>();
    let mut accesses = [input.clone(), static_scales.clone(), biases.clone()]
        .into_iter()
        .map(|l| (l, DeviceAccess::Read))
        .collect::<Vec<_>>();
    accesses.extend(workspaces.iter().enumerate().map(|(index, lease)| {
        // Unused offset dummies are initialized sentinels, not new writes.
        let access = if !l.with_bias && index % 7 == 4 {
            DeviceAccess::Read
        } else {
            DeviceAccess::Write
        };
        (lease.clone(), access)
    }));
    let readbacks = workspaces
        .iter()
        .map(|l| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let flight = service.submit(0, &accesses)?;
    let reference_dispatches = launch_reference(client, &l, batch, &input_handle, &reference)
        + launch_composites(
            client,
            &l,
            batch,
            &static_scales.handle(),
            &biases.handle(),
            &reference,
        );
    let candidate_dispatches = launch(client, &l, batch, &binding)?
        + launch_composites(
            client,
            &l,
            batch,
            &static_scales.handle(),
            &biases.handle(),
            &candidate,
        );
    let (ready, bytes) = flight.complete_with_readback(&outputs, &readbacks, |read| {
        if read.len() != 14 { return Err(Error::State("fused preparation readback count".into())); }
        for (index, name) in ["parameters", "scaled", "codes", "scales", "offsets", "block_status", "row_status"].iter().enumerate() {
            if read[index][..] != read[index + 7][..] {
                let (word, (a, b)) = read[index].chunks_exact(4).zip(read[index + 7].chunks_exact(4)).enumerate().find(|(_, (a,b))| a != b).ok_or_else(|| Error::State("fused activation readback length mismatch".into()))?;
                let expected = u32::from_ne_bytes(a.try_into().expect("reference word"));
                let actual = u32::from_ne_bytes(b.try_into().expect("candidate word"));
                return Err(Error::State(format!("fused activation {kind:?} B{batch} {case} {name}[{word}] {actual:08x} != original physical {expected:08x}")));
            }
        }
        for flags in read[12].chunks_exact(4) {
            let flags = u32::from_ne_bytes(flags.try_into().expect("status word"));
            if case == "nonfinite_input" && flags & activation_quant::INPUT_NONFINITE == 0 {
                return Err(Error::State("fused activation omitted input finite status".into()));
            }
            if case == "unused_sum_overflow" {
                let expected = if l.with_bias { activation_quant::SUM_NONFINITE } else { 0 };
                if flags != expected { return Err(Error::State("fused activation sum codec status".into())); }
            }
        }
        if !l.with_bias && read[11][..] != 0xa5a55a5au32.to_ne_bytes() {
            return Err(Error::State("fused activation wrote unused offset dummy".into()));
        }
        if case == "nonfinite_static" && read[13].chunks_exact(4).any(|word| u32::from_ne_bytes(word.try_into().expect("status word")) != activation_quant::COMPOSITE_NONFINITE) {
            return Err(Error::State("fused activation omitted composite finite status".into()));
        }
        Ok(())
    }).await?;
    slot.finish()?;
    let words = bytes.iter().map(|b| b.len() / 4).sum::<usize>();
    if reference_dispatches - candidate_dispatches != if l.signed_scale { 3 } else { 5 } {
        return Err(Error::State(
            "fused activation actual dispatch accounting".into(),
        ));
    }
    drop(ready);
    Ok(words)
}

#[test]
fn fixture_bits_cover_finite_boundaries_and_real_fault_words() {
    for width in [32, 256] {
        let zeros = inputs("signed_zero", width, 8, 2);
        assert!(zeros.iter().any(|x| x.to_bits() == 0));
        assert!(zeros.iter().any(|x| x.to_bits() == 0x80000000));
        assert_eq!(
            inputs("first_negative_tie", width, 2, 2)[0].to_bits(),
            (-127.0f32).to_bits()
        );
        assert_eq!(
            inputs("first_positive_tie", width, 2, 2)[0].to_bits(),
            127.0f32.to_bits()
        );
        for case in CASES {
            let values = inputs(case, width, 8, 2);
            assert_eq!(values.len(), width * 16);
            assert_eq!(
                values.iter().any(|x| !x.is_finite()),
                case == "nonfinite_input"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "release CUDA physical activation preparation differential qualification"]
fn cuda_all_codecs_b1_b2_b4_b8_exact_original_preparation() {
    if cfg!(debug_assertions) {
        panic!("physical activation preparation requires --release");
    }
    futures_lite::future::block_on(async {
        let service = ResidentService::new(
            cubecl::Device::Cuda(cubecl::device::CudaDevice::new(0)),
            SourceIdentity::for_new_allocation()?,
            ResourceOwner::new("activation-preparation", "physical-differential")?,
            64 * 1024 * 1024,
            5120 * 1024 * 1024,
        )?;
        let key = service.physical_key();
        let mut words = 0usize;
        let mut cases = 0usize;
        for kind in [QuantKind::Q4_0, QuantKind::Q4_1, QuantKind::Q8_0, QuantKind::Q6K] {
            for batch in [1, 2, 4, 8] {
                for case in CASES {
                    words += compare_case(&service, kind, batch, case).await?;
                    cases += 1;
                    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
                        return Err(Error::State("fused preparation leaked original leases/events".into()));
                    }
                }
            }
        }
        drop(service);
        if ResourceArbiter::process_shared().snapshot(&key)?.reserved != 0 {
            return Err(Error::State("fused preparation physical envelope remained reserved".into()));
        }
        println!("CUDA_FUSED_ACTIVATION_PREPARATION_PASS codecs=4 batches=1,2,4,8 cases={cases} compared_words={words} all_intermediates_exact=1 original_queue_and_leases=1 published_roots=0 performance_qualified=0");
        Ok::<(), Error>(())
    }).expect("physical activation preparation differential qualification");
}
