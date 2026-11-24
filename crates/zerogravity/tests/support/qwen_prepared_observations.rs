//! Bounded real-source observed execution/refusal on the original provider.
#[allow(dead_code)]
#[path = "qwen_source.rs"]
mod source;
#[allow(dead_code)]
#[path = "qwen_source_import.rs"]
mod source_import;
use std::sync::Arc;
use zerogravity::{
    Error, Result,
    accelerator::program::{InvocationPhase, PlanOperation, ValueId},
    gpu::{
        Device, ResidentOutput, ResidentTextPolicy, ResidentTextPreparation, ResidentTextProvider,
        ResidentTextPublication, TextFrame, TextFrameKind, TextObserver,
    },
    models::prepare_executable,
    provider::ObservationSupport,
    qwen35::source::binding::OriginalWindowModelSource,
};

struct Observer {
    expected: Vec<(ValueId, usize)>,
    base: usize,
    rows: usize,
    cursor: usize,
    offset: usize,
    logits: Option<ValueId>,
    logits_bytes: Vec<u8>,
}
impl TextObserver for Observer {
    fn wants(&self, kind: TextFrameKind) -> bool {
        matches!(kind, TextFrameKind::Value(_))
    }
    fn completed(&mut self, frame: TextFrame, actual: &[u8]) -> Result<()> {
        let Some(&(id, words)) = self.expected.get(self.cursor) else {
            return Err(Error::State("unexpected producer observation".into()));
        };
        if frame.kind != TextFrameKind::Value(id)
            || frame.base != self.base
            || frame.rows != self.rows
            || frame.total_words != words
            || frame.offset_words != self.offset
            || actual.is_empty()
            || !actual.len().is_multiple_of(4)
        {
            return Err(Error::State(
                "actual producer observation coordinates differ".into(),
            ));
        }
        self.offset += actual.len() / 4;
        if self.offset > words {
            return Err(Error::State(
                "producer observation exceeds original value".into(),
            ));
        }
        if Some(id) == self.logits {
            self.logits_bytes.extend_from_slice(actual);
        }
        if self.offset == words {
            self.offset = 0;
            self.cursor += 1;
        }
        Ok(())
    }
}

#[test]
#[ignore = "approved Qwen 0.8B original cache and isolated native GPU"]
fn prepared_source_observation_contract() -> Result<()> {
    futures_lite::future::block_on(async {
        // Existing canonical source loader; no downloaded/repinned fixtures.
        let loaded = source::Loaded::open_with_chunk(
            "08",
            &source_import::canonical_root("08")?,
            true,
            1024 * 1024,
        )?;
        let original = OriginalWindowModelSource::from_owner(loaded.model.owner().clone())?;
        #[cfg(feature = "hip")]
        let (seal, device, policy, support) = {
            let budget = std::env::var("ZG_QWEN_HIP_TOTAL_DEVICE_BYTES")
                .expect("approved explicit HIP budget")
                .parse::<u64>()
                .map_err(|_| Error::Malformed("HIP device budget".into()))?;
            (
                original.bind_hip_native_solve_forward_source()?,
                Device::Hip(Default::default()),
                ResidentTextPolicy::Hip(zerogravity::gpu::HipSourceBudget::new(budget)?),
                ObservationSupport::AtProducer,
            )
        };
        #[cfg(all(feature = "cuda", not(feature = "hip")))]
        let (seal, device, policy, support) = (
            original.bind_cuda_forward_source()?,
            Device::Cuda(Default::default()),
            ResidentTextPolicy::Cuda {
                provider_bytes: 4608 * 1024 * 1024,
                device_bytes: 5120 * 1024 * 1024,
                state: zerogravity::gpu::CudaSourceStatePolicy::Contiguous,
            },
            ObservationSupport::None,
        );
        let plan = seal.build_plan(512)?;
        let provider = ResidentTextProvider {
            device,
            policy,
            source_owner: loaded.clone(),
            max_sessions: 1,
            cohort_width: 0,
            cancelled: None,
        };
        #[cfg(feature = "hip")]
        {
            use zerogravity::resource_arbiter::{
                AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter,
                ResourceOwner,
            };
            let ResidentTextPolicy::Hip(budget) = &provider.policy else {
                unreachable!()
            };
            let Device::Hip(selected) = &provider.device else {
                unreachable!()
            };
            let observed = zerogravity::gpu::observe_hip_device(selected.clone())
                .expect("original HIP device observation");
            let key =
                PhysicalDeviceKey::new("pci", &observed.physical.pci_address.unwrap().to_string())?;
            let host = PhysicalDeviceKey::new("host", "system-ram")?;
            let arbiter = ResourceArbiter::process_shared();
            arbiter.register_matching(key.clone(), budget.physical_admission_bytes())?;
            let blocker_bytes =
                budget.physical_admission_bytes() - budget.total_device_bytes() as u64 + 1;
            let blocker = arbiter.admit(
                &key,
                ResourceOwner::new("admission-test", "pressure")?,
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: blocker_bytes,
                }],
            )?;
            let before = arbiter.snapshot(&host)?.reserved;
            assert!(matches!(
                prepare_executable(plan.model_program()?, &provider).await,
                Err(Error::Resource(_))
            ));
            assert_eq!(arbiter.snapshot(&host)?.reserved, before);
            let card = arbiter.snapshot(&key)?;
            assert_eq!(card.reserved, blocker_bytes);
            assert!(!card.fenced);
            drop(blocker);
            println!("PRE_CLIENT_CAPACITY_REFUSAL_PASS");
        }
        let executable = prepare_executable(plan.model_program()?, &provider).await?;
        let prepared = executable
            .prepared_text()
            .expect("actual prepared text contract");
        assert_eq!(prepared.observations(), support);
        assert!(Arc::ptr_eq(prepared.model().program(), plan.plan()));
        let model = executable.into_prepared();
        let mut session = model.session().await?;
        let mut observed = 0;
        for (base, emit) in [(0, false), (2, true)] {
            let invocation =
                plan.bind_invocation(InvocationPhase::Prefill, base, vec![11, 353].into(), emit)?;
            // Expected extents come from the original materialized operator
            // descriptors, independently of the static observation port adapter.
            let request = invocation.materialize_plan()?;
            let nodes = if emit {
                request.nodes()
            } else {
                request.state_nodes()
            };
            let mut expected = Vec::new();
            for node in nodes {
                let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
                    unreachable!()
                };
                expected.extend(
                    node.outputs
                        .iter()
                        .copied()
                        .zip(descriptor.output_elements()?),
                );
            }
            let mut observer = Observer {
                expected,
                base,
                rows: 2,
                cursor: 0,
                offset: 0,
                logits: emit.then(|| request.logits().unwrap()),
                logits_bytes: Vec::new(),
            };
            let output = if emit {
                ResidentOutput::Logits
            } else {
                ResidentOutput::StateOnly
            };
            let before = session.binding();
            let result = session
                .prepare_observed(
                    invocation.bound_invocation(),
                    InvocationPhase::Prefill,
                    output,
                    None,
                    &mut observer,
                )
                .await;
            let pending = if support == ObservationSupport::AtProducer {
                let ResidentTextPreparation::Ready(pending) = result? else {
                    return Err(Error::State("observed source call refused".into()));
                };
                assert_eq!(observer.cursor, observer.expected.len());
                assert_eq!(observer.offset, 0);
                if emit {
                    assert!(
                        observer
                            .logits_bytes
                            .chunks_exact(4)
                            .map(|word| { u32::from_le_bytes(word.try_into().unwrap()) })
                            .eq(pending.readback().logits.iter().map(|v| v.to_bits()))
                    );
                } else {
                    assert!(observer.logits_bytes.is_empty());
                    assert!(pending.readback().logits.is_empty());
                }
                observed += observer.cursor;
                pending
            } else {
                assert!(matches!(result, Err(Error::Unsupported(_))));
                assert_eq!(observer.cursor, 0);
                // Prove the refusal did not fence or consume the original root.
                let ResidentTextPreparation::Ready(pending) = session
                    .prepare(
                        invocation.bound_invocation(),
                        InvocationPhase::Prefill,
                        output,
                        None,
                    )
                    .await?
                else {
                    return Err(Error::State(
                        "ordinary call failed after observer refusal".into(),
                    ));
                };
                pending
            };
            assert_eq!(session.binding(), before);
            let ResidentTextPublication::Committed(readback) = session.commit(pending)? else {
                return Err(Error::State("prepared source publication refused".into()));
            };
            assert_eq!(session.binding().frontier(), (base + 2) as u64);
            if emit {
                assert_eq!(
                    readback.logits.len(),
                    plan.model_profile().vocabulary_size()
                );
            }
        }
        drop(session);
        model.shutdown().await?;
        drop((provider, plan, seal, original));
        loaded.close()?;
        println!(
            "PREPARED_SOURCE_OBSERVATION_PASS support={support:?} observed_values={observed} state_only_and_logits=true"
        );
        Ok(())
    })
}
