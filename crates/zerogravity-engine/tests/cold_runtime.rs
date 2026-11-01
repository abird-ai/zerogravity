use zerogravity_core::{
    physical::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
    run_policy::SourceSelector,
};
use zerogravity_engine::{Error, Runtime, TaskInput, TaskRequest};
use zerogravity_engine::{catalog::ModelLoadRequest, scheduler::SchedulerLimits};

#[test]
fn cold_runtime_neither_loads_sources_nor_invents_execution() {
    let runtime = Runtime::builder().build().unwrap();
    assert_eq!(runtime.models().list().unwrap().len(), 0);
    assert_eq!(runtime.status().execution_adapters, 0);
    assert!(matches!(
        runtime.models().load(&ModelLoadRequest {
            source: SourceSelector::Local("/no-such-fabric-model".into()),
        }),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        runtime.run(&TaskRequest {
            model: "unloaded".into(),
            input: TaskInput::Text("hello".into()),
            max_output: 1.try_into().unwrap(),
        }),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        zerogravity_engine::router::resolve(&runtime, "unloaded"),
        Err(Error::Unsupported(_))
    ));
}
#[test]
fn clones_close_together_but_other_runtime_and_original_ledger_survive() {
    let a = Runtime::builder().build().unwrap();
    let clone = a.clone();
    let b = Runtime::builder().build().unwrap();
    let direct = ResourceArbiter::process_shared();
    let key = PhysicalDeviceKey::new("host-test", "fabric-shared-ledger").unwrap();
    direct.register(key.clone(), 64).unwrap();
    assert!(a.resources().is_process_shared());
    let lease = a
        .resources()
        .admit(
            &key,
            ResourceOwner::new("fabric-test", "source-free").unwrap(),
            &[AllocationRequest {
                kind: AllocationClass::ImmutableWeights,
                bytes: 32,
            }],
        )
        .unwrap();
    assert_eq!(b.resources().snapshot(&key).unwrap().reserved, 32);
    clone.shutdown().unwrap();
    assert!(a.status().closed);
    assert!(a.models().list().is_err());
    assert!(!b.status().closed);
    assert_eq!(direct.snapshot(&key).unwrap().reserved, 32);
    drop(lease);
    assert_eq!(direct.snapshot(&key).unwrap().reserved, 0);
    assert!(matches!(
        a.models().load(&ModelLoadRequest {
            source: SourceSelector::Local("/not-opened-after-shutdown".into()),
        }),
        Err(Error::State(_))
    ));
    assert!(matches!(
        zerogravity_engine::router::resolve(&a, "unloaded"),
        Err(Error::State(_))
    ));
    assert!(matches!(
        a.run(&TaskRequest {
            model: "unloaded".into(),
            input: TaskInput::Text("test".into()),
            max_output: 1.try_into().unwrap(),
        }),
        Err(Error::State(_))
    ));
    a.shutdown().unwrap();
}
#[test]
fn invalid_bounds_and_request_refuse_before_future_execution() {
    assert!(
        Runtime::builder()
            .limits(SchedulerLimits {
                max_active: 0,
                max_queued: Some(1),
                max_input_bytes: 1,
            })
            .build()
            .is_err()
    );
    let runtime = Runtime::builder()
        .limits(SchedulerLimits {
            max_active: 1,
            max_queued: Some(0),
            max_input_bytes: 1,
        })
        .build()
        .unwrap();
    assert!(matches!(
        runtime.run(&TaskRequest {
            model: "unloaded".into(),
            input: TaskInput::Text("too large".into()),
            max_output: 1.try_into().unwrap(),
        }),
        Err(Error::Limit(_))
    ));
}
