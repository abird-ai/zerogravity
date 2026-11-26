#![cfg(all(feature = "acquisition", target_os = "linux"))]
//! Public facade only; real bounded cache publication, no model/network input.
use std::{
    collections::BTreeMap,
    io::Cursor,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use zerogravity::artifacts::acquisition::{
    Provider, PullLimits, PullRequest, Selection,
    http::{HttpRequest, HttpResponse, HttpTransport},
};
use zerogravity::artifacts::cache::{CacheOptions, CacheStore, CommitOutcome, TransferAdmission};
use zerogravity::artifacts::{ComponentDocument, HubSelection, PullMode};
use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
use zerogravity::{Error, Result, Runtime};
const PIN: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "zg-sdk-artifacts-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!path.exists());
        Self(path)
    }
    fn store(&self) -> CacheStore {
        CacheStore::open(CacheOptions::new(Some(self.0.clone()), 64 * 1024 * 1024)).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        if self.0.exists() {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
}
fn admission() -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("fixture", "transfer").unwrap();
    arbiter
        .register(resource.clone(), 128 * 1024 * 1024)
        .unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("fixture", "sdk").unwrap(),
    }
}
fn request() -> PullRequest {
    PullRequest {
        provider: Provider::HuggingFace,
        repository: "fixture/repo".into(),
        revision: PIN.into(),
        selection: Selection::Files {
            paths: vec!["LICENSE".into()],
            subfolder: None,
        },
    }
}
#[derive(Default)]
struct Transport {
    metadata: AtomicUsize,
    payload: AtomicUsize,
}
impl HttpTransport for Transport {
    fn get(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        let bytes = if request.url
            == format!("https://huggingface.co/api/models/fixture/repo/revision/{PIN}")
        {
            self.metadata.fetch_add(1, Ordering::Relaxed);
            serde_json::to_vec(&serde_json::json!({"sha":PIN})).unwrap()
        } else if request.url
            == format!(
                "https://huggingface.co/api/models/fixture/repo/tree/{PIN}?recursive=true&expand=false&limit=1000"
            )
        {
            self.metadata.fetch_add(1, Ordering::Relaxed);
            serde_json::to_vec(&serde_json::json!([{"type":"file","path":"LICENSE","size":3}]))
                .unwrap()
        } else if request.url
            == format!("https://huggingface.co/fixture/repo/resolve/{PIN}/LICENSE")
        {
            self.payload.fetch_add(1, Ordering::Relaxed);
            b"abc".to_vec()
        } else {
            panic!("unexpected fixture request: {}", request.url)
        };
        Ok(HttpResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: Box::new(Cursor::new(bytes)),
        })
    }
}
#[test]
fn sdk_hub_preserves_real_publication_pins_progress_and_offline_validation() {
    let fixture = Fixture::new();
    let store = fixture.store();
    let runtime = Runtime::builder().build().unwrap();
    let request = request();
    let hub = runtime
        .artifacts()
        .hub(
            HubSelection::Files(&request),
            PullLimits::default(),
            None,
            None,
        )
        .unwrap();
    assert_eq!(hub.reference(), format!("huggingface/fixture/repo/{PIN}"));
    let transport = Transport::default();
    let admission = admission();
    let cancel = Arc::new(AtomicBool::new(false));
    let prepared = hub
        .resolve(&transport, &admission, &cancel, PullMode::Transfer)
        .unwrap();
    assert_eq!(prepared.plan().source().revision(), PIN);
    let mut completed = 0;
    let receipt = prepared
        .execute(&store, &transport, &admission, &cancel, &mut |event| {
            if event.file_complete() {
                completed += 1;
                assert_eq!(event.total_bytes(), 3);
            }
        })
        .unwrap();
    assert!(matches!(receipt.outcome(), CommitOutcome::Durable(_)));
    assert_eq!(completed, 1);
    assert_eq!(transport.metadata.load(Ordering::Relaxed), 2);
    assert_eq!(transport.payload.load(Ordering::Relaxed), 1);
    let lease = hub.cached(&store, &cancel).unwrap();
    assert_eq!(lease.manifest().source().revision(), PIN);
    assert_eq!(lease.manifest().total_bytes(), 3);
    let mut wrong = request.clone();
    wrong.repository = "other/repo".into();
    let wrong = runtime
        .artifacts()
        .hub(
            HubSelection::Files(&wrong),
            PullLimits::default(),
            Some(hub.reference().into()),
            None,
        )
        .unwrap();
    assert!(wrong.cached(&store, &cancel).is_err());
    let mut narrow = PullLimits::default();
    narrow.cache.max_file_bytes = 2;
    let narrow = runtime
        .artifacts()
        .hub(HubSelection::Files(&request), narrow, None, None)
        .unwrap();
    assert!(matches!(
        narrow.cached(&store, &cancel),
        Err(Error::Limit(_))
    ));
    drop(receipt);
    drop(prepared);
    assert_eq!(
        admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .reserved,
        0
    );
}
#[test]
fn inspection_cancel_and_closed_runtime_never_transfer_or_publish() {
    let fixture = Fixture::new();
    let store = fixture.store();
    let runtime = Runtime::builder().build().unwrap();
    let request = request();
    let hub = runtime
        .artifacts()
        .hub(
            HubSelection::Files(&request),
            PullLimits::default(),
            None,
            None,
        )
        .unwrap();
    let transport = Transport::default();
    let admission = admission();
    let cancel = Arc::new(AtomicBool::new(true));
    assert!(
        hub.resolve(&transport, &admission, &cancel, PullMode::Transfer)
            .is_err()
    );
    assert_eq!(transport.metadata.load(Ordering::Relaxed), 0);
    cancel.store(false, Ordering::Release);
    let prepared = hub
        .resolve(&transport, &admission, &cancel, PullMode::Inspect)
        .unwrap();
    assert!(matches!(
        prepared.execute(&store, &transport, &admission, &cancel, &mut |_| {}),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(transport.payload.load(Ordering::Relaxed), 0);
    assert!(store.list_snapshots().unwrap().is_empty());
    runtime.shutdown().unwrap();
    assert!(matches!(
        hub.resolve(&transport, &admission, &cancel, PullMode::Transfer),
        Err(Error::State(_))
    ));
    assert!(matches!(hub.cached(&store, &cancel), Err(Error::State(_))));
    assert!(matches!(
        runtime.artifacts().hub(
            HubSelection::Files(&request),
            PullLimits::default(),
            None,
            None
        ),
        Err(Error::State(_))
    ));
    assert_eq!(transport.metadata.load(Ordering::Relaxed), 2);
}
#[test]
fn component_documents_preserve_strict_schema_bounds_and_offline_bindings() {
    let limits = PullLimits::default();
    let mut doc = serde_json::json!({"schema":"zerogravity.pull.components.v1","components":[{"label":"weights","role":"weights","provider":"huggingface","repository":"fixture/repo","revision":PIN,"files":["model.gguf"]}]});
    let parse = |value: &serde_json::Value| {
        ComponentDocument::from_bytes(&serde_json::to_vec(value).unwrap(), &limits)
    };
    assert_eq!(
        parse(&doc)
            .unwrap()
            .online_request()
            .unwrap()
            .components()
            .len(),
        1
    );
    doc["components"][0]["snapshot"] = serde_json::json!("a".repeat(64));
    let cached = parse(&doc).unwrap();
    assert_eq!(cached.bindings().len(), 1);
    assert!(cached.online_request().is_err());
    doc["components"][0]["credential"] = serde_json::json!("not allowed");
    assert!(parse(&doc).is_err());
    assert!(ComponentDocument::from_bytes(&vec![b' '; 1024 * 1024 + 1], &limits).is_err());
    assert!(
        ComponentDocument::from_bytes(
            br#"{"schema":"zerogravity.pull.components.v1","schema":"duplicate","components":[]}"#,
            &limits
        )
        .is_err()
    );
}
#[test]
fn repeated_sdk_transfer_clients_share_the_original_fixed_budget() {
    let a = zerogravity::artifacts::transfer_admission(128 * 1024 * 1024).unwrap();
    let b = zerogravity::artifacts::transfer_admission(128 * 1024 * 1024).unwrap();
    assert_eq!(a.resource, b.resource);
    use zerogravity::resource_arbiter::{AllocationClass, AllocationRequest};
    let held = a
        .arbiter
        .admit(
            &a.resource,
            a.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 128 * 1024 * 1024,
            }],
        )
        .unwrap();
    assert!(
        b.arbiter
            .admit(
                &b.resource,
                b.owner.clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 1
                }]
            )
            .is_err()
    );
    drop(held);
    assert_eq!(b.arbiter.snapshot(&b.resource).unwrap().reserved, 0);
    assert!(zerogravity::artifacts::transfer_admission(129 * 1024 * 1024).is_err());
    assert_eq!(
        a.arbiter.snapshot(&a.resource).unwrap().limit,
        128 * 1024 * 1024
    );
}

#[test]
fn two_control_clients_share_pause_resume_cancel_and_durable_outcome() {
    use std::time::{Duration, Instant};
    use zerogravity::control::{
        ControlAction, ControlEventKind, ControlPoll, ControlReply, DownloadCommand, DownloadStatus,
    };
    let fixture = Fixture::new();
    let store = fixture.store();
    let runtime = Runtime::builder().build().unwrap();
    let peer = runtime.clone();
    let control = runtime.control();
    let other = peer.control();
    let mut events = other.subscribe();
    let request = request();
    let transport = Transport::default();
    let admission = admission();
    let hub = runtime
        .artifacts()
        .hub(
            HubSelection::Files(&request),
            PullLimits::default(),
            None,
            None,
        )
        .unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let prepared = hub
        .resolve(&transport, &admission, &cancel, PullMode::Transfer)
        .unwrap();
    let mut previous = None;
    for cancel_while_paused in [false, true] {
        let cancel = Arc::new(AtomicBool::new(false));
        let operation = control.begin_resolved_download(cancel.clone()).unwrap();
        let id = operation.id();
        assert!(
            other
                .begin_download(Arc::new(AtomicBool::new(false)))
                .is_err()
        );
        if let Some(old) = previous {
            assert!(
                other
                    .control_download(old, DownloadCommand::Cancel)
                    .is_err()
            );
        }
        // Request pause after a real file has completed; this fixture has
        // no strong validator and therefore cannot park within the response.
        let result = std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                operation.execute_prepared(
                    &prepared,
                    &store,
                    &transport,
                    &admission,
                    &mut |event| {
                        if event.file_complete() {
                            other.control_download(id, DownloadCommand::Pause).unwrap();
                        }
                    },
                )
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let row = other.download().unwrap();
                assert_eq!(row.id, id);
                if row.status == DownloadStatus::Paused {
                    break;
                }
                if worker.is_finished() || Instant::now() >= deadline {
                    let _ = other.control_download(id, DownloadCommand::Cancel);
                    panic!("worker did not acknowledge a safe pause: {row:?}");
                }
                if let ControlPoll::Event(event) = events.poll(Duration::from_millis(20)) {
                    assert!(
                        matches!(event.kind, ControlEventKind::DownloadChanged { id, .. } if id == row.id)
                    );
                }
            }
            // A request alone did not publish. The first completed ref, if
            // present, stays the same until a later durable commit.
            let paused = control.download().unwrap();
            assert_eq!(paused.total_bytes, 3);
            assert_eq!(paused.completed_files, 1);
            assert_eq!(paused.checkpoint_bytes, 0);
            assert!(matches!(
                other
                    .dispatch(ControlAction::Download {
                        id,
                        command: if cancel_while_paused {
                            DownloadCommand::Cancel
                        } else {
                            DownloadCommand::Resume
                        },
                    })
                    .unwrap(),
                ControlReply::Done
            ));
            worker.join().unwrap()
        });
        let terminal = other.snapshot().unwrap().download.unwrap();
        if cancel_while_paused {
            assert!(result.is_err());
            assert_eq!(terminal.status, DownloadStatus::Cancelled);
            assert!(terminal.cancel_requested);
        } else {
            let receipt = result.unwrap();
            assert!(matches!(receipt.outcome(), CommitOutcome::Durable(_)));
            assert_eq!(terminal.status, DownloadStatus::Durable);
            assert_eq!(
                terminal.snapshot.as_deref(),
                Some(receipt.manifest().digest().unwrap().as_str())
            );
        }
        assert_eq!(
            store
                .resolve_ref(hub.reference())
                .unwrap()
                .manifest()
                .total_bytes(),
            3
        );
        assert_eq!(
            admission
                .arbiter
                .snapshot(&admission.resource)
                .unwrap()
                .reserved,
            0
        );
        previous = Some(id);
        // Subscribe afresh so the next operation never confuses prior events.
        events = other.subscribe();
    }
    runtime.shutdown().unwrap();
}
