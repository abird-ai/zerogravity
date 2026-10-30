//! Explicit real-source ordinary CUDA lifecycle target; fresh process per terminal cut.
use std::{
    future::{Future, poll_fn},
    path::PathBuf,
    pin::pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};
use zerogravity::{
    Error, Result,
    accelerator::{SourceIdentity, program::SourceCallMode},
    gpu::{Device, qualification::resident as diagnostics},
    qwen35_cuda,
    resource_arbiter::{ResourceArbiter, ResourceOwner},
};
const MIB: usize = 1024 * 1024;
fn require(value: bool, text: &str) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(Error::State(text.into()))
    }
}
struct ThreadWake(std::thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = pin!(future);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[test]
#[ignore = "isolated release ordinary real-model lifecycle; normal/cancel/unknown mode"]
fn ordinary_lifecycle_on_nvidia() -> Result<()> {
    require(!cfg!(debug_assertions), "release lifecycle target required")?;
    let mode = std::env::var("ZG_QWEN35_ORDINARY_LIFECYCLE")
        .map_err(|_| Error::State("lifecycle mode required".into()))?;
    require(
        matches!(mode.as_str(), "normal" | "cancel" | "unknown" | "refuse"),
        "unknown lifecycle mode",
    )?;
    let directory = PathBuf::from(
        std::env::var_os("ZG_QWEN35_MODEL_DIR")
            .ok_or_else(|| Error::State("model directory required".into()))?,
    );
    require(directory.is_absolute(), "absolute model directory required")?;
    let capacity = std::env::var("ZG_QWEN35_CONTEXT")
        .unwrap_or_else(|_| "512".into())
        .parse::<usize>()
        .map_err(|_| Error::State("context capacity".into()))?;
    block_on(async {
        let device = Device::cuda(0).map_err(|e| Error::State(format!("CUDA device: {e}")))?;
        let loaded = qwen35_cuda::ordinary_candidate::Model::load(
            device,
            &directory,
            capacity,
            qwen35_cuda::HOST_MIB,
            qwen35_cuda::PROVIDER_MIB * MIB,
            (qwen35_cuda::DEVICE_MIB * MIB) as u64,
            None,
        )
        .await;
        if mode == "refuse" {
            require(capacity == 32_768, "refusal context must be32K")?;
            return match loaded {
                Err(Error::Limit(detail))
                    if detail
                        == "ordinary CUDA exact cold plus two-root plan does not fit provider cap: 4840819049 > 4831838208" =>
                {
                    // This exact branch precedes physical-owner creation in load_ordinary_cuda.
                    println!(
                        "ORDINARY_CUDA_ADMISSION_REFUSAL_PASS family=2b capacity=32768 required=4840819049 available=4831838208 preflight=true"
                    );
                    Ok(())
                }
                Err(error) => Err(error),
                Ok(model) => {
                    model.shutdown().await?;
                    Err(Error::State("over-budget2B32K model admitted".into()))
                }
            };
        }
        let model = loaded?;
        let family = model.source_metadata().family;
        let resident = model.resident().clone();
        let service = resident.qualification();
        let witness = resident.qualification_witness();
        let mut session = resident.session().await?;
        let prompt =
            zerogravity::qwen35::source::binding::OriginalWindowModelSource::fixed_prompt_ids();
        let request = |mode, base, ids: &[u32]| {
            model
                .plan()
                .bind_invocation(mode, base, Arc::from(ids), true)?
                .materialize_source_plan()
        };
        let initial = session.binding();
        let bytes = service.resident_bytes()?;
        let cancelled = AtomicBool::new(true);
        require(
            session
                .prepare_ordinary_cuda_cancellable(
                    request(SourceCallMode::Prefill, 0, prompt.as_slice())?,
                    &cancelled,
                )
                .await
                .is_err(),
            "pre-launch cancellation accepted",
        )?;
        require(
            session.binding() == initial
                && service.pending_events()? == 0
                && service.resident_bytes()? == bytes,
            "pre-launch cancellation changed root/ownership",
        )?;
        service.ensure_usable()?;
        cancelled.store(false, Ordering::Release);
        let output = session
            .forward_ordinary_cuda(request(SourceCallMode::Prefill, 0, prompt.as_slice())?)
            .await?;
        require(
            output.logits.len() == 248_320 && output.logits.iter().all(|x| x.is_finite()),
            "initial logits extent/finite",
        )?;
        let token = output
            .logits
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &x)| {
                if x > best.1 { (i, x) } else { best }
            })
            .0 as u32;
        let prior = session.snapshot();
        let before = session.binding();
        let expected = prior.read_domains().await?;
        require(
            expected.len() == 42 && before.frontier() == 47,
            "initial root/state coverage",
        )?;
        if mode == "normal" {
            let prepared = session
                .prepare_ordinary_cuda(request(SourceCallMode::Decode, 47, &[token])?)
                .await?;
            require(session.binding() == before, "prepare published a root")?;
            session.restore(&prior)?;
            let restored = session.binding();
            require(
                session.commit(prepared).is_err() && session.binding() == restored,
                "stale ready published after restore",
            )?;
            let observed = session.snapshot().read_domains().await?;
            diagnostics::compare_domains(&expected, &observed)?;
            let _ = session
                .forward_ordinary_cuda(request(SourceCallMode::Decode, 47, &[token])?)
                .await?;
            require(
                session.binding().frontier() == 48 && service.pending_events()? == 0,
                "cached completion/root differs",
            )?;
            require(
                model.clone().shutdown().await.is_err(),
                "live-owner shutdown succeeded",
            )?;
            drop((prior, session, resident));
            model.shutdown().await?;
            // Successful shutdown checks zero bytes/events before removing the namespace.
            require(
                service.resident_bytes().is_err()
                    && service.ensure_usable().is_err()
                    && service.pending_events()? == 0
                    && !ResourceArbiter::process_shared()
                        .snapshot(&service.physical_key())?
                        .fenced,
                "normal shutdown namespace/event/fence differs",
            )?;
            println!(
                "ORDINARY_CUDA_LIFECYCLE_PASS mode=normal family={family} capacity={capacity} active=48 precancel=true stale_ready=refused restore=exact shutdown=retired"
            );
            return Ok(());
        }
        resident.qualification_reset_launch_seen();
        resident.qualification_arm_abort(1);
        let mut future = Box::pin(session.prepare_ordinary_cuda_cancellable(
            request(SourceCallMode::Decode, 47, &[token])?,
            &cancelled,
        ));
        let handles = poll_fn(|cx| {
            if future.as_mut().poll(cx).is_ready() {
                return Poll::Ready(Err(Error::State(
                    "terminal cut did not suspend after real launch".into(),
                )));
            }
            let handles = resident.qualification_launch_seen();
            if handles == 0 {
                Poll::Pending
            } else {
                Poll::Ready(Ok(handles))
            }
        })
        .await?;
        if mode == "cancel" {
            cancelled.store(true, Ordering::Release);
            resident.qualification_reset_launch_seen();
            require(
                future.await.is_err(),
                "after-launch cancellation returned ready",
            )?;
        } else {
            drop(future);
        }
        require(
            session.binding() == before && service.ensure_usable().is_err(),
            "terminal cut published or lost global fence",
        )?;
        let client = service.client();
        let quarantined = diagnostics::quarantined_handles_for(&client)?;
        require(
            quarantined == handles && service.pending_events()? == 1,
            "terminal original handle/event quarantine differs",
        )?;
        client
            .sync()
            .await
            .map_err(|e| Error::State(format!("diagnostic quiescence: {e}")))?;
        diagnostics::fused::physical_prior_after_abort(&client, &prior, &expected).await?;
        let other = diagnostics::QualificationService::new(
            Device::cuda(0).map_err(|e| Error::State(format!("CUDA: {e}")))?,
            SourceIdentity::for_new_allocation()?,
            ResourceOwner::new("ordinary-fault-peer", "refused")?,
            MIB,
            (qwen35_cuda::DEVICE_MIB * MIB) as u64,
        );
        require(other.is_err(), "quarantine admitted another model")?;
        let key = service.physical_key();
        let retained = service.resident_bytes()?;
        drop((prior, session, resident, model, service));
        let kept = witness
            .upgrade()
            .ok_or_else(|| Error::State("quarantine lost service owner".into()))?;
        let budget = ResourceArbiter::process_shared().snapshot(&key)?;
        require(
            budget.fenced
                && budget.reserved >= (qwen35_cuda::PROVIDER_MIB * MIB) as u64
                && kept.resident_bytes()? == retained
                && kept.pending_events()? == 1,
            "quarantine released original charges after external drops",
        )?;
        println!(
            "ORDINARY_CUDA_LIFECYCLE_PASS mode={mode} family={family} capacity={capacity} submitted_handles={handles} quarantined={quarantined} root_unchanged=true prior_bits=exact tail=zero peer=refused retained_bytes={retained} recovery=false"
        );
        Ok(())
    })
}
