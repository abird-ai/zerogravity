//! Small explicit CUDA qualification for unpublished multi-token B2 prefill.
//! No serving/router/test-harness code is linked here.
use std::{
    future::Future,
    path::{Path, PathBuf},
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread,
};
use zerogravity_core::{accelerator::program::StateSlot, resource_arbiter::ResourceArbiter};
use zerogravity_gpu::{Device, ResidentModel, ResidentSession};
use zerogravity_model_qwen35::{LoadedQwenModel, QwenDenseCpu, tokenize::QwenTokenizer};

const PROVIDER: usize = 4 * 1024 * 1024 * 1024;
const DEVICE_CAP: u64 = 5 * 1024 * 1024 * 1024;
const CAPACITY: usize = 512;
const IDS: [u32; 8] = [11, 353, 17, 42, 55, 78, 101, 135];

struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

fn exact(expected: &[f32], actual: &[f32], label: &str) {
    assert_eq!(expected.len(), actual.len(), "{label} length");
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "{label} bits differ at {index}"
        );
    }
}
fn domains(expected: &[(StateSlot, Vec<f32>)], actual: &[(StateSlot, Vec<f32>)], label: &str) {
    assert_eq!(expected.len(), 42, "{label} expected domain count");
    assert_eq!(actual.len(), 42, "{label} actual domain count");
    for (slot, expected) in expected {
        let actual = actual
            .iter()
            .find(|(actual, _)| actual == slot)
            .map(|(_, values)| values)
            .unwrap_or_else(|| panic!("{label} missing {slot:?}"));
        exact(expected, actual, label);
    }
}
fn best(logits: &[f32]) -> (u32, u32) {
    assert_eq!(logits.len(), 248_320);
    let index = (0..logits.len())
        .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then_with(|| b.cmp(a)))
        .unwrap();
    (index as u32, logits[index].to_bits())
}
async fn check_state(
    native: &[ResidentSession],
    b1: &[ResidentSession],
    cpu: &[QwenDenseCpu],
    label: &str,
) {
    for row in 0..native.len() {
        let expected = cpu[row].snapshot().domain_values();
        domains(
            &expected,
            &native[row].snapshot().read_domains().await.unwrap(),
            label,
        );
        domains(
            &expected,
            &b1[row].snapshot().read_domains().await.unwrap(),
            label,
        );
    }
}

async fn run(root: &Path) {
    let source =
        LoadedQwenModel::open(&root.join("model.gguf"), &root.join("config.json")).unwrap();
    assert_eq!(source.geometry().vocab_size, 248_320);
    let plan = source.forward_plan(CAPACITY, true, true).unwrap();
    let model = ResidentModel::load(
        Device::Cuda(cubecl::device::CudaDevice::new(0)),
        source.clone(),
        plan,
        PROVIDER,
        DEVICE_CAP,
    )
    .await
    .unwrap();
    let service = model.qualification();
    let key = service.physical_key();

    let mut native = vec![
        model.session().await.unwrap(),
        model.session().await.unwrap(),
    ];
    let mut b1 = vec![
        model.session().await.unwrap(),
        model.session().await.unwrap(),
    ];
    let mut cpu = vec![
        QwenDenseCpu::from_shared(source.clone()).unwrap(),
        QwenDenseCpu::from_shared(source.clone()).unwrap(),
    ];
    check_state(&native, &b1, &cpu, "initial").await;

    let first = [
        vec![IDS[0], IDS[1], IDS[2], IDS[3]],
        vec![IDS[1], IDS[2], IDS[3], IDS[4]],
    ];
    let before = native
        .iter()
        .map(ResidentSession::binding)
        .collect::<Vec<_>>();
    let before_domains = futures_lite::future::zip(
        native[0].snapshot().read_domains(),
        native[1].snapshot().read_domains(),
    )
    .await;
    let before_domains = [before_domains.0.unwrap(), before_domains.1.unwrap()];

    // The first native B2 call lazily creates a reusable band workspace. Warm
    // that already-accepted one-token path before measuring candidate rollback,
    // so the temporal assertion only detects private candidate retention.
    let warm_rows = [(&native[0], IDS[0]), (&native[1], IDS[1])];
    let warm = model
        .prepare_state_fused_cohort_admitted(&warm_rows)
        .await
        .unwrap()
        .expect("accepted B2 warmup candidate admitted");
    drop(warm);
    assert_eq!(
        native
            .iter()
            .map(ResidentSession::binding)
            .collect::<Vec<_>>(),
        before,
        "accepted workspace warmup published a root"
    );
    let bytes = service.resident_bytes().unwrap();

    let rows = native
        .iter()
        .zip(&first)
        .map(|(session, tokens)| (session, tokens.as_slice()))
        .collect::<Vec<_>>();

    let abandoned = model
        .prepare_state_temporal_fused_cohort_admitted(&rows)
        .await
        .unwrap()
        .expect("B2 temporal state candidate admitted");
    assert_eq!(abandoned.len(), 2);
    assert!(abandoned.fused_dispatches() > 0);
    assert_eq!(abandoned.native_greedy_dispatches(), 0);
    assert_eq!(
        native
            .iter()
            .map(ResidentSession::binding)
            .collect::<Vec<_>>(),
        before,
        "completed private temporal chain published early"
    );
    drop(abandoned);
    assert_eq!(service.pending_events().unwrap(), 0);
    assert_eq!(service.resident_bytes().unwrap(), bytes);
    for row in 0..2 {
        assert_eq!(native[row].binding(), before[row]);
        domains(
            &before_domains[row],
            &native[row].snapshot().read_domains().await.unwrap(),
            "abandoned temporal candidate",
        );
    }

    for row in 0..2 {
        for &id in &first[row] {
            let expected = cpu[row].forward(id).unwrap();
            let actual = b1[row].forward(&[id], false).await.unwrap();
            exact(&expected, &actual.logits, "state chunk CPU/B1 logits");
        }
    }
    let rows = native
        .iter()
        .zip(&first)
        .map(|(session, tokens)| (session, tokens.as_slice()))
        .collect::<Vec<_>>();
    let candidate = model
        .prepare_state_temporal_fused_cohort_admitted(&rows)
        .await
        .unwrap()
        .expect("B2 temporal state candidate admitted");
    assert_eq!(candidate.prior_frontiers().collect::<Vec<_>>(), vec![0, 0]);
    let reads = candidate
        .commit(&mut native.iter_mut().collect::<Vec<_>>())
        .unwrap();
    for row in 0..2 {
        assert_eq!(reads[row].state_only_steps, 4);
        assert_eq!(reads[row].logit_steps, 0);
        assert!(reads[row].logits.is_empty());
        assert!(reads[row].device_token.is_none());
        assert_eq!(native[row].binding().root(), before[row].root() + 1);
        assert_eq!(native[row].binding().frontier(), before[row].frontier() + 4);
    }
    check_state(&native, &b1, &cpu, "committed state chunk").await;

    let final_chunk = [vec![IDS[4], IDS[5]], vec![IDS[5], IDS[6]]];
    let before = native
        .iter()
        .map(ResidentSession::binding)
        .collect::<Vec<_>>();
    let mut expected_packets = Vec::with_capacity(2);
    for row in 0..2 {
        let mut logits = Vec::new();
        for &id in &final_chunk[row] {
            logits = cpu[row].forward(id).unwrap();
        }
        let expected = best(&logits);
        let read = b1[row].forward_greedy(&final_chunk[row]).await.unwrap();
        let packet = read.device_token.unwrap();
        assert_eq!((packet.id, packet.max_bits), expected);
        expected_packets.push(expected);
    }
    let rows = native
        .iter()
        .zip(&final_chunk)
        .map(|(session, tokens)| (session, tokens.as_slice()))
        .collect::<Vec<_>>();
    let candidate = model
        .prepare_greedy_temporal_fused_cohort_admitted(&rows)
        .await
        .unwrap()
        .expect("B2 temporal greedy candidate admitted");
    assert!(candidate.fused_dispatches() > 0);
    assert_eq!(candidate.native_greedy_dispatches(), 4);
    let reads = candidate
        .commit(&mut native.iter_mut().collect::<Vec<_>>())
        .unwrap();
    for row in 0..2 {
        assert_eq!(reads[row].state_only_steps, 1);
        assert_eq!(reads[row].logit_steps, 1);
        assert!(reads[row].logits.is_empty());
        let packet = reads[row].device_token.as_ref().unwrap();
        assert_eq!((packet.id, packet.max_bits), expected_packets[row]);
        assert_eq!(packet.qualification_producer(), Some(native[row].binding()));
        assert_eq!(native[row].binding().root(), before[row].root() + 1);
        assert_eq!(native[row].binding().frontier(), before[row].frontier() + 2);
    }
    check_state(&native, &b1, &cpu, "terminal greedy chunk").await;
    assert_eq!(service.pending_events().unwrap(), 0);

    drop(reads);
    drop(cpu);
    drop(b1);
    drop(native);
    drop(model);
    assert_eq!(service.pending_events().unwrap(), 0);
    assert_eq!(service.resident_bytes().unwrap(), 0);
    assert_eq!(
        ResourceArbiter::process_shared()
            .snapshot(&key)
            .unwrap()
            .reserved,
        PROVIDER as u64
    );
    drop(service);
    let final_state = ResourceArbiter::process_shared().snapshot(&key).unwrap();
    assert_eq!(final_state.reserved, 0);
    assert!(!final_state.fenced);
    println!(
        "P12_NATIVE_TEMPORAL_B2_PASS source=qwen35-0.8b state_chunk=4 terminal_chunk=2 unpublished_abandon=true roots=atomic CPU_B1_all42=exact terminal_greedy=exact final_reserved=0"
    );
}

fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let (oracle_only, root) = match args.as_slice() {
        [mode, root] if mode == "--oracle" => (true, PathBuf::from(root)),
        [root] => (false, PathBuf::from(root)),
        [] => (
            false,
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b"),
        ),
        _ => panic!("usage: temporal_prefill_b2 [--oracle] [model-dir]"),
    };
    if oracle_only {
        let source =
            LoadedQwenModel::open(&root.join("model.gguf"), &root.join("config.json")).unwrap();
        let tokenizer = QwenTokenizer::from_file(&root.join("tokenizer.json")).unwrap();
        let prompt = " a".repeat(10);
        let tokens = tokenizer.prompt_tokens(&prompt, false).unwrap();
        assert_eq!(tokens.len(), 10);
        let mut cpu = QwenDenseCpu::from_shared(source).unwrap();
        let logits = cpu.prefill_transaction(&tokens).unwrap();
        let (id, max_bits) = best(&logits);
        let text = tokenizer.decode(&[id], true).unwrap();
        let text_hex = text
            .as_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        println!(
            "HTTP_ORACLE prompt_tokens={} id={id} max_bits={max_bits} text_hex={text_hex}",
            tokens.len()
        );
        return;
    }
    block_on(run(&root));
}
