//! Small explicit CUDA qualification for unpublished multi-token B2/B4/B8 prefill.
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

type Domains = Vec<(StateSlot, Vec<f32>)>;

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
fn domains(expected: &Domains, actual: &Domains, label: &str) {
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
fn tokens(row: usize, start: usize, count: usize) -> Vec<u32> {
    (start..start + count)
        .map(|step| IDS[(row + step) % IDS.len()])
        .collect()
}

async fn run(root: &Path, width: usize, check_b1: bool) {
    assert!(
        matches!(width, 2 | 4 | 8),
        "temporal width must be B2/B4/B8"
    );
    let source =
        LoadedQwenModel::open(&root.join("model.gguf"), &root.join("config.json")).unwrap();
    assert_eq!(source.geometry().vocab_size, 248_320);
    let first = (0..width).map(|row| tokens(row, 0, 4)).collect::<Vec<_>>();
    let final_chunk = (0..width).map(|row| tokens(row, 4, 2)).collect::<Vec<_>>();

    // Keep only one zero-state snapshot. Wider temporal qualification must not
    // retain B8 copies of every full state domain in host memory.
    let initial = QwenDenseCpu::from_shared(source.clone())
        .unwrap()
        .snapshot()
        .domain_values();
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
    eprintln!("TEMPORAL_STAGE model_loaded B={width}");
    assert_eq!(service.pending_events().unwrap(), 0);

    let mut native = Vec::with_capacity(width);
    for _ in 0..width {
        native.push(model.session().await.unwrap());
    }
    for session in &native {
        domains(
            &initial,
            &session.snapshot().read_domains().await.unwrap(),
            "native initial",
        );
    }
    eprintln!("TEMPORAL_STAGE native_roots_ready B={width}");

    let before = native
        .iter()
        .map(ResidentSession::binding)
        .collect::<Vec<_>>();
    let mut before_domains = Vec::with_capacity(width);
    for session in &native {
        before_domains.push(session.snapshot().read_domains().await.unwrap());
    }

    // The first native call lazily creates a reusable band workspace. Warm that
    // already-accepted one-token path before measuring candidate rollback, so
    // the temporal assertion only detects private candidate retention.
    let warm_rows = native
        .iter()
        .enumerate()
        .map(|(row, session)| (session, IDS[row]))
        .collect::<Vec<_>>();
    let warm = model
        .prepare_state_fused_cohort_admitted(&warm_rows)
        .await
        .unwrap()
        .expect("accepted native warmup candidate admitted");
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
    eprintln!("TEMPORAL_STAGE native_warm B={width} bytes={bytes}");

    let rows = native
        .iter()
        .zip(&first)
        .map(|(session, tokens)| (session, tokens.as_slice()))
        .collect::<Vec<_>>();
    let abandoned = model
        .prepare_state_temporal_fused_cohort_admitted(&rows)
        .await
        .unwrap()
        .expect("temporal state candidate admitted");
    assert_eq!(abandoned.len(), width);
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
    for row in 0..width {
        assert_eq!(native[row].binding(), before[row]);
        domains(
            &before_domains[row],
            &native[row].snapshot().read_domains().await.unwrap(),
            "abandoned temporal candidate",
        );
    }
    eprintln!("TEMPORAL_STAGE abandon_verified B={width}");

    let rows = native
        .iter()
        .zip(&first)
        .map(|(session, tokens)| (session, tokens.as_slice()))
        .collect::<Vec<_>>();
    let candidate = model
        .prepare_state_temporal_fused_cohort_admitted(&rows)
        .await
        .unwrap()
        .expect("temporal state candidate admitted");
    assert_eq!(
        candidate.prior_frontiers().collect::<Vec<_>>(),
        vec![0; width]
    );
    let reads = candidate
        .commit(&mut native.iter_mut().collect::<Vec<_>>())
        .unwrap();
    eprintln!("TEMPORAL_STAGE state_commit_ready B={width}");
    for row in 0..width {
        assert_eq!(reads[row].state_only_steps, 4);
        assert_eq!(reads[row].logit_steps, 0);
        assert!(reads[row].logits.is_empty());
        assert!(reads[row].device_token.is_none());
        assert_eq!(native[row].binding().root(), before[row].root() + 1);
        assert_eq!(native[row].binding().frontier(), before[row].frontier() + 4);

        let mut cpu = QwenDenseCpu::from_shared(source.clone()).unwrap();
        let mut b1 = if check_b1 {
            Some(model.session().await.unwrap())
        } else {
            None
        };
        for &id in &first[row] {
            let expected = cpu.forward(id).unwrap();
            if let Some(b1) = b1.as_mut() {
                let actual = b1.forward(&[id], false).await.unwrap();
                exact(&expected, &actual.logits, "state chunk CPU/B1 logits");
            }
        }
        let expected = cpu.snapshot().domain_values();
        domains(
            &expected,
            &native[row].snapshot().read_domains().await.unwrap(),
            "committed state chunk",
        );
        if let Some(b1) = b1.as_ref() {
            domains(
                &expected,
                &b1.snapshot().read_domains().await.unwrap(),
                "B1 state chunk",
            );
        }
        drop(b1);
        drop(cpu);
        assert_eq!(service.pending_events().unwrap(), 0);
        eprintln!("TEMPORAL_STAGE state_reference_complete B={width} row={row}");
    }

    let before = native
        .iter()
        .map(ResidentSession::binding)
        .collect::<Vec<_>>();
    let rows = native
        .iter()
        .zip(&final_chunk)
        .map(|(session, tokens)| (session, tokens.as_slice()))
        .collect::<Vec<_>>();
    let candidate = model
        .prepare_greedy_temporal_fused_cohort_admitted(&rows)
        .await
        .unwrap()
        .expect("temporal greedy candidate admitted");
    assert!(candidate.fused_dispatches() > 0);
    assert_eq!(candidate.native_greedy_dispatches(), 4);
    let reads = candidate
        .commit(&mut native.iter_mut().collect::<Vec<_>>())
        .unwrap();
    eprintln!("TEMPORAL_STAGE greedy_commit_ready B={width}");
    for row in 0..width {
        assert_eq!(reads[row].state_only_steps, 1);
        assert_eq!(reads[row].logit_steps, 1);
        assert!(reads[row].logits.is_empty());
        let packet = reads[row].device_token.as_ref().unwrap();

        let mut cpu = QwenDenseCpu::from_shared(source.clone()).unwrap();
        for &id in &first[row] {
            cpu.forward(id).unwrap();
        }
        let mut logits = Vec::new();
        for &id in &final_chunk[row] {
            logits = cpu.forward(id).unwrap();
        }
        let expected_packet = best(&logits);
        assert_eq!(
            (packet.id, packet.max_bits),
            expected_packet,
            "terminal packet differs at row {row}"
        );
        assert_eq!(packet.qualification_producer(), Some(native[row].binding()));
        assert_eq!(native[row].binding().root(), before[row].root() + 1);
        assert_eq!(native[row].binding().frontier(), before[row].frontier() + 2);
        let expected = cpu.snapshot().domain_values();
        domains(
            &expected,
            &native[row].snapshot().read_domains().await.unwrap(),
            "terminal greedy chunk",
        );

        if check_b1 {
            let mut b1 = model.session().await.unwrap();
            domains(
                &initial,
                &b1.snapshot().read_domains().await.unwrap(),
                "B1 initial",
            );
            for &id in &first[row] {
                b1.forward(&[id], false).await.unwrap();
            }
            let read = b1.forward_greedy(&final_chunk[row]).await.unwrap();
            let packet = read.device_token.unwrap();
            assert_eq!(
                (packet.id, packet.max_bits),
                expected_packet,
                "B1 terminal packet differs at row {row}"
            );
            domains(
                &expected,
                &b1.snapshot().read_domains().await.unwrap(),
                "B1 terminal chunk",
            );
            drop(b1);
        }
        drop(cpu);
        assert_eq!(service.pending_events().unwrap(), 0);
        eprintln!("TEMPORAL_STAGE terminal_reference_complete B={width} row={row}");
    }

    assert_eq!(service.pending_events().unwrap(), 0);
    drop(reads);
    drop(native);
    assert_eq!(service.pending_events().unwrap(), 0);
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
    let source_name = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown");
    if check_b1 {
        println!(
            "P12_NATIVE_TEMPORAL_PASS source={source_name} B={width} state_chunk=4 terminal_chunk=2 unpublished_abandon=true roots=atomic CPU_B1_all42=exact terminal_greedy=exact final_reserved=0"
        );
    } else {
        println!(
            "P12_NATIVE_TEMPORAL_CUDA_CPU_PASS source={source_name} B={width} state_chunk=4 terminal_chunk=2 unpublished_abandon=true roots=atomic CPU_all42=exact terminal_greedy=exact B1=separate final_reserved=0"
        );
    }
}

fn model_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b")
}

fn main() {
    let mut args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let native_only = args.first().is_some_and(|value| value == "--native-only");
    if native_only {
        args.remove(0);
    }
    let (oracle_only, width, root) = match args.as_slice() {
        [mode, root] if mode == "--oracle" => (true, 2, PathBuf::from(root)),
        [mode] if mode == "--oracle" => (true, 2, model_root()),
        [width, root] => {
            let width = width
                .to_str()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|width| matches!(width, 2 | 4 | 8))
                .expect("temporal width must be 2, 4 or 8");
            (false, width, PathBuf::from(root))
        }
        [value] => match value.to_str().and_then(|value| value.parse::<usize>().ok()) {
            Some(width) if matches!(width, 2 | 4 | 8) => (false, width, model_root()),
            _ => (false, 2, PathBuf::from(value)),
        },
        [] => (false, 2, model_root()),
        _ => panic!(
            "usage: temporal_prefill [--native-only] [2|4|8] [model-dir] | --oracle [model-dir]"
        ),
    };
    assert!(
        !(native_only && oracle_only),
        "--native-only cannot qualify an oracle-only run"
    );
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
    block_on(run(&root, width, !native_only));
}
