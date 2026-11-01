use super::*;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};
use zerogravity_engine::router::{Request, RequestEvents, ResponseEvent};
#[test]
fn approved_tokenizer_reader_rejects_wrong_extent_and_symlink() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "zerogravity-tokenizer-reader-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("tokenizer.json");
    std::fs::write(&source, b"{}").unwrap();
    assert_eq!(
        read_approved_tokenizer(&source).unwrap_err().code,
        "tokenizer_inventory"
    );
    #[cfg(target_os = "linux")]
    {
        let link = root.join("linked-tokenizer.json");
        std::os::unix::fs::symlink(&source, &link).unwrap();
        assert_eq!(
            read_approved_tokenizer(&link).unwrap_err().code,
            "missing_tokenizer"
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
mod serving_population {
    #[cfg(test)]
    mod checkpoint;
    mod direct;

    use super::*;
    use std::collections::BTreeSet;
    use std::io::{Seek, SeekFrom, Write};

    const VOCAB: u64 = 248_320;
    const HIDDEN: usize = 1024;
    const LAYERS: usize = 24;
    const WORKERS: usize = 2;
    const ACTIVE_PER_WORKER: usize = 4;
    const ACTIVE_LIMIT: usize = WORKERS * ACTIVE_PER_WORKER;
    const HOST_MIB: usize = 4096;
    const MAX_CONTEXT: usize = 8;
    const MAX_OUTPUT: usize = 1;

    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }

    /// Genuine legal qwen35 GGUF with the FULL approved vocabulary and
    /// 24 full-attention layers. Token row 0 is the only non-zero
    /// embedding row, so greedy selection is the real id 0 ("!"). All
    /// zero ranges are real F32 weights written as sparse file holes; the
    /// fixture never builds a duplicate 1 GiB payload in memory.
    fn tensors() -> Vec<(String, Vec<u64>, f32)> {
        let mut tensors = Vec::new();
        tensors.push(("token_embd.weight".into(), vec![HIDDEN as u64, VOCAB], 0.0));
        tensors.push(("output_norm.weight".into(), vec![HIDDEN as u64], 1.0));
        for i in 0..LAYERS {
            for (name, shape, value) in [
                ("attn_norm.weight", vec![HIDDEN as u64], 1.0),
                ("post_attention_norm.weight", vec![HIDDEN as u64], 1.0),
                ("attn_q_norm.weight", vec![64], 1.0),
                ("attn_k_norm.weight", vec![64], 1.0),
                ("attn_q.weight", vec![HIDDEN as u64, 128], 0.0),
                ("attn_k.weight", vec![HIDDEN as u64, 64], 0.0),
                ("attn_v.weight", vec![HIDDEN as u64, 64], 0.0),
                ("attn_output.weight", vec![64, HIDDEN as u64], 0.0),
                ("ffn_gate.weight", vec![HIDDEN as u64, 1], 0.0),
                ("ffn_up.weight", vec![HIDDEN as u64, 1], 0.0),
                ("ffn_down.weight", vec![1, HIDDEN as u64], 0.0),
            ] {
                tensors.push((format!("blk.{i}.{name}"), shape, value));
            }
        }
        tensors
    }

    fn gguf(path: &Path) {
        let tensors = tensors();
        let mut infos = Vec::with_capacity(tensors.len());
        let mut payload_len = 0u64;
        for (name, shape, _) in &tensors {
            payload_len = (payload_len + 31) & !31;
            let offset = payload_len;
            let elements = shape.iter().product::<u64>();
            payload_len = payload_len
                .checked_add(elements * 4)
                .expect("fixture payload length");
            infos.push((name.clone(), shape.clone(), offset));
        }
        let mut header = b"GGUF".to_vec();
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        string(&mut header, "general.architecture");
        header.extend_from_slice(&8u32.to_le_bytes());
        string(&mut header, "qwen35");
        for (name, shape, offset) in &infos {
            string(&mut header, name);
            header.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dim in shape {
                header.extend_from_slice(&dim.to_le_bytes());
            }
            header.extend_from_slice(&0u32.to_le_bytes());
            header.extend_from_slice(&offset.to_le_bytes());
        }
        while !header.len().is_multiple_of(32) {
            header.push(0);
        }
        let payload_start = header.len() as u64;
        let total = payload_start
            .checked_add(payload_len)
            .expect("fixture total length");
        let mut file = std::fs::File::create(path).expect("fixture model");
        file.set_len(total).expect("sparse fixture extent");
        file.seek(SeekFrom::Start(0)).expect("fixture header seek");
        file.write_all(&header).expect("fixture header");
        for ((name, shape, value), (_, _, offset)) in tensors.iter().zip(&infos) {
            let start = payload_start + *offset;
            let elements = shape.iter().product::<u64>() as usize;
            let mut bytes = Vec::new();
            if name.as_str() == "token_embd.weight" {
                for _ in 0..HIDDEN {
                    bytes.extend_from_slice(&1.0f32.to_le_bytes());
                }
            } else if *value != 0.0 {
                bytes.reserve(elements * 4);
                for _ in 0..elements {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            if !bytes.is_empty() {
                file.seek(SeekFrom::Start(start))
                    .expect("fixture data seek");
                file.write_all(&bytes).expect("fixture data");
            }
        }
        file.sync_all().expect("fixture sync");
    }

    fn fixture(tag: &str) -> PathBuf {
        let approved = PathBuf::from(
            std::env::var_os("ZG_APPROVED_TOKENIZER")
                .expect("set ZG_APPROVED_TOKENIZER to the pinned approved tokenizer.json"),
        );
        let metadata = std::fs::metadata(&approved).expect("approved tokenizer metadata");
        assert_eq!(
            metadata.len(),
            crate::tokenize::APPROVED_TOKENIZER_BYTES,
            "approved tokenizer extent"
        );
        let root = std::env::temp_dir().join(format!("zg-serve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        // Approved source copied exactly: a regular file with the pinned
        // length/SHA that `read_approved_tokenizer` verifies.
        std::fs::copy(&approved, root.join("tokenizer.json")).unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": HIDDEN as u64,
                "head_dim": 64,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": VOCAB,
                "max_position_embeddings": 32768,
                "num_hidden_layers": LAYERS,
                "layer_types": vec!["full_attention"; LAYERS],
                "full_attention_interval": 1,
                "linear_conv_kernel_dim": 4,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1
            }
        });
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        gguf(&root.join("model.gguf"));
        root
    }

    fn request() -> Request {
        Request {
            model: "tiny".into(),
            input: "!".into(),
            chat: false,
            max_tokens: MAX_OUTPUT,
            stream: false,
            token_metrics: None,
        }
    }

    /// Drain one real receipt and return its id and actual token counts.
    /// The one terminal event must carry the real decoded id 0 output.
    fn drain(receipt: &RequestEvents) -> (u64, usize, usize, String) {
        let id = receipt.id;
        let mut terminal = None;
        let mut texts = 0;
        while let Ok(event) = receipt.events.recv() {
            match event {
                ResponseEvent::Text(_) => texts += 1,
                ResponseEvent::Done {
                    output,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                } => {
                    assert!(terminal.is_none(), "duplicate terminal event");
                    assert!(prompt_tokens >= 1, "actual prompt token count");
                    assert_eq!(completion_tokens, 1, "actual completion token count");
                    assert_eq!(finish_reason, "length");
                    assert_eq!(output, "!", "decoded greedy id 0");
                    terminal = Some((prompt_tokens, completion_tokens, output));
                }
                ResponseEvent::TokenText { .. } => panic!("unexpected token metric event"),
                ResponseEvent::Error(error) => {
                    panic!("request {id} failed: {} {}", error.status, error.code)
                }
            }
        }
        assert_eq!(texts, 0, "non-streaming request emits only the terminal");
        let (prompt_tokens, completion_tokens, output) = terminal.expect("missing terminal event");
        (id, prompt_tokens, completion_tokens, output)
    }

    fn reexec_with(marker: &str, test: &str, done: &str, envs: &[(&str, &str)]) -> bool {
        if std::env::var_os(marker).is_some() {
            return false;
        }
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--ignored",
                "--exact",
                test,
                "--nocapture",
                "--test-threads=1",
            ])
            .env(marker, "1");
        for (key, value) in envs {
            command.env(key, value);
        }
        let output = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "child failed: {stdout}\n{stderr}");
        assert!(stdout.contains(done), "child marker missing: {stdout}");
        true
    }

    fn reexec(marker: &str, test: &str, done: &str) -> bool {
        reexec_with(marker, test, done, &[])
    }

    #[test]
    #[ignore = "requires approved tokenizer metadata"]
    fn real_serving_populations_match_actual_router_state() {
        let test = "builtins::text::tests::serving_population::real_serving_populations_match_actual_router_state";
        if reexec("ZG_SERVE_POPULATION_CHILD", test, "POPULATION_CHILD_OK") {
            return;
        }
        let root = fixture("population");
        let model = root.join("model.gguf");
        // Exact current envelope: 4096 MiB configured host and 8 total
        // active slots (2 workers x 4). Every population is offered to
        // the original Router and must complete with real decoded output.
        // Both route names point at the same actual fixture path so the
        // router must share one `Model` owner and one tokenizer credit.
        let router = match load_router(
            vec![
                ("tiny".to_string(), model.clone()),
                ("alias".to_string(), model.clone()),
            ],
            WORKERS,
            ACTIVE_PER_WORKER,
            MAX_CONTEXT,
            MAX_OUTPUT,
            HOST_MIB,
            None,
        ) {
            Ok(router) => router,
            Err(error) => panic!(
                "4096 MiB / 8 active host refused: status={} code={} message={}",
                error.status, error.code, error.message
            ),
        };
        assert_eq!(router.active_limit(), ACTIVE_LIMIT);
        assert_eq!(router.queue_snapshot().limit, None);
        let tiny = router.model("tiny").expect("actual tiny route");
        let alias = router.model("alias").expect("actual alias route");
        assert!(
            tiny.shares_source(&alias),
            "same fixture path must share one Model/tokenizer source credit"
        );
        drop((tiny, alias));
        let host = router.host().clone();
        let baseline = host.snapshot().unwrap().3;
        assert!(baseline > 0, "model/queue/worker serving charges are live");
        println!(
            "baseline serving_bytes={baseline} memory={:?} active_limit={}",
            router.memory_snapshot(),
            router.active_limit()
        );

        for population in [16usize, 32, 64] {
            // Straight offer loop: every real request is offered to the
            // original Router and its receipt is retained before any
            // receipt is drained, received or dropped.
            let mut receipts = Vec::with_capacity(population);
            let mut ids = BTreeSet::new();
            let mut tokens = 0usize;
            for offered in 0..population {
                let receipt = match router.request(request()) {
                    Ok(receipt) => receipt,
                    Err(error) => panic!(
                        "population {population} offer={offered} refused: status={} code={} message={} queue={:?} memory={:?}",
                        error.status,
                        error.code,
                        error.message,
                        router.queue_snapshot(),
                        router.memory_snapshot()
                    ),
                };
                receipts.push(receipt);
            }
            assert_eq!(receipts.len(), population);
            let offered_queue = router.queue_snapshot();
            let offered_active = router.active_sessions();
            assert_eq!(offered_queue.limit, None);
            assert_eq!(offered_queue.consumers, WORKERS);
            assert!(
                offered_queue.pending <= offered_queue.capacity,
                "pending fits the ring"
            );
            assert!(offered_active <= router.active_limit());
            println!(
                "population {population} offered={} active={offered_active} pending={} in_progress={} capacity={} backing={} consumers={}",
                receipts.len(),
                offered_queue.pending,
                offered_queue.in_progress,
                offered_queue.capacity,
                offered_queue.backing_bytes,
                offered_queue.consumers
            );

            let with_receipts = host.snapshot().unwrap().3;
            assert!(with_receipts > baseline);
            for receipt in &receipts {
                let (id, prompt, completion, output) = drain(receipt);
                assert!(ids.insert(id), "duplicate request receipt {id}");
                assert!(prompt >= 1);
                assert_eq!(completion, 1);
                assert_eq!(output, "!");
                tokens += completion;
            }
            drop(receipts);
            let after_receipts = host.snapshot().unwrap().3;
            assert!(after_receipts < with_receipts, "receipt charges released");
            // The grown queue backing is legitimately retained while
            // the router and its workers still own it.
            assert!(after_receipts > baseline, "live queue backing retained");
            let queue = router.queue_snapshot();
            let active = router.active_sessions();
            let peak = router.peak_active_sessions();
            assert_eq!(ids.len(), population, "unique completed request ids");
            assert_eq!(tokens, population, "one actual output token per request");
            assert_eq!(queue.pending, 0);
            assert_eq!(queue.in_progress, 0);
            assert_eq!(queue.limit, None);
            assert_eq!(queue.consumers, WORKERS);
            assert!(queue.capacity > 0);
            assert!(queue.backing_bytes > 0);
            assert!(active <= router.active_limit());
            assert!(peak >= 1 && peak <= router.active_limit());
            println!(
                "population {population} completed: completed={} tokens={tokens} active={active} pending={} in_progress={} capacity={} backing={} peak={peak}",
                ids.len(),
                queue.pending,
                queue.in_progress,
                queue.capacity,
                queue.backing_bytes
            );
        }

        let final_serving = host.snapshot().unwrap().3;
        assert!(final_serving > 0);
        drop(router);
        let closed = host.snapshot().unwrap().3;
        assert_eq!(
            closed, 0,
            "router drop releases model/request/queue/workers"
        );
        let memory = host.snapshot().expect("host snapshot");
        assert_eq!(memory.3, 0);
        assert!(memory.1 > 0, "permanent bootstrap/Rayon/root reservations");
        let persistent_reserved = memory.1;
        println!(
            "router closure: serving_before={final_serving} serving_after={closed} reserved={} limit={} remaining={}",
            memory.1, memory.0, memory.2
        );

        // Keep real source completion/cleanup under the optional count cap.
        // The held-ticket 429/refill subcase lives in engine unit tests, where
        // the real queue is visible without exporting a qualification escape.
        let capped = match load_router(
            vec![("tiny".to_string(), model)],
            WORKERS,
            ACTIVE_PER_WORKER,
            MAX_CONTEXT,
            MAX_OUTPUT,
            HOST_MIB,
            Some(1),
        ) {
            Ok(router) => router,
            Err(error) => panic!(
                "capped 4096 MiB / 8 active router refused: status={} code={} message={}",
                error.status, error.code, error.message
            ),
        };
        assert_eq!(capped.active_limit(), ACTIVE_LIMIT);
        assert_eq!(capped.queue_snapshot().limit, Some(1));
        let capped_host = capped.host().clone();
        let capped_baseline = capped_host.snapshot().unwrap().3;
        let receipt = capped.request(request()).expect("capped real request");
        let (id, prompt, completion, output) = drain(&receipt);
        assert!(id > 0);
        assert!(prompt >= 1);
        assert_eq!(completion, 1);
        assert_eq!(output, "!");
        let accepted_queue = capped.queue_snapshot();
        assert_eq!(accepted_queue.pending, 0);
        assert_eq!(accepted_queue.in_progress, 0);
        let with_receipt = capped_host.snapshot().unwrap().3;
        assert!(with_receipt > capped_baseline);
        drop(receipt);
        let after_receipt = capped_host.snapshot().unwrap().3;
        assert!(after_receipt < with_receipt);
        assert!(
            after_receipt > capped_baseline,
            "capped queue backing retained"
        );
        println!(
            "cap: limit=1 accepted_id={id} tokens={completion} serving_receipts={with_receipt} serving_after_receipt={after_receipt}"
        );
        drop(capped);
        let capped_closed = capped_host.snapshot().unwrap().3;
        assert_eq!(
            capped_closed, 0,
            "capped router drop releases serving charges"
        );
        let capped_memory = capped_host.snapshot().expect("capped host snapshot");
        assert_eq!(capped_memory.3, 0);
        assert_eq!(
            capped_memory.1, persistent_reserved,
            "second router releases model/queue/worker charges and reuses globals"
        );
        println!(
            "cap closure: serving_after={capped_closed} reserved={} persistent_reserved={persistent_reserved}",
            capped_memory.1
        );
        std::fs::remove_dir_all(&root).unwrap();
        println!("POPULATION_CHILD_OK");
    }

    #[test]
    #[ignore = "requires approved tokenizer metadata"]
    fn cancelled_requests_keep_source_credit_and_later_genuine_encode_succeeds() {
        let test = "builtins::text::tests::serving_population::cancelled_requests_keep_source_credit_and_later_genuine_encode_succeeds";
        if reexec("ZG_SERVE_CANCEL_CHILD", test, "CANCEL_CHILD_OK") {
            return;
        }
        let root = fixture("cancel");
        let router = load_router(
            vec![("tiny".to_string(), root.join("model.gguf"))],
            WORKERS,
            ACTIVE_PER_WORKER,
            MAX_CONTEXT,
            MAX_OUTPUT,
            HOST_MIB,
            None,
        )
        .expect("4096 MiB / 8 active router");
        let host = router.host().clone();
        let source = router.model("tiny").expect("actual source route");
        let baseline = host.snapshot().expect("host snapshot");
        let before = source
            .prepared()
            .tokenizer()
            .prompt_tokens("!", false)
            .expect("genuine encode before cancellation");
        assert!(!before.is_empty());

        // Dropping the receipt is the request's cancellation owner. Phase
        // timing is intentionally not asserted: whether the row was still
        // queued or already active is not pinned by this test. The source
        // model keeps its `Retained` R+G+V lease independent of
        // cancellation.
        let receipt = router.request(request()).expect("real cancellable request");
        drop(receipt);
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let queue = router.queue_snapshot();
            if queue.pending == 0 && queue.in_progress == 0 && router.active_sessions() == 0 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "cancelled request did not retire: queue={queue:?} active={}",
                router.active_sessions()
            );
            thread::sleep(Duration::from_millis(10));
        }
        let after_cancel = host.snapshot().expect("host snapshot after cancel");
        assert_eq!(
            after_cancel.0, baseline.0,
            "configured host limit is unchanged"
        );
        assert!(
            after_cancel.1 >= baseline.1,
            "source/bootstrap reservations survive request cancellation"
        );
        assert_eq!(
            source
                .prepared()
                .tokenizer()
                .prompt_tokens("!", false)
                .expect("genuine encode after cancellation"),
            before,
            "the source tokenizer and its retained credit remain live"
        );

        // The same source still serves a real request after cancellation.
        let receipt = router
            .request(request())
            .expect("request after cancellation");
        let (_, prompt, completion, output) = drain(&receipt);
        assert!(prompt >= 1);
        assert_eq!(completion, 1);
        assert_eq!(output, "!");
        drop(receipt);
        assert_eq!(
            source
                .prepared()
                .tokenizer()
                .prompt_tokens("!", false)
                .expect("genuine encode after serving"),
            before
        );
        drop(source);
        drop(router);
        assert_eq!(host.snapshot().unwrap().3, 0);
        std::fs::remove_dir_all(&root).unwrap();
        println!("CANCEL_CHILD_OK");
    }

    #[test]
    #[ignore = "requires approved tokenizer metadata"]
    fn application_host_refusal_is_distinct_from_the_pending_cap() {
        let test = "builtins::text::tests::serving_population::application_host_refusal_is_distinct_from_the_pending_cap";
        if reexec("ZG_SERVE_HOST_CHILD", test, "HOST_CHILD_OK") {
            return;
        }
        // Same genuine 24-layer/full-vocabulary source. Retained `R + G + V`
        // (plus bootstrap/constructor rows) exceeds a 1 GiB application
        // host, so the refusal is the host_budget stage before weights are
        // opened, not the optional pending cap.
        let root = fixture("host");
        let source = root.join("model.gguf");
        let error = match load_router(
            vec![("tiny".to_string(), source)],
            1,
            1,
            MAX_CONTEXT,
            MAX_OUTPUT,
            1024,
            None,
        ) {
            Err(error) => error,
            Ok(_) => panic!("host admission must refuse the tokenizer retention"),
        };
        assert_eq!(
            (error.status, error.code),
            (503, "host_budget"),
            "actual host refusal: {}",
            error.message
        );
        assert!(
            error.message.contains("stage=retained-inventory"),
            "actual host stage: {}",
            error.message
        );
        assert!(
            error.message.contains("owner=model=tiny"),
            "actual host owner: {}",
            error.message
        );
        assert!(
            error.message.contains("session=retained-inventory"),
            "actual host session: {}",
            error.message
        );
        std::fs::remove_dir_all(&root).unwrap();
        println!("HOST_CHILD_OK: {}", error.message);
    }
}
