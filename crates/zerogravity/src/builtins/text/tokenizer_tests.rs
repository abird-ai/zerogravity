use super::*;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;
use zerogravity_engine::router::{GeneratedText, Presentation, ResponseEvent, event_channel::mpsc};
pub(crate) static CHANNEL_TESTS: Mutex<()> = Mutex::new(());
fn fixture_tokenizer(byte_level: bool) -> QwenTokenizer {
    let decoder = if byte_level {
        serde_json::json!({"type":"ByteLevel", "add_prefix_space":false,
                "trim_offsets":true, "use_regex":true})
    } else {
        serde_json::json!({"type":"Fuse"})
    };
    let source = serde_json::json!({
        "version":"1.0", "truncation":null, "padding":null,
        "normalizer":null, "pre_tokenizer":null, "post_processor":null,
        "added_tokens":[
            {"id":0,"content":"<|im_start|>","single_word":false,
                "lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":1,"content":"<|im_end|>","single_word":false,
                "lstrip":false,"rstrip":false,"normalized":false,"special":true}
        ],
        "decoder":decoder,
        "model":{"type":"WordLevel","unk_token":"[UNK]","vocab":{
            "<|im_start|>":0,"<|im_end|>":1,"A":2,"B":3,"Ã":4,"©":5,"[UNK]":6
        }}
    });
    QwenTokenizer::from_bytes(&serde_json::to_vec(&source).unwrap()).unwrap()
}

fn output_tokenizer(host: &ServingHost, byte_level: bool) -> Arc<dyn TextTokenizer> {
    tokenizer_owner(
        Arc::new(fixture_tokenizer(byte_level)),
        |owner| owner.as_ref(),
        host,
        "utf8-test",
    )
    .unwrap()
}

// Source-free tokenizer fixture; no source identity or model is registered.
struct ObservedTokenizer {
    tokenizer: QwenTokenizer,
    host: ServingHost,
    reserved_on_drop: Arc<AtomicU64>,
}
impl Drop for ObservedTokenizer {
    fn drop(&mut self) {
        self.reserved_on_drop
            .store(self.host.snapshot().unwrap().1, Ordering::Release);
    }
}

#[test]
fn adapter_charge_covers_aligned_arc_control_and_wrapper_layout() {
    #[repr(align(64))]
    struct Aligned;
    let bytes = tokenizer_owner_bytes::<Aligned>().unwrap();
    let without_slack = bytes - ADAPTER_ALLOC_SLACK as u64;
    assert!(
        without_slack
            >= (std::mem::size_of::<TokenizerOwner<Aligned>>()
                + 2 * std::mem::size_of::<AtomicUsize>()) as u64
    );
    assert!(without_slack.is_multiple_of(std::mem::align_of::<TokenizerOwner<Aligned>>() as u64));
}

#[test]
fn cloned_adapter_retains_one_charge_and_parent_retires_before_credit() {
    let _serial = CHANNEL_TESTS.lock().unwrap();
    let host = ServingHost::new(256).unwrap();
    let baseline = host.snapshot().unwrap();
    let expected = host
        .admit(
            "retained-adapter-test",
            "tokenizer-adapter",
            AllocationClass::ImmutableWeights,
            tokenizer_owner_bytes::<ObservedTokenizer>().unwrap(),
        )
        .unwrap();
    let charged_bytes = expected.reserved_bytes().unwrap();
    drop(expected);
    assert_eq!(host.snapshot().unwrap(), baseline);

    let observed = Arc::new(AtomicU64::new(u64::MAX));
    let adapter = tokenizer_owner(
        ObservedTokenizer {
            tokenizer: fixture_tokenizer(false),
            host: host.clone(),
            reserved_on_drop: observed.clone(),
        },
        |owner| &owner.tokenizer,
        &host,
        "retained-adapter-test",
    )
    .unwrap();
    let retained = host.snapshot().unwrap();
    assert_eq!(retained.1, baseline.1 + charged_bytes);
    assert_eq!(
        retained.3, baseline.3,
        "adapter uses the original source ledger, not serving bytes"
    );
    let first = adapter.clone();
    let last = adapter.clone();
    assert_eq!(host.snapshot().unwrap(), retained);
    drop((adapter, first));
    assert_eq!(observed.load(Ordering::Acquire), u64::MAX);
    assert_eq!(last.decode(&[2, 3], true).unwrap(), "AB");
    assert_eq!(host.snapshot().unwrap(), retained);
    drop(last);
    assert_eq!(
        observed.load(Ordering::Acquire),
        retained.1,
        "parent/tokenizer retires while wrapper admission is still held"
    );
    assert_eq!(host.snapshot().unwrap(), baseline);
}

#[test]
fn adapter_host_refusal_releases_parent_without_allocating_or_leaking_credit() {
    let _serial = CHANNEL_TESTS.lock().unwrap();
    let host = ServingHost::new(256).unwrap();
    let baseline = host.snapshot().unwrap();
    let observed = Arc::new(AtomicU64::new(u64::MAX));
    let parent = ObservedTokenizer {
        tokenizer: fixture_tokenizer(false),
        host: host.clone(),
        reserved_on_drop: observed.clone(),
    };
    // Reserve the existing host's remaining capacity; no payload allocation or
    // alternate resource domain. Probe its original exact lease envelope first.
    let probe = host
        .admit("adapter-pressure-test", "hold", AllocationClass::Scratch, 0)
        .unwrap();
    let envelope = probe.reserved_bytes().unwrap();
    drop(probe);
    let pressure = host
        .admit(
            "adapter-pressure-test",
            "hold",
            AllocationClass::Scratch,
            host.snapshot().unwrap().2.checked_sub(envelope).unwrap(),
        )
        .unwrap();
    let full = host.snapshot().unwrap();
    assert_eq!(full.2, 0);
    let error = tokenizer_owner(
        parent,
        |owner| &owner.tokenizer,
        &host,
        "refused-adapter-test",
    )
    .err()
    .expect("wrapper must be refused before Arc allocation");
    assert_eq!((error.status, error.code), (503, "host_budget"));
    assert_eq!(observed.load(Ordering::Acquire), full.1);
    assert_eq!(host.snapshot().unwrap(), full);
    drop(pressure);
    assert_eq!(host.snapshot().unwrap(), baseline);
}

fn text(event: ResponseEvent) -> String {
    match event {
        ResponseEvent::Text(value) => value,
        _ => panic!("expected text event"),
    }
}

fn done(event: ResponseEvent, expected: &str, tokens: usize, reason: &str) {
    match event {
        ResponseEvent::Done {
            output,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            ..
        } => {
            assert_eq!(output, expected);
            assert_eq!(prompt_tokens, 7);
            assert_eq!(completion_tokens, tokens);
            assert_eq!(finish_reason, reason);
        }
        _ => panic!("expected one terminal event"),
    }
}

#[test]
fn incomplete_utf8_waits_then_emits_the_original_complete_decode_once() {
    let _serial = CHANNEL_TESTS.lock().unwrap();
    let host = ServingHost::new(256).unwrap();
    let baseline = host.snapshot().unwrap();
    let (sender, receiver) = mpsc::try_sync_channel(&host, "utf8-test", "complete", 1).unwrap();
    let mut output = GeneratedText::new(output_tokenizer(&host, true), sender, true, 2, 7);
    assert_eq!(
        output.present_committed(4, Instant::now()).unwrap(),
        Presentation::Ready
    );
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    assert_eq!(
        output.present_committed(5, Instant::now()).unwrap(),
        Presentation::Blocked
    );
    assert_eq!(text(receiver.recv().unwrap()), "é");
    assert_eq!(output.resume().unwrap(), Presentation::Finished);
    done(receiver.recv().unwrap(), "é", 2, "length");
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    drop((output, receiver));
    assert_eq!(host.snapshot().unwrap(), baseline);
}

#[test]
fn final_incomplete_utf8_keeps_original_replacement_flush_semantics() {
    let _serial = CHANNEL_TESTS.lock().unwrap();
    let host = ServingHost::new(256).unwrap();
    let baseline = host.snapshot().unwrap();
    let (sender, receiver) = mpsc::try_sync_channel(&host, "utf8-test", "flush", 1).unwrap();
    let mut output = GeneratedText::new(output_tokenizer(&host, true), sender, true, 1, 7);
    assert_eq!(
        output.present_committed(4, Instant::now()).unwrap(),
        Presentation::Blocked
    );
    assert_eq!(text(receiver.recv().unwrap()), "�");
    assert_eq!(output.resume().unwrap(), Presentation::Finished);
    done(receiver.recv().unwrap(), "�", 1, "length");
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    drop((output, receiver));
    assert_eq!(host.snapshot().unwrap(), baseline);
}

#[test]
fn owned_tokenizer_tuple_keeps_inventory_until_the_last_adapter_reader() {
    let _serial = CHANNEL_TESTS.lock().unwrap();
    let host = ServingHost::new(256).unwrap();
    let baseline = host.snapshot().unwrap();
    let observed = Arc::new(AtomicU64::new(u64::MAX));
    let inventory = host
        .admit(
            "tuple-test",
            "inventory",
            AllocationClass::ImmutableWeights,
            4096,
        )
        .unwrap();
    let owner = ObservedTokenizer {
        tokenizer: fixture_tokenizer(false),
        host: host.clone(),
        reserved_on_drop: observed.clone(),
    };
    let adapter = tokenizer_owner(
        (owner, inventory),
        |owner| &owner.0.tokenizer,
        &host,
        "tuple-test",
    )
    .unwrap();
    let retained = host.snapshot().unwrap();
    let last = adapter.clone();
    drop(adapter);
    assert_eq!(host.snapshot().unwrap(), retained);
    assert_eq!(observed.load(Ordering::Acquire), u64::MAX);
    assert_eq!(last.decode(&[2, 3], true).unwrap(), "AB");
    drop(last);
    assert_eq!(
        observed.load(Ordering::Acquire),
        retained.1,
        "inventory and wrapper still cover the tokenizer's destruction"
    );
    assert_eq!(host.snapshot().unwrap(), baseline);
}
