//! Opt-in source-trained Rust numerical activation tracer.
//! Example: cargo run --release --example trace_tokens -- model.gguf config.json output.jsonl 48 16451 ...
//! Does NOT call or link an independent reference runtime.
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
use zerogravity::qwen35::QwenDenseCpu;
fn main() {
    let args = env::args().collect::<Vec<_>>();
    assert!(
        args.len() >= 5,
        "usage: trace_tokens model.gguf config.json output.jsonl raw-token-id [ids...]"
    );
    let mut model =
        QwenDenseCpu::open(Path::new(&args[1]), Path::new(&args[2])).expect("source model");
    let mut out = BufWriter::new(File::create(&args[3]).expect("trace output file"));
    let only_last = env::var("ZG_TRACE_ONLY_LAST").ok().as_deref() == Some("1");
    let full_trace = env::var("ZG_TRACE_ALL_LAYERS").ok().as_deref() == Some("1");
    let traced_layer = env::var("ZG_TRACE_LAYER")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(0);
    for (idx, text) in args[4..].iter().enumerate() {
        let token = text.parse::<u32>().expect("token integer");
        let (_, trace) = model
            .prefill_transaction_with_layer_trace(&[token])
            .expect("CPU token");
        for observation in trace {
            if (!only_last || idx + 5 == args.len())
                && (full_trace
                    || observation.layer == traced_layer
                    || observation.stage == "embedding")
            {
                serde_json::to_writer(&mut out, &observation).expect("encode trace");
                writeln!(&mut out).expect("end record");
            }
        }
    }
    out.flush().expect("flush trace");
    eprintln!(
        "RUST_TRACE_TOKENS={} PUBLISHED_ROOT={} POSITION={}",
        args.len() - 4,
        model.root_id(),
        model.position()
    );
}
