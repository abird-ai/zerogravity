//! Developer-only resumable whole-model CPU context executor.
//!
//! Built ONLY with --features diagnostic-cpu-checkpoint. This is not a
//! production/session migration protocol. A separate qualification driver
//! binds exact locally hashed GGUF and full raw input token stream.
//!
//! Subcommands:
//!   tokenize <tokenizer.json> <UTF8_prompt_file> <ids.json>
//!   step <model.gguf> <config.json> <ids.json> <snapshot.bin> <result.json>
//!        <model_sha256> <input_tokens_sha256> <chunk_tokens>
//!
//! Each step processes AT MOST chunk_tokens of the real trained model.
//! On intermediate steps there is one atomic commit of complete recurrent
//! and KV domains and a versioned/identity-bound checkpoint, no logits.
//! On the terminal step the correct next-token logits are preserved to
//! JSON with exact-source provenance. Validation driver independently
//! compares these to source llama.cpp after all chunks finish.
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use zerogravity::math::top_logit_ranks;
use zerogravity::qwen35::QwenDenseCpu;
use zerogravity::tokenize::QwenTokenizer;
use zerogravity::{Error, Result};

const MAX_INPUT_TOKENS: usize = 32_768;

fn digest(text: &str) -> Result<[u8; 32]> {
    if text.len() != 64 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::Malformed(
            "SHA256 must be exactly 64 hex digits".into(),
        ));
    }
    let mut result = [0u8; 32];
    for (i, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)
            .map_err(|_| Error::Malformed("invalid SHA256 byte".into()))?;
    }
    Ok(result)
}

fn tokens_from_file(path: &Path) -> Result<Vec<u32>> {
    let file = File::open(path)?;
    if file.metadata()?.len() > 450_000 {
        return Err(Error::Limit(
            "raw token list exceeds local diagnostic size".into(),
        ));
    }
    #[derive(serde::Deserialize)]
    struct RawTokenIds {
        raw_tokens: Vec<u32>,
        count: usize,
    }
    let encoded: RawTokenIds = serde_json::from_reader(BufReader::new(file))
        .map_err(|e| Error::Malformed(format!("raw token IDs: {e}")))?;
    if encoded.count != encoded.raw_tokens.len() {
        return Err(Error::Malformed("raw token count is inconsistent".into()));
    }
    let tokens = encoded.raw_tokens;
    if tokens.is_empty() || tokens.len() > MAX_INPUT_TOKENS {
        return Err(Error::Limit("raw input must be 1..32768 tokens".into()));
    }
    Ok(tokens)
}

fn atomic_json(path: &Path, data: &serde_json::Value) -> Result<()> {
    let tmp = path.with_extension("json.part");
    let result = (|| {
        let f = File::create(&tmp)?;
        let mut w = BufWriter::new(f);
        serde_json::to_writer(&mut w, data)
            .map_err(|e| Error::Malformed(format!("encode result JSON: {e}")))?;
        w.write_all(b"\n")?;
        w.flush()?;
        w.get_ref().sync_all()?;
        Result::Ok(())
    })();
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    fs::rename(tmp, path)?;
    Ok(())
}

fn tokenize(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        return Err(Error::Malformed(
            "tokenize tokenizer.json input.utf8 output-ids.json".into(),
        ));
    }
    let tokenizer = QwenTokenizer::from_file(Path::new(&args[2]))?;
    let mut content = String::new();
    File::open(&args[3])?
        .take(1024 * 1024 + 1)
        .read_to_string(&mut content)?;
    if content.len() > 1024 * 1024 {
        return Err(Error::Limit("raw prompt exceeds 1 MiB".into()));
    }
    let ids = tokenizer.prompt_tokens(&content, false)?;
    if ids.is_empty() || ids.len() > MAX_INPUT_TOKENS {
        return Err(Error::Limit(
            "raw tokenizer output exceeds 32768 tokens".into(),
        ));
    }
    atomic_json(
        Path::new(&args[4]),
        &serde_json::json!({"raw_tokens":ids, "count":ids.len()}),
    )?;
    println!("CPU_CONTEXT_TOKENIZED count={}", ids.len());
    Ok(())
}

fn step(args: &[String]) -> Result<()> {
    if args.len() != 10 {
        return Err(Error::Malformed(
            "step GGUF config.json tokens.json checkpoint.bin result.json model_sha256 tokens_sha256 chunk_size"
                .into(),
        ));
    }
    let model = PathBuf::from(&args[2]);
    let config = PathBuf::from(&args[3]);
    let ids_file = PathBuf::from(&args[4]);
    let snapshot = PathBuf::from(&args[5]);
    let result = PathBuf::from(&args[6]);
    let model_digest = digest(&args[7])?;
    let ids_digest = digest(&args[8])?;
    let chunk_size = args[9]
        .parse::<usize>()
        .map_err(|_| Error::Malformed("invalid chunk size".into()))?;
    if !(1..=256).contains(&chunk_size) {
        return Err(Error::Limit("diagnostic chunk size must be 1..256".into()));
    }

    let ids = tokens_from_file(&ids_file)?;
    if result.exists() {
        return Err(Error::State(
            "final CPU context result already exists; step must not overwrite it".into(),
        ));
    }
    let mut model_runner = QwenDenseCpu::open(&model, &config)?;
    if snapshot.exists() {
        model_runner.restore_cpu_checkpoint(&snapshot, model_digest, ids_digest)?;
    }
    let cursor = model_runner.position();
    if cursor >= ids.len() {
        return Err(Error::State(
            "snapshot already at final position but next-logits result is missing".into(),
        ));
    }
    let end = (cursor + chunk_size).min(ids.len());
    let chunk = &ids[cursor..end];
    let terminal = end == ids.len();

    if terminal {
        let logits = model_runner.prefill_transaction(chunk)?;
        let top = top_logit_ranks(&logits, 20)?;
        let selected = top[0].id;
        let record = serde_json::json!({
            "schema": "zg-qwen-cpu-context-f32-v1",
            "source_model_sha256": args[7].to_ascii_lowercase(),
            "raw_input_sha256": args[8].to_ascii_lowercase(),
            "actual_reference_prompt_tokens": ids.len(),
            "position": model_runner.position(),
            "rust_published_roots": model_runner.root_id(),
            "state_fingerprint": model_runner.state_fingerprint(),
            "state_bytes": model_runner.state_bytes(),
            "selected_token": selected,
            "top": top,
        });
        // Write a terminal result only after entire trained context commits.
        // A partially completed snapshot never appears as a final report.
        atomic_json(&result, &record)?;
        println!(
            "CPU_CONTEXT_STEP_DONE position={} source_chunks_complete=true selected_token={selected}",
            model_runner.position()
        );
    } else {
        model_runner.prefill_chunk_without_logits(chunk)?;
        model_runner.save_cpu_checkpoint(&snapshot, model_digest, ids_digest)?;
        println!(
            "CPU_CONTEXT_STEP_SAVED position={}/{} root={} state_bytes={}",
            model_runner.position(),
            ids.len(),
            model_runner.root_id(),
            model_runner.state_bytes()
        );
    }
    Ok(())
}

/// Observe source-trained stage values at the final input token after
/// restoring an EXISTING source-bound prefix checkpoint. This never writes
/// the checkpoint and cannot overwrite the accepted final result.
fn trace(args: &[String]) -> Result<()> {
    if args.len() != 9 {
        return Err(Error::Malformed(
            "trace GGUF config.json raw-ids.json prefix-checkpoint.bin output-layers.jsonl model_sha256 tokens_sha256".into()
        ));
    }
    let ids = tokens_from_file(Path::new(&args[4]))?;
    let mut model = QwenDenseCpu::open(Path::new(&args[2]), Path::new(&args[3]))?;
    model.restore_cpu_checkpoint(Path::new(&args[5]), digest(&args[7])?, digest(&args[8])?)?;
    let position = model.position();
    if position >= ids.len() {
        return Err(Error::State(
            "long-context layer observation must start from a committed proper prefix".into(),
        ));
    }
    let (logits, stages) = model.prefill_transaction_with_layer_trace(&ids[position..])?;
    if stages.len() != 146 || model.position() != ids.len() {
        return Err(Error::State(
            "full trained-model final-token observation incomplete".into(),
        ));
    }
    let top = top_logit_ranks(&logits, 20)?;
    if !top[0].logprob.is_finite() {
        return Err(Error::Malformed(
            "nonfinite final-token diagnostic logits".into(),
        ));
    }
    let path = Path::new(&args[6]);
    if path.exists() {
        return Err(Error::State(
            "long-context source trace already exists; refusing overwrite".into(),
        ));
    }
    let temp = path.with_extension("jsonl.part");
    let persisted = (|| -> Result<()> {
        let file = File::create(&temp)?;
        let mut writer = BufWriter::new(file);
        for stage in stages {
            serde_json::to_writer(&mut writer, &stage)
                .map_err(|e| Error::Malformed(format!("encode full-model stage: {e}")))?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        Ok(())
    })();
    if let Err(e) = persisted {
        let _ = fs::remove_file(&temp);
        return Err(e);
    }
    fs::rename(temp, path)?;
    println!(
        "CPU_CONTEXT_FINAL_LAYER_OBSERVATION source_prefix={} total_tokens={}          stages=146 top1={} committed_root={} final_state_fingerprint={} output={}",
        position,
        ids.len(),
        top[0].id,
        model.root_id(),
        model.state_fingerprint(),
        path.display()
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let operation = match args.get(1).map(String::as_str) {
        Some("tokenize") => tokenize(&args),
        Some("step") => step(&args),
        Some("trace") => trace(&args),
        _ => Err(Error::Malformed(
            "usage: cpu_context_step tokenize|step [arguments...]".into(),
        )),
    };
    if let Err(err) = operation {
        eprintln!("CPU_CONTEXT_STEP_ERROR={err}");
        std::process::exit(2);
    }
}
