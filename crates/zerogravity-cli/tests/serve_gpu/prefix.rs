//! Genuine HTTP source IDs; state-only progress is visible only in health.
use super::*;

fn queued_prefix_group(
    server: &OwnedServer,
    bodies: &[Value],
    bindings: &[String],
) -> Proof<Vec<Response>> {
    if bodies.len() != bindings.len() || !matches!(bodies.len(), 2 | 4 | 8) {
        return Err("native prefix HTTP requires exact bounded rows/identities".into());
    }
    let before = server.health()?["admission"].clone();
    let attempts = counter(&before, "gpu_step_attempts")?;
    let tokens = counter(&before, "completed_model_token_steps")?;
    let gate = open_request(
        server.address,
        "POST",
        "/v1/completions",
        Some(&payload(
            &server.aliases[0],
            &"An original GPU teacher prefix gate. ".repeat(12),
            8,
            true,
        )),
    )?;
    server.wait_health(|health| {
        health["active_sessions"] == 1
            && health["gpu_step_attempts"]
                .as_u64()
                .is_some_and(|n| n > attempts)
            && health["completed_model_token_steps"].as_u64() == Some(tokens)
    })?;
    let streams = bodies
        .iter()
        .zip(bindings)
        .map(|(body, identity)| {
            open_bound_request(
                server.address,
                "POST",
                "/v1/completions",
                Some(body),
                Some(identity),
            )
        })
        .collect::<Proof<Vec<_>>>()?;
    server.wait_health_for(Duration::from_secs(5), |health| {
        health["queued_requests"] == bodies.len()
            && health["completed_model_token_steps"].as_u64() == Some(tokens)
    })?;
    let _ = gate.shutdown(Shutdown::Both);
    drop(gate);
    streams.into_iter().map(Response::read).collect()
}

fn qualify_prefix_http(name: &str, width: usize) -> Proof<()> {
    let _serial = GPU_HTTP_LOCK
        .lock()
        .map_err(|_| "physical HTTP lock poisoned")?;
    if cfg!(debug_assertions) {
        return Err("native prefix HTTP requires --release".into());
    }
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/models")
        .join(name);
    let path = directory.join("model.gguf");
    let source =
        LoadedQwenModel::open(&path, &directory.join("config.json")).map_err(|e| e.to_string())?;
    let tokenizer =
        QwenTokenizer::from_file(&directory.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let prompts = [
        "Continue counting all whole numbers: one, two, three, four, five,",
        "Write the next number in this sequence: 1, 2, 3, 4, 5, 6, 7,",
        "A small bird flies above a very wide green field at sunrise.",
        "The following list of colors begins red, green, blue, yellow, orange,",
        "The answer to this simple question is going to use several words.",
        "There are many independent requests in this original teacher prefix.",
        "A long source prompt contains several independent original tokens.",
        "After the first line comes a second line and then a third line.",
    ];
    let oracles = prompts
        .iter()
        .take(width)
        .map(|prompt| Oracle::compute(&source, &tokenizer, prompt))
        .collect::<Proof<Vec<_>>>()?;
    if oracles
        .iter()
        .any(|oracle| oracle.prompt_tokens < 3 || oracle.ids.is_empty())
        || oracles
            .iter()
            .map(|oracle| oracle.prompt_tokens)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            < 2
    {
        return Err(format!(
            "native prefix fixture needs multi-token ragged prompts and nonempty non-EOS output: lengths={:?} outputs={:?}",
            oracles.iter().map(|o| o.prompt_tokens).collect::<Vec<_>>(),
            oracles.iter().map(|o| o.ids.len()).collect::<Vec<_>>()
        ));
    }
    let server = OwnedServer::start_configured(&path, width, true)?;
    let initial = server.health()?["admission"].clone();
    let limits = (0..width).map(|row| 2 + row % 3).collect::<Vec<_>>();
    let bodies = oracles
        .iter()
        .zip(&limits)
        .enumerate()
        .map(|(row, (oracle, &limit))| {
            let mut body = payload(&server.aliases[row % 2], &oracle.prompt, limit, true);
            body["return_token_ids"] = json!(true);
            body
        })
        .collect::<Vec<_>>();
    let identities = (0..width)
        .map(|row| format!("native-prefix:{row}"))
        .collect::<Vec<_>>();
    let responses = queued_prefix_group(&server, &bodies, &identities)?;
    for (row, ((response, oracle), &limit)) in
        responses.into_iter().zip(&oracles).zip(&limits).enumerate()
    {
        let count = oracle.ids.len().min(limit);
        let records = observed_sse_completion(
            response,
            &server.aliases[row % 2],
            &oracle.expected(&tokenizer, limit)?,
            &oracle.ids[..count],
            oracle.prompt_tokens,
            &identities[row],
            None,
        )?;
        // The first observable ID belongs to the final prompt-token root.
        // Prefix teacher roots cannot manufacture SSE IDs or completion usage.
        let first = records
            .first()
            .ok_or("native prefix HTTP lacks first non-EOS generated record")?;
        if first["gpu"]["producer_frontier"] != oracle.prompt_tokens {
            return Err("HTTP emitted a generated ID from a teacher prefix root".into());
        }
    }
    let health = server
        .wait_health(|health| health["active_sessions"] == 0 && health["queued_requests"] == 0)?;
    physical_accounting(&health)?;
    // Ragged tails may finish through B1 before a sibling reaches generation.
    if counter(&health, "gpu_fused_state_only_batches")?
        <= counter(&initial, "gpu_fused_state_only_batches")?
        || counter(&health, "gpu_fused_state_only_rows")?
            < counter(&initial, "gpu_fused_state_only_rows")? + width as u64
    {
        return Err(format!(
            "HTTP lacks actual native teacher-prefix receipts: {health}"
        ));
    }
    println!(
        "P12_NATIVE_PREFIX_HTTP_PASS source={name} rows={width} actual_native_state_only_rows=true ragged_prompts=true source_ID_text_usage_terminal_DONE=exact prefix_not_presented=true health={health}"
    );
    Ok(())
}
macro_rules! case {
    ($test:ident, $name:literal, $rows:literal) => {
        #[test]
        #[ignore = "release real CUDA native teacher-prefix HTTP; serial"]
        fn $test() -> Proof<()> {
            qualify_prefix_http($name, $rows)
        }
    };
}
case!(original_qwen35_08b_native_prefix_b2_http, "qwen35-0.8b", 2);
case!(original_qwen35_08b_native_prefix_b4_http, "qwen35-0.8b", 4);
case!(original_qwen35_08b_native_prefix_b8_http, "qwen35-0.8b", 8);
case!(original_qwen35_2b_native_prefix_b2_http, "qwen35-2b", 2);
case!(original_qwen35_2b_native_prefix_b4_http, "qwen35-2b", 4);
case!(original_qwen35_2b_native_prefix_b8_http, "qwen35-2b", 8);
