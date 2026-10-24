use super::*;
pub(super) fn parse(args: &[String], alias: Option<&str>) -> Result<Arguments> {
    if args.len() > 2048
        || args
            .iter()
            .try_fold(0usize, |n, s| n.checked_add(s.len()))
            .is_none_or(|n| n > 2 * 1024 * 1024)
    {
        return Err(Error::Limit("run argument count/bytes".into()));
    }
    let mut a = Arguments::default();
    let mut rest = args.iter();
    if let Some(alias) = alias {
        a.backend = Some(Backend::parse(match alias {
            "run-cpu" => "cpu",
            "run-cuda" => "cuda",
            "run-rocm" => "rocm",
            "run-vulkan" | "run-vulkan-amd" | "run-vulkan-nvidia" => "vulkan",
            _ => return Err(bad("unregistered run alias")),
        })?);
        a.vendor = match alias {
            "run-vulkan-amd" => Some("amd".into()),
            "run-vulkan-nvidia" => Some("nvidia".into()),
            _ => None,
        };
    }
    let mut seen = std::collections::BTreeSet::new();
    while let Some(raw) = rest.next() {
        let (flag, inline) = if raw.starts_with('-') {
            raw.split_once('=')
                .map_or((raw.as_str(), None), |(a, b)| (a, Some(b)))
        } else {
            (raw.as_str(), None)
        };
        if matches!(flag, "-h" | "--help") {
            if args.len() != 1 {
                return Err(bad("run help must stand alone"));
            }
            a.help = true;
            continue;
        }
        if !flag.starts_with('-') {
            if inline.is_some() || a.positional.replace(raw.clone()).is_some() {
                return Err(bad("at most one run positional selector"));
            }
            continue;
        }
        if matches!(flag, "--raw" | "--dry-run" | "--explain" | "--realtime") {
            if !seen.insert(flag.to_string()) {
                return Err(bad(format!("duplicate {flag}")));
            }
            let b = match inline {
                None | Some("true") => true,
                Some("false") => false,
                _ => return Err(bad(format!("{flag} accepts true/false"))),
            };
            match flag {
                "--raw" => a.task.raw = Some(b),
                "--dry-run" => a.diagnostics.dry_run = Some(b),
                "--explain" => a.diagnostics.explain = Some(b),
                _ => {
                    a.graph.push("--realtime-value".into());
                    a.graph.push(b.to_string());
                }
            }
            continue;
        }
        let v = inline
            .map(str::to_string)
            .or_else(|| rest.next().cloned())
            .ok_or_else(|| bad(format!("{flag} requires a value")))?;
        let canonical = match flag {
            "-p" => "--prompt",
            "-n" => "--max-tokens",
            "-d" => "--device",
            other => other,
        };
        let repeat = matches!(
            canonical,
            "--device"
                | "--backend"
                | "--define-model"
                | "--step"
                | "--link"
                | "--route"
                | "--draft-device"
                | "--chat-context-source"
                | "--chat-context-match"
                | "--chat-context-decision"
        );
        if !repeat && !seen.insert(canonical.into()) {
            return Err(bad(format!("duplicate {canonical}")));
        }
        match canonical {
            "--native-artifacts" => a.native_artifacts = Some(v),
            "--snapshot" => a.snapshot = Some(v),
            "--config" => a.config = Some(v),
            "--model" => a.model = Some(v),
            "--model-config" => a.model_config = Some(v),
            "--tokenizer" => a.tokenizer = Some(v),
            "--backend" => {
                let b = Backend::parse(&v)?;
                if a.backend.as_ref().is_some_and(|old| *old != b) {
                    return Err(bad("contradictory explicit backend/alias values"));
                }
                a.backend = Some(b);
            }
            "--device" => a.devices.get_or_insert_with(Vec::new).push(v),
            "--draft-device" => a.draft_device.get_or_insert_with(Vec::new).push(v),
            "--headroom" => a.headroom = Some(Headroom::parse(&v)?),
            "--entrypoint" => a.entrypoint = Some(Entrypoint::parse(&v)?),
            "--numerical" => a.numerical = Some(v),
            "--speculation" => a.speculation = Some(Speculation::parse(&v)?),
            "--prompt" => a.task.prompt = Some(v),
            "--context" => {
                a.task.context = Some(
                    v.parse()
                        .map_err(|_| bad("context requires positive integer"))?,
                )
            }
            "--max-tokens" => {
                a.task.max_tokens = Some(
                    v.parse()
                        .map_err(|_| bad("max-tokens requires positive integer"))?,
                )
            }
            "--prefill-chunk" => {
                a.task.prefill_chunk = Some(
                    v.parse()
                        .map_err(|_| bad("prefill-chunk requires positive integer"))?,
                )
            }
            "--teacher-ids" => {
                a.task.teacher_ids = Some(
                    v.split(',')
                        .map(|s| {
                            s.parse()
                                .map_err(|_| bad("invalid comma-separated teacher IDs"))
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            "--audit-logits-jsonl" => a.diagnostics.audit_logits_jsonl = Some(v),
            "--audit-layers-jsonl" => a.diagnostics.audit_layers_jsonl = Some(v),
            "--define-model"
            | "--name"
            | "--input"
            | "--output"
            | "--step"
            | "--link"
            | "--route"
            | "--chat-context-from"
            | "--chat-context-enabled"
            | "--chat-context-max-bytes"
            | "--chat-context-source"
            | "--chat-context-match"
            | "--chat-context-decision" => {
                a.graph.push(canonical.into());
                a.graph.push(v);
            }
            // Existing explicit resource flags remain available only as typed
            // operator policy in a later qualified adapter, never free-memory.
            _ => return Err(bad(format!("unknown run option {canonical}"))),
        }
    }
    if a.positional.is_some() && (a.config.is_some() || a.model.is_some()) {
        return Err(bad(
            "positional selector conflicts with explicit --config/--model; old model metadata --config migrated to --model-config",
        ));
    }
    if let Some(v) = a.positional.take() {
        if !SourceSelector::has_provider_scheme(&v)
            && Path::new(&v)
                .extension()
                .is_some_and(|e| e == "yaml" || e == "yml")
        {
            a.config = Some(v)
        } else {
            a.model = Some(v)
        }
    }
    if let Some(devices) = &a.devices {
        Devices::parse(devices)?.validate_backend(a.backend.as_ref().unwrap_or(&Backend::Auto))?;
    }
    Ok(a)
}
