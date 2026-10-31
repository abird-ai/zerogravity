//! CLI argument syntax; no source/model execution.
use super::GpuBackend;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Command {
    Help,
    Version,
    Inspect(String),
    Run {
        model: String,
        prompt: Option<String>,
        max_tokens: usize,
        raw: bool,
        tokenizer: Option<String>,
        config: Option<String>,
        audit_jsonl: Option<String>,
        audit_layers_jsonl: Option<String>,
        teacher_ids: Option<Vec<u32>>,
        prefill_chunk: usize,
    },
    GpuRun {
        backend: GpuBackend,
        args: Vec<String>,
    },
    Pull(Vec<String>),
    Models(Vec<String>),
    Pipeline(Vec<String>),
    Serve(Vec<String>),
    CacheRun(Vec<String>),
    CacheManage(Vec<String>),
    NotImplemented(String),
    Invalid(String),
    Unknown(String),
}

pub(super) fn parse<I, S>(args: I) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    match args.next() {
        None => Command::Help,
        Some(first) => {
            match first.as_ref() {
                "-h" | "--help" | "help" => Command::Help,
                "-V" | "--version" | "version" => Command::Version,
                "inspect" => match (args.next(), args.next()) {
                    (Some(path), None) if !path.as_ref().is_empty() => {
                        Command::Inspect(path.as_ref().to_owned())
                    }
                    _ => Command::Invalid("inspect requires exactly one local path".into()),
                },
                "run" => {
                    let Some(model) = args.next() else {
                        return Command::Invalid("run requires a local model GGUF file".into());
                    };
                    let model = model.as_ref().to_owned();
                    if model.is_empty() {
                        return Command::Invalid("missing model path".into());
                    }
                    let mut prompt = None;
                    let mut max_tokens = 16usize;
                    let mut raw = false;
                    let mut tokenizer = None;
                    let mut config = None;
                    let mut audit_jsonl = None;
                    let mut audit_layers_jsonl = None;
                    let mut teacher_ids = None;
                    let mut prefill_chunk = 128usize;
                    while let Some(flag) = args.next() {
                        let value = flag.as_ref();
                        match value {
                            "--raw" => raw = true,
                            "--prompt"
                            | "-p"
                            | "--max-tokens"
                            | "-n"
                            | "--tokenizer"
                            | "--config"
                            | "--audit-logits-jsonl"
                            | "--audit-layers-jsonl"
                            | "--teacher-ids"
                            | "--prefill-chunk" => {
                                let Some(next) = args.next() else {
                                    return Command::Invalid(format!("{value} requires a value"));
                                };
                                let n = next.as_ref().to_owned();
                                match value {
                                    "--prompt" | "-p" => prompt = Some(n),
                                    "--max-tokens" | "-n" => match n.parse::<usize>() {
                                        Ok(x) if (1..=128).contains(&x) => max_tokens = x,
                                        _ => {
                                            return Command::Invalid(
                                                "max tokens must be in 1..128".into(),
                                            );
                                        }
                                    },
                                    "--tokenizer" => tokenizer = Some(n),
                                    "--config" => config = Some(n),
                                    "--audit-logits-jsonl" => audit_jsonl = Some(n),
                                    "--audit-layers-jsonl" => audit_layers_jsonl = Some(n),
                                    "--prefill-chunk" => match n.parse::<usize>() {
                                        Ok(value) if (1..=32768).contains(&value) => {
                                            prefill_chunk = value
                                        }
                                        _ => {
                                            return Command::Invalid(
                                                "prefill chunk must be 1..32768".into(),
                                            );
                                        }
                                    },
                                    "--teacher-ids" => {
                                        let ids = n
                                            .split(',')
                                            .map(str::parse::<u32>)
                                            .collect::<std::result::Result<Vec<_>, _>>();
                                        match ids {
                                        Ok(v) if !v.is_empty() && v.len() <= 128 => teacher_ids = Some(v),
                                        _ => return Command::Invalid("invalid comma-separated teacher token IDs (max 128)".into()),
                                    }
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            _ => return Command::Invalid(format!("unknown run option '{value}'")),
                        }
                    }
                    if let Some(ref ids) = teacher_ids
                        && (ids.len() != max_tokens || audit_jsonl.is_none())
                    {
                        return Command::Invalid(
                            "--teacher-ids requires --audit-logits-jsonl and -n matching teacher count".into()
                        );
                    }
                    Command::Run {
                        model,
                        prompt,
                        max_tokens,
                        raw,
                        tokenizer,
                        config,
                        audit_jsonl,
                        audit_layers_jsonl,
                        teacher_ids,
                        prefill_chunk,
                    }
                }
                "run-vulkan-amd" | "run-vulkan-nvidia" | "run-cuda" | "run-rocm" => {
                    let backend = match first.as_ref() {
                        "run-vulkan-amd" => GpuBackend::VulkanAmd,
                        "run-vulkan-nvidia" => GpuBackend::VulkanNvidia,
                        "run-cuda" => GpuBackend::Cuda,
                        _ => GpuBackend::Hip,
                    };
                    Command::GpuRun {
                        backend,
                        args: args.map(|arg| arg.as_ref().to_owned()).collect(),
                    }
                }
                "pull" => Command::Pull(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "models" => Command::Models(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "pipeline" => Command::Pipeline(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "cache" => Command::CacheManage(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "run-cache" => Command::CacheRun(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "serve" => Command::Serve(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "config" => Command::NotImplemented(first.as_ref().to_owned()),
                other => Command::Unknown(other.to_owned()),
            }
        }
    }
}
