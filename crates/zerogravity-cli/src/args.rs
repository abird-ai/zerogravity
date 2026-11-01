//! CLI argument syntax; no source/model execution.

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Command {
    Help,
    Version,
    Inspect(String),
    Run {
        args: Vec<String>,
        alias: Option<String>,
    },
    Devices(Vec<String>),
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
        Some(first) => match first.as_ref() {
            "-h" | "--help" | "help" => Command::Help,
            "-V" | "--version" | "version" => Command::Version,
            "inspect" => match (args.next(), args.next()) {
                (Some(path), None) if !path.as_ref().is_empty() => {
                    Command::Inspect(path.as_ref().to_owned())
                }
                _ => Command::Invalid("inspect requires exactly one local path".into()),
            },
            "run" | "run-cpu" | "run-cuda" | "run-rocm" | "run-vulkan" | "run-vulkan-amd"
            | "run-vulkan-nvidia" => Command::Run {
                alias: (first.as_ref() != "run").then(|| first.as_ref().to_owned()),
                args: args.map(|arg| arg.as_ref().to_owned()).collect(),
            },
            "devices" => Command::Devices(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "pull" => Command::Pull(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "models" => Command::Models(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "pipeline" => Command::Pipeline(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "cache" => Command::CacheManage(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "run-cache" => Command::CacheRun(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "serve" => Command::Serve(args.map(|arg| arg.as_ref().to_owned()).collect()),
            "config" => Command::NotImplemented(first.as_ref().to_owned()),
            other => Command::Unknown(other.to_owned()),
        },
    }
}
