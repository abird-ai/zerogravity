//! Explicit native Xet frontend; the original fetch executor owns publication.
use super::{Action, Options, invalid};
use std::collections::BTreeSet;
use zerogravity::Result;
use zerogravity::acquisition::{Provider, Selection, xet::XetOptions};

fn parse(args: &[String]) -> Result<(Options, XetOptions)> {
    if args.is_empty() || args.len() > 8192 || args.iter().any(|arg| arg.len() > 4096) {
        return Err(invalid());
    }
    let mut common = vec!["hf".to_owned(), args[0].clone()];
    let mut native = XetOptions::default();
    let mut seen = BTreeSet::new();
    let mut explicit_memory = false;
    let mut rest = args[1..].iter();
    while let Some(flag) = rest.next() {
        if flag.starts_with("--xet-") {
            if !seen.insert(flag.as_str()) {
                return Err(invalid());
            }
            let value = rest.next().ok_or_else(invalid)?;
            let number = value.parse::<u64>().map_err(|_| invalid())?;
            let count = || usize::try_from(number).map_err(|_| invalid());
            match flag.as_str() {
                "--xet-max-terms" => native.max_terms = count()?,
                "--xet-max-fetch-groups" => native.max_fetch_groups = count()?,
                "--xet-max-ranges" => native.max_ranges = count()?,
                "--xet-max-chunks" => native.max_chunks = count()?,
                "--xet-max-requests" => native.max_requests = count()?,
                "--xet-max-wire-bytes" => native.max_wire_bytes = number,
                "--xet-max-refreshes" => native.max_refreshes = count()?,
                _ => return Err(invalid()),
            }
        } else {
            match flag.as_str() {
                "--dry-run" | "--offline" => common.push(flag.clone()),
                "--cache"
                | "--ref"
                | "--revision"
                | "--file"
                | "--quant"
                | "--include"
                | "--exclude"
                | "--subfolder"
                | "--disk-limit-bytes"
                | "--memory-limit-bytes"
                | "--max-file-bytes"
                | "--max-snapshot-bytes"
                | "--max-files"
                | "--max-metadata-bytes"
                | "--max-pages"
                | "--timeout-seconds" => {
                    explicit_memory |= flag == "--memory-limit-bytes";
                    common.push(flag.clone());
                    common.push(rest.next().ok_or_else(invalid)?.clone());
                }
                _ => return Err(invalid()),
            }
        }
    }
    native.validate()?;
    let mut options = super::parse(&common)?;
    if !explicit_memory {
        options.memory = 256 * 1024 * 1024;
    }
    if !(matches!(&options.action, Action::Fetch(request) if request.provider==Provider::HuggingFace && matches!(request.selection,Selection::Files{..}))
        || matches!(&options.action,Action::Quant(request) if request.source().provider()==Provider::HuggingFace)
        || matches!(&options.action,Action::Filters(request) if request.source().provider()==Provider::HuggingFace))
        || options.unknown_size_cap.is_some()
        || (options.offline && options.dry_run)
    {
        return Err(invalid());
    }
    Ok((options, native))
}
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "help" | "--help" | "-h") {
        help();
        return Ok(());
    }
    let (options, native) = parse(args)?;
    let request = match &options.action {
        Action::Fetch(request) => super::FetchRequest::Files(request),
        Action::Quant(request) => super::FetchRequest::Quant(request),
        Action::Filters(request) => super::FetchRequest::Filters(request),
        _ => return Err(invalid()),
    };
    super::run_fetch(&options, request, Some(&native))
}
fn help() {
    println!("USAGE:
    zerogravity pull xet OWNER/REPO --revision REV --file PATH [--file PATH ...]
    zerogravity pull xet [same HF request flags] --dry-run
    zerogravity pull xet [same HF request flags] --offline [--ref DIGEST_OR_REF]

OPTIONS:
    --quant TAG                 Explicit GGUF terminal filename tag instead of --file, no fallback
    --include GLOB              Additive case-exact full-path Unicode filters, repeatable
    --exclude GLOB              Exclude from --include or --quant before family ambiguity
                                Legacy --file grammar remains component-local; these flags conflict with --file
                                Offline requires --ref SNAPSHOT_DIGEST; cached subset only
    --subfolder PATH            Original bounded HF exact/glob selection and model closure
    --cache ABS_PATH --ref NAME Original XDG cache/ref publisher (default HF request ref)
    --dry-run                   Hub metadata/discovery only; no CAS grants or signed ranges
    --offline                   Full verified cached HF data before credentials/client/admission/signals
    --disk-limit-bytes N --memory-limit-bytes N (native default256MiB)
    --max-file-bytes N --max-snapshot-bytes N --max-files N
    --max-metadata-bytes N --max-pages N --timeout-seconds N
    --xet-max-terms N            Default4096; hard65536
    --xet-max-fetch-groups N     Default4096; hard65536
    --xet-max-ranges N           Default8192; hard65536
    --xet-max-chunks N           Default262144; hard1048576
    --xet-max-requests N         Default16384 actual attempts; hard65536
    --xet-max-wire-bytes N       Default64GiB; hard1TiB
    --xet-max-refreshes N        Default2; range0..8 (combined grant/signed-URL reissues)

All other limits are positive; library validation applies before I/O. Credentials
use HF_TOKEN only through the original origin-scoped owner. Hub/CAS origins cannot
be overridden by CLI. Exact file sizes and ordinary required model closure remain;
GitHub/releases/components and unknown-size caps refuse. Explicit native selection
has no silent compatibility fallback. Online SIGINT/SIGTERM, byte progress, bounded
admission, full raw SHA/length/EOF and typed completion use the shared fetch path.
Offline cache verification does not attest native Xet protocol execution, remote
freshness or current authorization. Native scratch and the original cache transaction
retain concurrent admission envelopes; explicit memory limits are preserved. The cap
covers process transfer staging, not all host RAM. No inference support or licensing
decision follows from acquired data.");
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(extra: &[&str]) -> Vec<String> {
        ["owner/repo", "--revision", "main", "--file", "a.txt"]
            .into_iter()
            .chain(extra.iter().copied())
            .map(str::to_owned)
            .collect()
    }
    #[test]
    fn original_selector_values_are_not_reinterpreted_as_native_flags() {
        let input = [
            "owner/repo",
            "--revision",
            "main",
            "--file",
            "--xet-max-terms",
            "--xet-max-refreshes",
            "0",
        ]
        .map(str::to_owned);
        let (options, native) = parse(&input).unwrap();
        assert_eq!(native.max_refreshes, 0);
        let Action::Fetch(request) = options.action else {
            panic!("expected fetch");
        };
        assert!(
            matches!(request.selection,Selection::Files{paths,..} if paths==["--xet-max-terms"])
        );
    }
    #[test]
    fn native_memory_default_preserves_explicit_limits_and_ordinary_default() {
        assert_eq!(parse(&args(&[])).unwrap().0.memory, 256 * 1024 * 1024);
        for bytes in [1, 128 * 1024 * 1024] {
            assert_eq!(
                parse(&args(&["--memory-limit-bytes", &bytes.to_string()]))
                    .unwrap()
                    .0
                    .memory,
                bytes
            );
        }
        let value = [
            "owner/repo",
            "--revision",
            "main",
            "--file",
            "--memory-limit-bytes",
        ]
        .map(str::to_owned);
        assert_eq!(parse(&value).unwrap().0.memory, 256 * 1024 * 1024);
        let ordinary =
            ["hf", "owner/repo", "--revision", "main", "--file", "a.txt"].map(str::to_owned);
        assert_eq!(
            super::super::parse(&ordinary).unwrap().memory,
            128 * 1024 * 1024
        );
    }
    #[test]
    fn native_numeric_bounds_and_forbidden_surfaces_fail_before_io() {
        for extra in [
            vec!["--xet-max-terms", "0"],
            vec!["--xet-max-requests", "65537"],
            vec!["--xet-max-chunks", "1048577"],
            vec!["--xet-max-wire-bytes", "1099511627777"],
            vec!["--xet-max-refreshes", "9"],
            vec!["--xet-max-terms", "1", "--xet-max-terms", "2"],
            vec!["--unknown-size-cap-bytes", "10"],
            vec!["--hub-origin", "https://NEVER_ECHO"],
            vec!["--release", "tag"],
            vec!["--asset", "asset"],
            vec!["--token", "NEVER_ECHO"],
        ] {
            assert!(parse(&args(&extra)).is_err());
        }
    }
}
