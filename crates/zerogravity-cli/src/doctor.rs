//! Cold local diagnostics. Configuration, inventory and cache owners remain
//! authoritative; no runtime loading, provider lookup, allocation or writes.
use crate::args::Format;
use serde::Serialize;
use serde_json::{Value, json};
use std::path::Path;
use zerogravity::{
    Error, Result, configuration, devices,
    run_policy::{Request, SourceSelector},
};

#[derive(Serialize)]
struct Check {
    component: String,
    status: &'static str,
    message: &'static str,
    details: Value,
    #[serde(skip)]
    lines: Vec<String>,
}
#[derive(Default)]
struct Report {
    checks: Vec<Check>,
}
impl Report {
    fn add(
        &mut self,
        component: impl Into<String>,
        status: &'static str,
        message: &'static str,
        details: Value,
    ) {
        self.checks.push(Check {
            component: component.into(),
            status,
            message,
            details,
            lines: Vec::new(),
        });
    }
    fn lines(&mut self, lines: impl IntoIterator<Item = String>) {
        self.checks
            .last_mut()
            .expect("check added")
            .lines
            .extend(lines);
    }
    fn error(&mut self, component: impl Into<String>, message: &'static str) {
        self.add(component, "error", message, Value::Null);
    }
    fn finish(self, format: Format) -> Result<()> {
        let failed = self.checks.iter().any(|row| row.status == "error");
        let warned = self.checks.iter().any(|row| row.status == "warning");
        let status = if failed {
            "error"
        } else if warned {
            "warning"
        } else {
            "ok"
        };
        if format == Format::Json {
            crate::output::json(&json!({
                "schema": "zerogravity.doctor.v1", "status": status,
                "checks": self.checks, "execution_qualified": false,
            }))?;
        } else {
            println!("ZeroGravity doctor — cold local observations");
            for row in &self.checks {
                println!(
                    "{:<12} {:<24} {}",
                    row.status.to_uppercase(),
                    row.component,
                    row.message
                );
                for line in &row.lines {
                    println!("    {line}");
                }
            }
            println!("No model loaded; no execution qualification or network access.");
        }
        if failed {
            Err(Error::State(
                "local diagnostic problems; see doctor report".into(),
            ))
        } else {
            Ok(())
        }
    }
}

fn config(report: &mut Report, selection: Option<(&str, &str)>) -> Option<Request> {
    match crate::config_cli::read_selected_document(selection) {
        Ok(None) => {
            report.add(
                "config",
                "ok",
                "Implicit document absent; built-in defaults",
                Value::Null,
            );
            None
        }
        Ok(Some((path, doc))) => match configuration::inspect_policy(&path, &doc) {
            Ok(request) => {
                crate::output::install_document(doc.ui.unwrap_or_default());
                report.add("config", "ok", "Selected document and policy valid", json!({
                    "selection": match selection { Some(("--profile", _)) => "profile", Some(_) => "explicit_file", None => "default" },
                    "backend": request.backend.label(), "models": request.graph.models.len(),
                    "pipelines": request.graph.pipelines.len()
                }));
                report.lines([format!(
                    "Backend: {} · {} models · {} pipelines",
                    request.backend.label(),
                    request.graph.models.len(),
                    request.graph.pipelines.len()
                )]);
                Some(request)
            }
            Err(_) => {
                report.error(
                    "config",
                    "Selected document policy invalid; use config validate",
                );
                None
            }
        },
        Err(_) => {
            report.error(
                "config",
                "Selected document unavailable or invalid; use config validate",
            );
            None
        }
    }
}

fn local_path(path: &Path, directory: Option<bool>) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| match directory {
        Some(true) => meta.is_dir(),
        Some(false) => meta.is_file(),
        None => meta.is_dir() || meta.is_file(),
    })
}

fn source_paths(report: &mut Report, request: &Request) {
    for model in &request.graph.models {
        let mut unavailable = Vec::new();
        let mut deferred = false;
        if let Some(source) = &model.source {
            match SourceSelector::parse(source) {
                Ok(SourceSelector::Local(path)) if !local_path(&path, None) => {
                    unavailable.push("source")
                }
                Ok(SourceSelector::Local(_)) => {}
                Ok(_) => deferred = true, // no remote lookup or source resolution
                Err(_) => unavailable.push("source selector"),
            }
        } else {
            deferred = true;
        }
        for (name, path, directory) in [
            ("model_config", &model.model_config, false),
            ("tokenizer", &model.tokenizer, false),
            ("native_artifacts", &model.native_artifacts, true),
        ] {
            if path
                .as_ref()
                .is_some_and(|path| !local_path(Path::new(path), Some(directory)))
            {
                unavailable.push(name);
            }
        }
        report.add(
            format!("model:{}", model.id),
            if !unavailable.is_empty() {
                "warning"
            } else if deferred {
                "unprobed"
            } else {
                "ok"
            },
            if !unavailable.is_empty() {
                "Declared local path unavailable or wrong type"
            } else if deferred {
                "Source selection deferred; no payload verification"
            } else {
                "Declared local paths present; payload not verified"
            },
            json!({"unavailable_fields": unavailable}),
        );
        if !unavailable.is_empty() {
            report.lines([unavailable.join(", ")]);
        }
    }
}

fn hardware(report: &mut Report, request: Option<&Request>) {
    let inventory = match devices::discover() {
        Ok(inventory) => inventory,
        Err(_) => {
            report.error("devices", "Local device discovery failed");
            return;
        }
    };
    report.add(
        "devices",
        "ok",
        "Physical inventory; availability is observational",
        json!({
            "observed_unix_ms": inventory.observed_unix_ms.to_string(),
            "devices": inventory.devices,
        }),
    );
    report.lines(
        inventory
            .devices
            .iter()
            .map(crate::run_cli::devices::display),
    );
    for (name, compiled, vendor) in [
        ("cpu", cfg!(feature = "cpu"), Some("cpu")),
        ("cuda", cfg!(feature = "cuda"), Some("nvidia")),
        ("rocm", cfg!(feature = "hip"), Some("amd")),
        ("vulkan", cfg!(feature = "vulkan"), None),
    ] {
        let candidates = inventory
            .devices
            .iter()
            .filter(|row| vendor.map_or(row.class == "gpu", |vendor| row.vendor == vendor))
            .collect::<Vec<_>>();
        let observed = candidates.iter().any(|row| {
            row.backends.iter().any(|backend| {
                backend == name || backend.strip_suffix(":physical-binding-observed") == Some(name)
            })
        });
        let (status, message) = if !compiled {
            ("unavailable", "Backend not compiled")
        } else if candidates.is_empty() {
            ("unavailable", "No matching physical observation")
        } else if observed {
            ("ok", "Backend metadata observed; execution not qualified")
        } else {
            (
                "unprobed",
                "Backend compiled; runtime binding not established by cold inventory",
            )
        };
        report.add(name, status, message, json!({
            "compiled": compiled, "physical_candidates": candidates.iter().map(|row| &row.id).collect::<Vec<_>>(),
            "runtime_binding_observed": observed,
        }));
        if !candidates.is_empty() {
            report.lines([candidates
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")]);
        }
    }
    if let Some(request) = request {
        let backend = request.backend.label();
        if report
            .checks
            .iter()
            .any(|row| row.component == backend && row.status == "unavailable")
        {
            report.add(
                "backend policy",
                "warning",
                "Configured backend is not compiled or lacks matching hardware",
                Value::Null,
            );
        }
        match devices::allowed(request, &inventory) {
            Ok(rows) if !rows.is_empty() => {
                report.add(
                    "device policy",
                    "ok",
                    "Selected physical policy resolves; no source admission",
                    json!({"candidates": rows.iter().map(|row| &row.id).collect::<Vec<_>>() }),
                );
                report.lines([rows
                    .iter()
                    .map(|row| row.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")]);
            }
            Ok(_) | Err(_) => report.add(
                "device policy",
                "warning",
                "Selected policy has no usable observation or contradicts selected hardware",
                Value::Null,
            ),
        }
        if let Some(zerogravity::run_policy::Devices::Ordered(selectors)) = &request.draft_device
            && selectors
                .iter()
                .any(|device| devices::observe_selected(&inventory, device).is_err())
        {
            report.add(
                "draft device",
                "warning",
                "A configured draft device is not observed",
                Value::Null,
            );
        }
    }
}

#[cfg(all(feature = "cache", target_os = "linux"))]
fn cache(report: &mut Report) {
    use zerogravity::cache::{CacheMetadata, paths};
    let result = (|| -> Result<_> {
        let root = paths::resolve_root(
            None,
            std::env::var_os("XDG_CACHE_HOME").as_deref(),
            std::env::var_os("HOME").as_deref(),
        )?;
        let metadata = match CacheMetadata::open(&root) {
            Ok(metadata) => metadata,
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let snapshots = metadata.snapshots()?;
        let bytes = snapshots
            .iter()
            .try_fold(0u64, |sum, snapshot| {
                sum.checked_add(snapshot.manifest.total_bytes())
            })
            .ok_or_else(|| Error::Limit("logical cache bytes".into()))?;
        Ok(Some(json!({
            "snapshots": snapshots.len(), "logical_snapshot_bytes": bytes,
            "configured_disk_limit_bytes": metadata.disk_limit_bytes()?,
            "physical_charge_observed": false, "payload_verified": false,
        })))
    })();
    match result {
        Ok(None) => report.add(
            "cache",
            "ok",
            "Cache not created; no changes made",
            Value::Null,
        ),
        Ok(Some(detail)) => {
            let summary = format!(
                "{} snapshots · {} logical bytes · {} byte configured limit",
                detail["snapshots"],
                detail["logical_snapshot_bytes"],
                detail["configured_disk_limit_bytes"]
            );
            report.add(
                "cache",
                "ok",
                "Readable cache metadata; logical bytes may share blobs, not physical quota usage",
                detail,
            );
            report.lines([summary]);
        }
        Err(_) => report.error(
            "cache",
            "Cache root or metadata unavailable; use model list for detail",
        ),
    }
}
#[cfg(not(all(feature = "cache", target_os = "linux")))]
fn cache(report: &mut Report) {
    report.add(
        "cache",
        "unavailable",
        "Cache feature unavailable on this build",
        Value::Null,
    );
}

pub(super) fn run(document: Option<(&'static str, String)>, format: Format) -> Result<()> {
    let mut report = Report::default();
    let request = config(
        &mut report,
        document
            .as_ref()
            .map(|(flag, value)| (*flag, value.as_str())),
    );
    cache(&mut report);
    hardware(&mut report, request.as_ref());
    if let Some(request) = &request {
        source_paths(&mut report, request);
    }
    report.finish(format)
}
