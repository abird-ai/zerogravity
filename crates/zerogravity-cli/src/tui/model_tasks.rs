//! Model detail projection over the foreground slot and original runtime owners.
#[cfg(all(feature = "cache", target_os = "linux"))]
use super::format;
use super::{
    App,
    library::Key,
    theme::Theme,
    view::{label, text},
};
use crate::control_host::progress::{Subject, Task};
use ratatui::text::Line;

fn digest(key: &Key) -> Option<&str> {
    match key {
        Key::Snapshot(digest) => Some(digest),
        #[cfg(feature = "model-qwen35")]
        Key::Target(target) => zerogravity::models::qwen_source_snapshot(target),
        _ => None,
    }
}

#[cfg(all(feature = "acquisition", target_os = "linux"))]
fn catalog_repository(key: &Key) -> Option<&'static str> {
    let Key::Target(target) = key else {
        return None;
    };
    zerogravity::targets::MODEL_TARGETS
        .iter()
        .find(|entry| entry.slug == target)?
        .repository
}

fn registration_matches(app: &App, key: &Key, registration: u64) -> bool {
    digest(key).is_some_and(|digest| {
        app.snapshot.runtime.catalog.iter().any(|model| {
            model.registration_id == registration
                && model
                    .capabilities
                    .as_ref()
                    .and_then(|cap| cap.details.artifact.as_ref())
                    .and_then(|artifact| artifact.cache_snapshot.as_deref())
                    == Some(digest)
        })
    })
}

/// A repository is not an exact cached payload selection. Related pulls
/// are explicitly distinguished from actions on the selected snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Relation {
    Exact,
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    Related,
}

#[cfg(all(feature = "acquisition", target_os = "linux"))]
fn source_relation(
    app: &App,
    key: &Key,
    provider: &str,
    repository: &str,
    revision: &str,
) -> Option<Relation> {
    if let Key::Snapshot(digest) = key {
        let source = app.library.source(digest)?;
        return (source.provider == provider
            && source.repository == repository
            && source.revision == revision)
            .then_some(Relation::Related);
    }
    (provider == "huggingface" && catalog_repository(key) == Some(repository))
        .then_some(Relation::Related)
}

fn relation(app: &App, key: &Key, task: &Task) -> Option<Relation> {
    let exact = match task.subject.as_ref()? {
        Subject::Snapshot(source) => digest(key) == Some(source),
        Subject::Catalog(target) => {
            matches!(key, Key::Target(selected) if target == selected) || {
                #[cfg(feature = "model-qwen35")]
                {
                    zerogravity::models::qwen_source_snapshot(target)
                        .is_some_and(|source| digest(key) == Some(source))
                }
                #[cfg(not(feature = "model-qwen35"))]
                {
                    false
                }
            }
        }
        Subject::Registration(id) => registration_matches(app, key, *id),
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        Subject::Repository {
            provider,
            repository,
            revision,
        } => return source_relation(app, key, provider, repository, revision),
    };
    exact.then_some(Relation::Exact)
}

fn execution(rows: &mut Vec<Line<'static>>, app: &App, key: &Key, theme: Theme) {
    let Some(digest) = digest(key) else { return };
    for model in &app.snapshot.runtime.catalog {
        let Some(cap) = &model.capabilities else {
            continue;
        };
        if cap
            .details
            .artifact
            .as_ref()
            .and_then(|artifact| artifact.cache_snapshot.as_deref())
            != Some(digest)
        {
            continue;
        }
        let provider = cap
            .details
            .execution
            .as_ref()
            .map(|e| format!("{:?}", e.selected.provider));
        let device = match &cap.details.memory {
            Some(zerogravity::models::ModelMemory::Resident(memory)) => {
                Some(memory.device.device.as_str())
            }
            Some(zerogravity::models::ModelMemory::ResidentText { admission, .. }) => {
                Some(admission.device.device.as_str())
            }
            Some(zerogravity::models::ModelMemory::Host { .. }) => Some("host"),
            None => None,
        };
        rows.push(Line::default());
        rows.push(label(
            format!("Registration #{}", model.registration_id),
            theme,
        ));
        rows.push(text(format!(
            "{:?} · {} admitted requests",
            model.lifecycle, model.admitted_requests
        )));
        if let Some(provider) = provider {
            rows.push(text(format!(
                "Actual execution · {provider}{}",
                device.map_or_else(String::new, |d| format!(" · {d}"))
            )));
        }
    }
}

pub(super) fn append(rows: &mut Vec<Line<'static>>, app: &App, key: &Key, theme: Theme) {
    let (active, finished) = app.actions.tasks();
    let active = active
        .as_ref()
        .and_then(|task| relation(app, key, task).map(|relation| (task, relation)));
    if let Some((task, relation)) = active {
        rows.push(Line::default());
        rows.push(label(
            if relation == Relation::Exact {
                "Active task"
            } else {
                "Related source task"
            },
            theme,
        ));
        rows.push(
            text(format!(
                "{} · #{} · {}s",
                task.action.title,
                task.action.id,
                task.started.elapsed().as_secs()
            ))
            .style(theme.info()),
        );
        rows.push(text(task.progress.stage.label()));
        if let Some(requested) = &task.requested {
            rows.push(text(format!("Requested · {requested}")));
        }
        if let Some(resolved) = &task.progress.resolved {
            rows.push(text(format!("Selected backend · {resolved}")));
        }
        #[cfg(all(feature = "cache", target_os = "linux"))]
        if let Some(progress) = task.progress.verification {
            rows.push(text(format!(
                "Files verified · {} / {}",
                progress.files_verified, progress.total_files
            )));
            rows.push(text(format!(
                "Hashed · {}",
                format::progress(progress.bytes_read, Some(progress.total_bytes))
            )));
        }
        if task.action.cancel_requested {
            rows.push(text("Cancellation requested · awaiting owner").style(theme.warning()));
        }
    }
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    if let Some(download) = &app.snapshot.download
        && !download.status.terminal()
    {
        let relation = if download
            .snapshot
            .as_deref()
            .is_some_and(|source| digest(key) == Some(source))
        {
            Some(Relation::Exact)
        } else {
            download.source.as_ref().and_then(|source| {
                source_relation(
                    app,
                    key,
                    source.provider(),
                    source.repository(),
                    source.revision(),
                )
            })
        };
        if let Some(relation) = relation {
            rows.push(Line::default());
            rows.push(label(
                if relation == Relation::Exact {
                    "Download"
                } else {
                    "Related source download"
                },
                theme,
            ));
            rows.push(
                text(format!(
                    "{:?} · {:?} · #{}",
                    download.phase, download.status, download.id
                ))
                .style(theme.info()),
            );
            append_download(rows, download, true);
        }
    }
    if let Some(finished) = finished.as_ref()
        && let Some(relation) = relation(app, key, &finished.task)
        && active.is_none()
    {
        rows.push(Line::default());
        rows.push(label(
            if relation == Relation::Exact {
                "Last observed action"
            } else {
                "Last related source action"
            },
            theme,
        ));
        rows.push(
            text(format!(
                "{} · #{} · {}s",
                finished.task.action.title,
                finished.task.action.id,
                finished.elapsed.as_secs()
            ))
            .style(if finished.failed {
                theme.warning()
            } else {
                theme.muted()
            }),
        );
        if finished.failed {
            rows.push(text(format!(
                "Failed during {}",
                finished.task.progress.stage.label()
            )));
        }
        if let Some(requested) = &finished.task.requested {
            rows.push(text(format!("Requested · {requested}")));
        }
        if let Some(resolved) = &finished.task.progress.resolved {
            rows.push(text(format!("Selected backend · {resolved}")));
        }
        #[cfg(all(feature = "cache", target_os = "linux"))]
        if !finished.failed
            && finished.task.progress.stage == crate::control_host::progress::Stage::Verifying
        {
            rows.push(text("Verification finished · observation only").style(theme.success()));
        }
        // The actual bounded result, not an inferred success from a cancel request.
        rows.extend(finished.message.lines().map(text));
    }
    execution(rows, app, key, theme);
}

#[cfg(all(feature = "acquisition", target_os = "linux"))]
pub(super) fn append_download(
    rows: &mut Vec<Line<'static>>,
    download: &zerogravity::control::DownloadSummary,
    compact: bool,
) {
    use zerogravity::control::DownloadStatus;
    if !compact && let Some(source) = &download.source {
        rows.push(text(format!(
            "{}:{}@{}",
            source.provider(),
            source.repository(),
            source.revision()
        )));
    }
    rows.push(text(format!(
        "Files: {} / {}",
        download.completed_files, download.selected_files
    )));
    let total = format::progress(download.total_bytes, download.total_expected_bytes);
    rows.push(text(format!(
        "Total: {total}{}",
        if download.total_expected_bytes.is_none() {
            format!(" (max {})", format::bytes(download.total_maximum_bytes))
        } else {
            String::new()
        }
    )));
    if let Some(path) = &download.path {
        rows.push(text(format!("File: {path}")));
        rows.push(text(format::progress(
            download.file_bytes,
            download.file_expected_bytes,
        )));
    }
    if !compact {
        rows.push(text(format!(
            "Cached: {} · resumed: {}",
            format::bytes(download.cached_bytes),
            format::bytes(download.resumed_bytes)
        )));
        rows.push(text(format!(
            "Last observed checkpoint: {}",
            format::bytes(download.checkpoint_bytes)
        )));
    }
    if download.status == DownloadStatus::Running
        && !download.cancel_requested
        && let Some(rate) = download.logical_bytes_per_second.filter(|rate| *rate > 0)
    {
        rows.push(text(format!("Accepted: {}/s", format::bytes(rate))));
        if !compact && let Some(total) = download.total_expected_bytes {
            rows.push(text(format!(
                "Approximate transfer ETA: {}s (not publication)",
                total.saturating_sub(download.total_bytes).div_ceil(rate)
            )));
        }
    }
    if download.cancel_requested && !download.status.terminal() {
        rows.push(text(
            "Cancellation requested; awaiting original executor outcome.",
        ));
    }
    if !compact && let Some(snapshot) = &download.snapshot {
        rows.push(text(format!("Snapshot: {snapshot}")));
    }
    if download.status == DownloadStatus::DurabilityUnknown {
        rows.push(text(
            "Publication durability unknown; inspect before retrying.",
        ));
    }
}
