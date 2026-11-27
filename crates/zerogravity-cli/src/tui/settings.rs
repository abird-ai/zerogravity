//! Native presentation of the shared cold configuration owner.
use super::library::{Key, Row};
pub(super) use crate::control_host::Target;
use crate::control_host::settings as shared;
pub(crate) use shared::{Action, Snapshot, group};
use std::collections::BTreeMap;
use zerogravity::{Error, Result};

pub(super) struct Observed {
    pub snapshot: Snapshot,
    pub rows: Vec<Row>,
    pub values: BTreeMap<String, String>,
    pub effective_choices: BTreeMap<String, String>,
    pub warning: Option<String>,
}

/// Presentation choices for the finite scalar settings only. The existing
/// document validator still owns every accepted value and dependency rule.
pub(super) fn choices(key: &str) -> Option<&'static [&'static str]> {
    Some(match key {
        "ui.color" | "ui.mouse" | "ui.progress" | "ui.interactive" => &["auto", "always", "never"],
        "run.backend" => &["auto", "cpu", "cuda", "rocm", "vulkan"],
        "run.numerical" => &["source", "native"],
        "run.speculation" => &["auto", "off", "mtp"],
        "serve.cohort_width" => &["auto", "0", "4", "8"],
        "run.task.raw"
        | "run.diagnostics.explain"
        | "run.diagnostics.dry_run"
        | "serve.enable_error_details"
        | "serve.enable_per_request_metrics"
        | "downloads.ask_before_pull" => &["false", "true"],
        _ => return None,
    })
}
fn preview(value: &str) -> String {
    let flat = value.replace('\n', " ");
    let mut result: String = flat.chars().take(100).collect();
    if flat.chars().count() > 100 {
        result.push('…');
    }
    result
}

pub(super) fn observe(target: Target) -> Result<Observed> {
    let observed = shared::observe(target)?;
    let mut result = Observed {
        snapshot: observed.snapshot,
        rows: Vec::new(),
        values: BTreeMap::new(),
        effective_choices: BTreeMap::new(),
        warning: observed.warning,
    };
    for row in observed.rows {
        let raw = serde_yaml::to_string(&row.value)
            .map_err(|_| Error::State("setting encoding".into()))?;
        if row.explicit {
            result.values.insert(row.key.clone(), raw.clone());
        }
        if choices(&row.key).is_some() {
            result
                .effective_choices
                .insert(row.key.clone(), raw.trim().to_owned());
        }
        let shown = preview(raw.trim_end());
        result.rows.push(Row {
            key: Key::Setting(row.key.clone()),
            title: format!("{} = {shown}", row.key),
            annotation: None,
            details: vec![
                if row.explicit {
                    "Explicit value".into()
                } else {
                    "Inherited / built-in".into()
                },
                row.help.into(),
            ],
            technical: vec![
                format!("Origin: {}", row.origin),
                format!("YAML value: {}", raw.trim_end()),
            ],
        });
    }
    Ok(result)
}
pub(super) fn profiles() -> Result<Vec<Row>> {
    let names = shared::profiles()?;
    names
        .into_iter()
        .map(|name| {
            let target = Target::profile(&name)?;
            Ok(Row {
                key: Key::Profile(name.clone()),
                title: name.clone(),
                annotation: None,
                details: vec![if target.path.is_file() {
                    "Stored document".into()
                } else {
                    "Built-in defaults (no document created)".into()
                }],
                technical: vec![target.path.display().to_string()],
            })
        })
        .collect()
}
