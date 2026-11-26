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
    pub warning: Option<String>,
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
        warning: observed.warning,
    };
    for row in observed.rows {
        let raw = serde_yaml::to_string(&row.value)
            .map_err(|_| Error::State("setting encoding".into()))?;
        if row.explicit {
            result.values.insert(row.key.clone(), raw.clone());
        }
        let shown = preview(raw.trim_end());
        result.rows.push(Row {
            key: Key::Setting(row.key.clone()),
            title: format!("{} = {shown}", row.key),
            details: vec![
                row.key,
                format!("Value: {shown}"),
                format!("Origin: {}", row.origin),
                if row.explicit {
                    "Explicit value".into()
                } else {
                    "Inherited / built-in".into()
                },
                row.help.into(),
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
                title: if name == "default" {
                    "default · config.yaml".into()
                } else {
                    format!("{name} · profiles/{name}.yaml")
                },
                details: vec![
                    target.path.display().to_string(),
                    if target.path.is_file() {
                        "Stored document".into()
                    } else {
                        "Built-in defaults (no document created)".into()
                    },
                    "Enter opens editor; activation is explicit.".into(),
                ],
            })
        })
        .collect()
}
