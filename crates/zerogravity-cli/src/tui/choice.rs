//! Finite typed setting and operational-filter choices. Applying a setting
//! still submits the original exact ConfigFile draft.
use super::{
    settings::Snapshot,
    view::{Action, Form, FormKind, LogAction},
};
use crate::control_host::{
    execution::{Choice, Execution},
    logs::{Component, Level},
};

const LEVELS: [Option<Level>; 4] = [
    None,
    Some(Level::Info),
    Some(Level::Warning),
    Some(Level::Error),
];
#[derive(Debug)]
enum Kind {
    Setting {
        file: Snapshot,
        key: String,
        values: &'static [&'static str],
    },
    Level,
    Component,
    Execution {
        digest: String,
        revision: u64,
        choices: Vec<Choice>,
    },
}
#[derive(Debug)]
pub(super) struct Picker {
    kind: Kind,
    pub cursor: usize,
    pub scroll: u16,
}
impl Picker {
    pub fn setting(
        file: Snapshot,
        key: String,
        values: &'static [&'static str],
        current: usize,
    ) -> Self {
        Self {
            kind: Kind::Setting { file, key, values },
            cursor: current,
            scroll: 0,
        }
    }
    pub fn execution(
        digest: String,
        revision: u64,
        choices: Vec<Choice>,
        current: Execution,
        profile: &str,
    ) -> Self {
        let mut choices = crate::control_host::execution::for_profile(choices, profile);
        let cursor = choices
            .iter()
            .position(|choice| choice.execution == current)
            .or_else(|| {
                (profile == "Auto"
                    && current.backend == Some(zerogravity::run_policy::Backend::Auto)
                    && current.device.is_none())
                .then_some(0)
            })
            .unwrap_or_else(|| {
                choices.push(Choice {
                    label: current.summary(profile),
                    execution: current,
                    refusal: Some("Not currently observed"),
                });
                choices.len() - 1
            });
        Self {
            kind: Kind::Execution {
                digest,
                revision,
                choices,
            },
            cursor,
            scroll: 0,
        }
    }
    pub fn level(level: Option<Level>) -> Self {
        Self {
            kind: Kind::Level,
            cursor: LEVELS.iter().position(|v| *v == level).unwrap_or(0),
            scroll: 0,
        }
    }
    pub fn component(component: Option<Component>) -> Self {
        Self {
            kind: Kind::Component,
            cursor: component
                .and_then(|v| Component::ALL.iter().position(|c| *c == v))
                .map_or(0, |i| i + 1),
            scroll: 0,
        }
    }
    pub fn count(&self) -> usize {
        match &self.kind {
            Kind::Setting { values, .. } => values.len(),
            Kind::Level => LEVELS.len(),
            Kind::Component => Component::ALL.len() + 1,
            Kind::Execution { choices, .. } => choices.len(),
        }
    }
    pub fn title(&self) -> String {
        match &self.kind {
            Kind::Setting { key, file, .. } => format!("{key} · {}", file.target.label()),
            Kind::Level => "Log level".into(),
            Kind::Component => "Log component".into(),
            Kind::Execution { .. } => "Execution · model load checks fit".into(),
        }
    }
    pub fn is_setting(&self) -> bool {
        matches!(self.kind, Kind::Setting { .. })
    }
    pub fn label(&self, index: usize) -> &str {
        match &self.kind {
            Kind::Setting { values, .. } => values[index],
            Kind::Level => LEVELS[index].map_or("All levels", Level::label),
            Kind::Execution { choices, .. } => &choices[index].label,
            Kind::Component => {
                if index == 0 {
                    "All components"
                } else {
                    Component::ALL[index - 1].label()
                }
            }
        }
    }
    pub fn refusal(&self, index: usize) -> Option<&'static str> {
        match &self.kind {
            Kind::Execution { choices, .. } => choices[index].refusal,
            _ => None,
        }
    }
    pub fn apply(self) -> Action {
        if self.refusal(self.cursor).is_some() {
            return Action::Redraw;
        }
        match self.kind {
            Kind::Setting {
                file, key, values, ..
            } => Action::Submit(Form {
                kind: FormKind::ConfigValue { file, key },
                value: values[self.cursor].into(),
            }),
            Kind::Level => Action::Logs(LogAction::SetLevel(LEVELS[self.cursor])),
            Kind::Execution {
                digest,
                revision,
                mut choices,
            } => Action::SetExecution {
                digest,
                revision,
                execution: choices.swap_remove(self.cursor).execution,
            },
            Kind::Component => Action::Logs(LogAction::SetComponent(
                self.cursor.checked_sub(1).map(|i| Component::ALL[i]),
            )),
        }
    }
    pub fn into_yaml(self) -> Option<Form> {
        if let Kind::Setting { file, key, values } = self.kind {
            Some(Form {
                kind: FormKind::ConfigValue { file, key },
                value: values[self.cursor].into(),
            })
        } else {
            None
        }
    }
    pub fn move_by(&mut self, delta: isize) {
        self.cursor = self
            .cursor
            .saturating_add_signed(delta)
            .min(self.count() - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity::run_policy::Backend;

    #[test]
    fn unavailable_execution_choice_cannot_apply() {
        let cpu = Execution {
            backend: Some(Backend::Cpu),
            device: None,
        };
        let mut choices = crate::control_host::execution::choices(&[]);
        let unavailable = choices.iter_mut().find(|row| row.execution == cpu).unwrap();
        unavailable.refusal = Some("GGUF required");
        let picker = Picker::execution("fixture".into(), 1, choices, cpu, "Auto");
        assert_eq!(picker.refusal(picker.cursor), Some("GGUF required"));
        assert_eq!(picker.apply(), Action::Redraw);
    }
}
