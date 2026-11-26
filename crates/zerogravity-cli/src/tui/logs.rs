//! Native log filters; operational history belongs to the foreground host.
use crate::control_host::{
    Actions,
    logs::{Component, Entry, Level},
};
#[derive(Default)]
pub(super) struct Logs {
    actions: Actions,
    level: Option<Level>,
    component: Option<Component>,
    pub query: String,
    paused_at: Option<u64>,
    pub selected: Option<u64>,
}
impl Logs {
    pub fn new(actions: Actions) -> Self {
        Self {
            actions,
            ..Default::default()
        }
    }
    pub fn following(&self) -> bool {
        self.paused_at.is_none()
    }
    pub fn toggle_follow(&mut self) {
        self.selected = None;
        self.paused_at = if self.following() {
            Some(self.actions.with_logs(|ring| ring.revision()))
        } else {
            None
        };
    }
    pub fn level(&self) -> Option<Level> {
        self.level
    }
    pub fn component(&self) -> Option<Component> {
        self.component
    }
    pub fn set_level(&mut self, level: Option<Level>) {
        self.level = level;
    }
    pub fn set_component(&mut self, component: Option<Component>) {
        self.component = component;
    }
    pub fn cycle_level(&mut self) {
        self.level = match self.level {
            None => Some(Level::Info),
            Some(Level::Info) => Some(Level::Warning),
            Some(Level::Warning) => Some(Level::Error),
            Some(Level::Error) => None,
        };
    }
    pub fn cycle_component(&mut self) {
        self.component = match self.component {
            None => Some(Component::ALL[0]),
            Some(value) => Component::ALL
                .iter()
                .position(|component| *component == value)
                .and_then(|i| Component::ALL.get(i + 1))
                .copied(),
        };
    }
    pub fn heading(&self) -> String {
        format!(
            "{} · level {} · component {}",
            if self.following() {
                "Following"
            } else {
                "Paused view (collection continues)"
            },
            self.level.map_or("all", Level::label),
            self.component.map_or("all", Component::label),
        )
    }
    pub fn entries(&self) -> Vec<Entry> {
        let query = self.query.to_ascii_lowercase();
        self.actions.with_logs(|ring| {
            ring.entries()
                .iter()
                .filter(|entry| {
                    self.paused_at.is_none_or(|end| entry.sequence < end)
                        && self.level.is_none_or(|level| entry.level == level)
                        && self
                            .component
                            .is_none_or(|component| entry.component == component)
                        && (query.is_empty() || entry.line().to_ascii_lowercase().contains(&query))
                })
                .cloned()
                .collect()
        })
    }
    pub fn select(&mut self, sequence: u64) -> bool {
        if !self.entries().iter().any(|row| row.sequence == sequence) {
            return false;
        }
        if self.following() {
            self.toggle_follow();
        }
        self.selected = Some(sequence);
        true
    }
    pub fn move_selection(&mut self, delta: i8) -> bool {
        let entries = self.entries();
        let Some(first) = entries.first() else {
            self.selected = None;
            return false;
        };
        let current = self
            .selected
            .and_then(|id| entries.iter().position(|row| row.sequence == id));
        let index = current.map_or_else(
            || if delta < 0 { entries.len() - 1 } else { 0 },
            |i| {
                i.saturating_add_signed(delta as isize)
                    .min(entries.len() - 1)
            },
        );
        self.select(entries.get(index).unwrap_or(first).sequence)
    }
    pub fn detail(&self, sequence: u64) -> Option<Vec<String>> {
        self.entries()
            .into_iter()
            .find(|entry| entry.sequence == sequence)
            .map(|entry| {
                vec![
                    entry.message,
                    String::new(),
                    format!("Level: {}", entry.level.label()),
                    format!("Component: {}", entry.component.label()),
                    format!("Correlation: {}", entry.correlation),
                    format!("Sequence: {}", entry.sequence),
                ]
            })
    }
    pub fn dropped(&self) -> u64 {
        self.actions.with_logs(|ring| ring.dropped())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_host::logs::CAPACITY;
    use zerogravity::control::{ControlEvent, ControlEventKind};
    #[test]
    fn logs_follow_filter_and_retain_only_bounded_operational_rows() {
        let mut logs = Logs::default();
        logs.actions.with_logs_mut(|ring| {
            ring.event(ControlEvent {
                sequence: 2,
                kind: ControlEventKind::RequestAccepted {
                    registration_id: 17,
                },
            })
        });
        logs.toggle_follow();
        logs.actions
            .with_logs_mut(|ring| ring.finish(1, "Run", false));
        assert_eq!(logs.entries().len(), 1);
        logs.toggle_follow();
        logs.query = "registration 17".into();
        assert_eq!(logs.entries().len(), 1);
        let id = logs.entries()[0].sequence;
        assert!(
            logs.detail(id)
                .unwrap()
                .iter()
                .any(|line| line.contains("registration 17"))
        );
        assert!(logs.select(id));
        logs.toggle_follow();
        logs.query.clear();
        for _ in 0..CAPACITY {
            logs.actions.with_logs_mut(|ring| ring.gap(1));
        }
        assert_eq!(logs.entries().len(), CAPACITY);
        assert_eq!(logs.dropped(), 2);
        assert!(logs.detail(id).is_none());
        assert!(!logs.select(id));
        logs.cycle_level();
        assert!(logs.entries().is_empty());
    }
}
