//! Native log filters; operational history belongs to the foreground host.
use crate::control_host::{
    Actions,
    logs::{Component, Level},
};
#[derive(Default)]
pub(super) struct Logs {
    actions: Actions,
    level: Option<Level>,
    component: Option<Component>,
    pub query: String,
    paused_at: Option<u64>,
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
        self.paused_at = if self.following() {
            Some(self.actions.with_logs(|ring| ring.revision()))
        } else {
            None
        };
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
            "{} · level {} · component {} · search {}",
            if self.following() {
                "Following"
            } else {
                "Paused view (collection continues)"
            },
            self.level.map_or("all", Level::label),
            self.component.map_or("all", Component::label),
            if self.query.is_empty() {
                "all"
            } else {
                &self.query
            }
        )
    }
    pub fn lines(&self) -> Vec<String> {
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
                })
                .map(|entry| entry.line())
                .filter(|line| query.is_empty() || line.to_ascii_lowercase().contains(&query))
                .collect()
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
        assert_eq!(logs.lines().len(), 1);
        logs.toggle_follow();
        logs.query = "registration 17".into();
        assert_eq!(logs.lines().len(), 1);
        logs.query.clear();
        for _ in 0..CAPACITY {
            logs.actions.with_logs_mut(|ring| ring.gap(1));
        }
        assert_eq!(logs.lines().len(), CAPACITY);
        assert_eq!(logs.dropped(), 2);
        logs.cycle_level();
        assert!(logs.lines().is_empty());
    }
}
