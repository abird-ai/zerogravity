//! Load-time smart configuration: advisory planning, never execution authority.
//!
//! Models contribute typed overrides and qualified candidate evaluation. This
//! layer only defines the common mode/report contract so new knobs can be added
//! without growing one universal model-specific configuration enum.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SmartConfigMode {
    /// Smart configuration owns supported knobs; user values for those knobs
    /// are intentionally ignored.
    Exclusive,
    /// Smart configuration runs for unspecified knobs; user values win.
    #[default]
    Enabled,
    /// Smart configuration is off; use user values or static model defaults.
    Disabled,
}

impl SmartConfigMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Exclusive => "exclusive",
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmartConfig<O> {
    pub mode: SmartConfigMode,
    pub overrides: O,
}

impl<O: Default> Default for SmartConfig<O> {
    fn default() -> Self {
        Self {
            mode: SmartConfigMode::Enabled,
            overrides: O::default(),
        }
    }
}

impl<O> SmartConfig<O> {
    pub const fn new(mode: SmartConfigMode, overrides: O) -> Self {
        Self { mode, overrides }
    }

    pub fn exclusive(overrides: O) -> Self {
        Self::new(SmartConfigMode::Exclusive, overrides)
    }

    pub fn enabled(overrides: O) -> Self {
        Self::new(SmartConfigMode::Enabled, overrides)
    }

    pub fn disabled(overrides: O) -> Self {
        Self::new(SmartConfigMode::Disabled, overrides)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmartConfigSource {
    Auto,
    Default,
    User,
}

impl SmartConfigSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Default => "default",
            Self::User => "user",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmartConfigDecision<T> {
    pub value: T,
    pub source: SmartConfigSource,
    pub reason: String,
}

/// Apply the same precedence contract to any independently tunable model knob.
/// The auto closure is never evaluated when a user value/static default decides
/// the knob, so disabled/overridden knobs add zero startup probing work.
pub fn resolve<T>(
    mode: SmartConfigMode,
    user: Option<T>,
    default: T,
    auto: impl FnOnce() -> crate::Result<(T, String)>,
) -> crate::Result<SmartConfigDecision<T>> {
    match mode {
        SmartConfigMode::Exclusive => {
            let ignored = user.is_some();
            let (value, reason) = auto()?;
            Ok(SmartConfigDecision {
                value,
                source: SmartConfigSource::Auto,
                reason: if ignored {
                    format!("{reason}; user value ignored by exclusive mode")
                } else {
                    reason
                },
            })
        }
        SmartConfigMode::Enabled => match user {
            Some(value) => Ok(SmartConfigDecision {
                value,
                source: SmartConfigSource::User,
                reason: "user value overrides smart selection".into(),
            }),
            None => {
                let (value, reason) = auto()?;
                Ok(SmartConfigDecision {
                    value,
                    source: SmartConfigSource::Auto,
                    reason,
                })
            }
        },
        SmartConfigMode::Disabled => Ok(match user {
            Some(value) => SmartConfigDecision {
                value,
                source: SmartConfigSource::User,
                reason: "smart configuration disabled; using user value".into(),
            },
            None => SmartConfigDecision {
                value: default,
                source: SmartConfigSource::Default,
                reason: "smart configuration disabled; using static model default".into(),
            },
        }),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmartConfigChoice {
    pub key: &'static str,
    pub value: String,
    pub source: SmartConfigSource,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmartConfigReport {
    pub model: String,
    pub backend: &'static str,
    pub mode: SmartConfigMode,
    pub observed_unix_ms: Option<u128>,
    pub choices: Vec<SmartConfigChoice>,
}

impl SmartConfigReport {
    pub fn new(
        model: impl Into<String>,
        backend: &'static str,
        mode: SmartConfigMode,
        observed_unix_ms: Option<u128>,
    ) -> Self {
        Self {
            model: model.into(),
            backend,
            mode,
            observed_unix_ms,
            choices: Vec::new(),
        }
    }

    pub fn push(
        &mut self,
        key: &'static str,
        value: impl Into<String>,
        source: SmartConfigSource,
        reason: impl Into<String>,
    ) {
        self.choices.push(SmartConfigChoice {
            key,
            value: value.into(),
            source,
            reason: reason.into(),
        });
    }

    /// One compact startup record. The normal model/resource admission remains
    /// authoritative if the observed environment changes after this decision.
    pub fn emit(&self) {
        let choices = self
            .choices
            .iter()
            .map(|choice| {
                format!(
                    "{}={}({}: {})",
                    choice.key,
                    choice.value,
                    choice.source.as_str(),
                    choice.reason
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        eprintln!(
            "smart-config model={} backend={} mode={} observed_unix_ms={} {}",
            self.model,
            self.backend,
            self.mode.as_str(),
            self.observed_unix_ms
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".into()),
            choices
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct Overrides {
        state: Option<u8>,
        batch: Option<u8>,
    }

    #[test]
    fn generic_mode_keeps_independent_extensible_overrides() {
        let config = SmartConfig::enabled(Overrides {
            state: Some(2),
            batch: None,
        });
        assert_eq!(config.mode, SmartConfigMode::Enabled);
        assert_eq!(config.overrides.state, Some(2));
        assert_eq!(config.overrides.batch, None);

        let enabled = resolve(SmartConfigMode::Enabled, Some(7u8), 1, || {
            panic!("enabled user override must skip auto")
        })
        .unwrap();
        assert_eq!(enabled.value, 7);
        assert_eq!(enabled.source, SmartConfigSource::User);

        let disabled = resolve(SmartConfigMode::Disabled, None, 3u8, || {
            panic!("disabled mode must skip auto")
        })
        .unwrap();
        assert_eq!(disabled.value, 3);
        assert_eq!(disabled.source, SmartConfigSource::Default);

        let exclusive = resolve(SmartConfigMode::Exclusive, Some(9u8), 1, || {
            Ok((5u8, "runtime fit".into()))
        })
        .unwrap();
        assert_eq!(exclusive.value, 5);
        assert_eq!(exclusive.source, SmartConfigSource::Auto);
        assert!(exclusive.reason.contains("ignored"));
    }
}
