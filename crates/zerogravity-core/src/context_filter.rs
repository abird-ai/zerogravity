//! Reusable, bounded context-source selection and ordered filtering.
//!
//! This module neither reads a conversation nor grants access to one. Its
//! records are supplied by a caller after the application/session source
//! owner has checked real scope permissions. A config can *request* app-owned
//! system/developer prompts or another consented user's shared records, but
//! does not thereby make those records available. A decision adapter cannot
//! restore anything removed by source authorization or earlier filters.
//! Decision-model execution is not implemented by this crate.
use crate::{Error, Result};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum RecordKind {
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    SystemPrompt,
    DeveloperPrompt,
}
impl RecordKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::UserMessage => "user_message",
            Self::AssistantMessage => "assistant_message",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::SystemPrompt => "system_prompt",
            Self::DeveloperPrompt => "developer_prompt",
        }
    }
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "user_message" => Ok(Self::UserMessage),
            "assistant_message" => Ok(Self::AssistantMessage),
            "tool_call" => Ok(Self::ToolCall),
            "tool_result" => Ok(Self::ToolResult),
            "system_prompt" => Ok(Self::SystemPrompt),
            "developer_prompt" => Ok(Self::DeveloperPrompt),
            _ => Err(Error::Unsupported(
                "unknown context source record kind".into(),
            )),
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ContextSource {
    /// session, application or explicitly named shared resource:
    /// shared:<grant-id>. Shared grants can cover other users only when the
    /// owning source service authorizes those records for this caller.
    pub origin: String,
    pub kind: RecordKind,
}
impl ContextSource {
    pub fn parse(input: &str) -> Result<Self> {
        let (origin, kind) = input
            .rsplit_once('/')
            .ok_or_else(|| Error::Malformed("context sources use ORIGIN/KIND".into()))?;
        let valid = origin == "session"
            || origin == "application"
            || origin.strip_prefix("shared:").is_some_and(|id| {
                !id.is_empty()
                    && id.len() <= 128
                    && !id.starts_with('.')
                    && !id.contains("..")
                    && id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
            });
        if !valid {
            return Err(Error::Malformed(
                "invalid context origin or shared-grant reference".into(),
            ));
        }
        Ok(Self {
            origin: origin.into(),
            kind: RecordKind::parse(kind)?,
        })
    }
    pub fn label(&self) -> String {
        format!("{}/{}", self.origin, self.kind.label())
    }
}
pub fn default_sources() -> Vec<ContextSource> {
    ["session/user_message", "session/assistant_message"]
        .iter()
        .map(|s| ContextSource::parse(s).expect("static source labels"))
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FilterAction {
    Include,
    Exclude,
}
impl FilterAction {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "include" => Ok(Self::Include),
            "exclude" => Ok(Self::Exclude),
            _ => Err(Error::Malformed(
                "context filter action must be include or exclude".into(),
            )),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchField {
    Kind,
    ToolName,
    Origin,
    TextContains,
}
impl MatchField {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "kind" => Ok(Self::Kind),
            "tool_name" => Ok(Self::ToolName),
            "origin" => Ok(Self::Origin),
            "text_contains" => Ok(Self::TextContains),
            _ => Err(Error::Unsupported(
                "unregistered context filter field".into(),
            )),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionUnavailable {
    /// Safe default: unverified/no verdict contributes no record.
    Omit,
    /// Refuse the entire assembly if a required decision is unavailable.
    Reject,
    /// Retain a *previously authorized and selected* record. Never bypass
    /// source grant checks or earlier exclusions, even with this explicit choice.
    KeepAuthorized,
}
impl DecisionUnavailable {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "omit" => Ok(Self::Omit),
            "reject" => Ok(Self::Reject),
            "keep_authorized" => Ok(Self::KeepAuthorized),
            _ => Err(Error::Malformed(
                "invalid decision unavailability policy".into(),
            )),
        }
    }
}
#[derive(Clone, Debug)]
pub enum ContextFilter {
    Match {
        action: FilterAction,
        field: MatchField,
        values: Vec<String>,
    },
    Decision {
        /// One declared model instance; could be NanoJev or another future
        /// source-verified model with a typed decision head.
        model: String,
        question: String,
        min_confidence_bps: u16,
        unavailable: DecisionUnavailable,
    },
}
#[derive(Clone, Debug)]
pub struct ContextFilterPlan {
    /// What the application REQUESTS, not what it is authorized to read.
    /// Defaults to visible same-session user/assistant messages only.
    pub sources: Vec<ContextSource>,
    /// Ordered transformation of authenticated, already-granted records.
    pub stages: Vec<ContextFilter>,
}
impl Default for ContextFilterPlan {
    fn default() -> Self {
        Self {
            sources: default_sources(),
            stages: vec![],
        }
    }
}
fn clean(value: &str, max: usize) -> bool {
    !value.is_empty() && value.len() <= max && !value.chars().any(char::is_control)
}
impl ContextFilterPlan {
    pub fn validate(&self) -> Result<()> {
        if self.sources.is_empty() || self.sources.len() > 64 || self.stages.len() > 32 {
            return Err(Error::Limit("context source or filter stage count".into()));
        }
        if self.sources.iter().collect::<BTreeSet<_>>().len() != self.sources.len() {
            return Err(Error::Malformed("duplicate context source selector".into()));
        }
        for source in &self.sources {
            ContextSource::parse(&source.label())?;
        }
        let mut total = 0usize;
        for stage in &self.stages {
            match stage {
                ContextFilter::Match { field, values, .. } => {
                    if values.is_empty() || values.len() > 32 {
                        return Err(Error::Limit("context match values".into()));
                    }
                    for value in values {
                        if !clean(value, 256) {
                            return Err(Error::Malformed("context filter value invalid".into()));
                        }
                        if *field == MatchField::Kind {
                            RecordKind::parse(value)?;
                        }
                        total = total
                            .checked_add(value.len())
                            .ok_or_else(|| Error::Limit("context filter bytes".into()))?;
                    }
                }
                ContextFilter::Decision {
                    model,
                    question,
                    min_confidence_bps,
                    ..
                } => {
                    if !clean(model, 64)
                        || !model
                            .bytes()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
                        || !clean(question, 1024)
                        || *min_confidence_bps > 10_000
                    {
                        return Err(Error::Malformed(
                            "invalid typed decision filter contract".into(),
                        ));
                    }
                    total = total
                        .checked_add(question.len())
                        .ok_or_else(|| Error::Limit("context filter bytes".into()))?;
                }
            }
        }
        if total > 16 * 1024 {
            return Err(Error::Limit(
                "aggregate context filter control bytes".into(),
            ));
        }
        Ok(())
    }
    /// Compatibility selector for already-authorized record sets. A caller
    /// that needs to evaluate model decisions from record text should use
    /// `select_authorized_with_record`, which supplies the exact candidate.
    pub fn select_authorized(
        &self,
        candidates: &[AuthorizedRecord<'_>],
        mut decision: impl FnMut(&str, &str, usize, usize) -> Option<DecisionVerdict>,
    ) -> Result<Vec<usize>> {
        self.select_authorized_with_record(candidates, |model, question, stage, index, _record| {
            Ok(decision(model, question, stage, index))
        })
    }

    /// Execute ordered deterministic filtering on trusted, source-granted
    /// record VIEWS. A typed decision callback sees only survivors of earlier
    /// stages and no unrequested/unauthorized data. Propagate adapter errors
    /// rather than assuming an unknown model verdict or restoring a record.
    pub fn select_authorized_with_record(
        &self,
        candidates: &[AuthorizedRecord<'_>],
        mut decision: impl FnMut(
            &str,
            &str,
            usize,
            usize,
            &AuthorizedRecord<'_>,
        ) -> Result<Option<DecisionVerdict>>,
    ) -> Result<Vec<usize>> {
        self.validate()?;
        let sources: BTreeSet<_> = self.sources.iter().cloned().collect();
        let mut retained = Vec::new();
        for (index, item) in candidates.iter().enumerate() {
            // Permission is proved by the source owner, before calling this
            // pure filter. It only receives already granted views.
            if !sources.contains(&item.source) {
                continue;
            }
            let mut keep = true;
            for (stage_index, stage) in self.stages.iter().enumerate() {
                if !keep {
                    break;
                }
                match stage {
                    ContextFilter::Match {
                        action,
                        field,
                        values,
                    } => {
                        let matching = values.iter().any(|value| match field {
                            MatchField::Kind => item.source.kind.label() == value,
                            MatchField::Origin => item.source.origin == *value,
                            MatchField::ToolName => {
                                item.tool_name.is_some_and(|tool| tool == value)
                            }
                            MatchField::TextContains => item.text.contains(value),
                        });
                        keep = match action {
                            FilterAction::Include => matching,
                            FilterAction::Exclude => !matching,
                        };
                    }
                    ContextFilter::Decision {
                        model,
                        question,
                        min_confidence_bps,
                        unavailable,
                    } => {
                        let verdict = decision(model, question, stage_index, index, item)?;
                        keep = match verdict {
                            Some(v)
                                if v.confidence_bps <= 10_000
                                    && v.confidence_bps >= *min_confidence_bps =>
                            {
                                v.include
                            }
                            _ => match unavailable {
                                DecisionUnavailable::Omit => false,
                                DecisionUnavailable::Reject => {
                                    return Err(Error::Unsupported(
                                        "required context decision unavailable".into(),
                                    ));
                                }
                                DecisionUnavailable::KeepAuthorized => true,
                            },
                        };
                    }
                }
            }
            if keep {
                retained.push(index);
            }
        }
        Ok(retained)
    }
}
#[derive(Clone, Debug)]
pub struct AuthorizedRecord<'a> {
    /// The caller already checked this record's grant/provenance. This struct
    /// does not look up any user data or source metadata by itself.
    pub source: ContextSource,
    pub tool_name: Option<&'a str>,
    pub text: &'a str,
}
#[derive(Clone, Copy, Debug)]
pub struct DecisionVerdict {
    pub include: bool,
    pub confidence_bps: u16,
}
#[cfg(test)]
mod tests {
    use super::*;
    fn record<'a>(from: &str, tool: Option<&'a str>, text: &'a str) -> AuthorizedRecord<'a> {
        AuthorizedRecord {
            source: ContextSource::parse(from).unwrap(),
            tool_name: tool,
            text,
        }
    }
    #[test]
    fn default_omits_tools_and_app_prompts_but_explicit_granted_filter_selects_them() {
        let input = [
            record("session/user_message", None, "what about last week's plan"),
            record("session/tool_call", Some("search"), "query"),
            record("session/tool_result", Some("search"), "result"),
            record("application/system_prompt", None, "app-owned instructions"),
            record(
                "shared:team-alpha/assistant_message",
                None,
                "authorized shared transcript",
            ),
        ];
        let plan = ContextFilterPlan::default();
        assert_eq!(
            plan.select_authorized(&input, |_, _, _, _| None).unwrap(),
            [0]
        );
        let mut expanded = plan.clone();
        expanded.sources.extend(
            [
                "session/tool_call",
                "session/tool_result",
                "application/system_prompt",
                "shared:team-alpha/assistant_message",
            ]
            .into_iter()
            .map(|s| ContextSource::parse(s).unwrap()),
        );
        assert_eq!(
            expanded
                .select_authorized(&input, |_, _, _, _| None)
                .unwrap(),
            [0, 1, 2, 3, 4]
        );
        expanded.stages.push(ContextFilter::Match {
            action: FilterAction::Exclude,
            field: MatchField::Kind,
            values: vec!["tool_result".into()],
        });
        assert_eq!(
            expanded
                .select_authorized(&input, |_, _, _, _| None)
                .unwrap(),
            [0, 1, 3, 4]
        );
        expanded.stages.push(ContextFilter::Match {
            action: FilterAction::Exclude,
            field: MatchField::ToolName,
            values: vec!["search".into()],
        });
        assert_eq!(
            expanded
                .select_authorized(&input, |_, _, _, _| None)
                .unwrap(),
            [0, 3, 4]
        );
    }
    #[test]
    fn ordered_decision_cannot_resurrect_previously_filtered_or_unrequested_records() {
        let inputs = [
            record("session/user_message", None, "useful"),
            record("session/tool_result", Some("db"), "secret"),
            record("session/assistant_message", None, "off topic"),
        ];
        let mut policy = ContextFilterPlan::default();
        policy.stages.push(ContextFilter::Match {
            action: FilterAction::Exclude,
            field: MatchField::TextContains,
            values: vec!["off topic".into()],
        });
        policy.stages.push(ContextFilter::Decision {
            model: "decision".into(),
            question: "Is this relevant to current speech?".into(),
            min_confidence_bps: 8_000,
            unavailable: DecisionUnavailable::Omit,
        });
        let result = policy
            .select_authorized(&inputs, |_, _, _, _| {
                Some(DecisionVerdict {
                    include: true,
                    confidence_bps: 9_900,
                })
            })
            .unwrap();
        assert_eq!(result, [0]);
        assert!(
            policy
                .select_authorized(&inputs, |_, _, _, _| Some(DecisionVerdict {
                    include: false,
                    confidence_bps: 9_900
                }))
                .unwrap()
                .is_empty()
        );
        assert!(
            policy
                .select_authorized(&inputs, |_, _, _, _| None)
                .unwrap()
                .is_empty()
        );
        if let ContextFilter::Decision { unavailable, .. } = &mut policy.stages[1] {
            *unavailable = DecisionUnavailable::KeepAuthorized;
        }
        assert_eq!(
            policy
                .select_authorized(&inputs, |_, _, _, _| None)
                .unwrap(),
            [0]
        );
        if let ContextFilter::Decision { unavailable, .. } = &mut policy.stages[1] {
            *unavailable = DecisionUnavailable::Reject;
        }
        assert!(
            policy
                .select_authorized(&inputs, |_, _, _, _| None)
                .is_err()
        );
    }
    #[test]
    fn invalid_duplicate_sources_scope_names_bounds_and_decisions_fail_closed() {
        for input in [
            "session/tool_result",
            "application/developer_prompt",
            "shared:team-one/tool_call",
        ] {
            assert_eq!(ContextSource::parse(input).unwrap().label(), input);
        }
        for input in [
            "shared:/user_message",
            "shared:../user_message",
            "other/user_message",
            "session/not-a-kind",
            "shared:a/b/c",
        ] {
            assert!(ContextSource::parse(input).is_err(), "{input}");
        }
        let mut policy = ContextFilterPlan::default();
        policy.sources.push(policy.sources[0].clone());
        assert!(policy.validate().is_err());
        policy.sources.pop();
        policy.stages.push(ContextFilter::Decision {
            model: "nanojev".into(),
            question: "decision".into(),
            min_confidence_bps: 10_001,
            unavailable: DecisionUnavailable::Omit,
        });
        assert!(policy.validate().is_err());
    }
}
