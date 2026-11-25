//! Cold direct CPU decisions and original owner-retaining materialization.
use crate::{
    Error, Result, devices,
    prepare::selection::{PreparedCpu, Selection},
    run_policy::{Diagnostics, Entrypoint, Request, Task},
};
use std::collections::BTreeMap;

pub struct Plan {
    request: Request,
    selected: Selection,
    decision: Option<(String, u64, u64)>,
    explanation: serde_json::Value,
    exclusions: Vec<String>,
}
pub struct Prepared {
    pub source: PreparedCpu,
    pub source_label: String,
    pub task: Task,
    pub diagnostics: Diagnostics,
}
impl Plan {
    /// Inspects bounded source metadata; no admission, materialization or stdin.
    pub fn new(mut request: Request, inventory: devices::Inventory) -> Result<Self> {
        super::validate_source_request(&request)?;
        let Entrypoint::Model(id) = &request.entrypoint else {
            request.graph.validate_entrypoint(&request.entrypoint)?;
            return Err(Error::Unsupported("graph run adapter is not qualified; graph validation succeeded, routes remain inert".into()));
        };
        let i = request
            .graph
            .models
            .iter()
            .position(|model| &model.id == id)
            .ok_or_else(|| Error::Malformed("direct entrypoint model absent".into()))?;
        if request.graph.models[i].native_artifacts.is_some() {
            return Err(Error::Unsupported(
                "native artifacts require the explicit finite diagnostic route".into(),
            ));
        }
        let candidates = devices::allowed(&request, &inventory)?;
        if !candidates.iter().any(|d| d.class == "host-memory") {
            return Err(Error::Unsupported("generic GGUF GPU run lacks source-issued complete cold/compiler costs and runtime binding; exact finite HF source uses --native-artifacts".into()));
        }
        let selected = Selection::resolve(&request.graph.models[i], request.task.context, None)?;
        if request.graph.models[i].target.is_empty() {
            request.graph.models[i].target = selected.model_target().to_owned();
        } else if request.graph.models[i].target != selected.model_target() {
            return Err(Error::Malformed(
                "declared model target differs from actual source geometry".into(),
            ));
        }
        request.graph.validate_entrypoint(&request.entrypoint)?;
        if request.task.context as u64 > selected.geometry().max_position_embeddings {
            return Err(Error::Limit(
                "requested context exceeds actual model geometry".into(),
            ));
        }

        let inventory = if devices::fresh(&inventory).is_err() {
            devices::discover()?
        } else {
            inventory
        };
        devices::fresh(&inventory)?;
        let candidates = devices::allowed(&request, &inventory)?;
        let mut decision = None;
        let mut exclusions = Vec::new();
        let backing = crate::resource_arbiter::ResourceArbiter::process_shared().host_backing()?;
        for d in &candidates {
            if d.class != "host-memory" {
                exclusions.push(format!(
                    "{}: {}",
                    d.id,
                    d.exclusion
                        .as_deref()
                        .unwrap_or("unqualified source/runtime binding")
                ));
                continue;
            }
            if !cfg!(all(
                feature = "cpu",
                feature = "model-qwen35",
                feature = "text"
            )) {
                exclusions.push("cpu:0: CPU/text adapter not compiled".into());
                continue;
            }
            let Some((ceiling, available)) = d.ceiling.zip(d.available) else {
                exclusions.push("cpu:0: current availability unknown".into());
                continue;
            };
            let margin = request.headroom.bytes(ceiling, true)?;
            let usable = backing.conservative_available(ceiling, available, margin);
            if usable >= selected.planning_bytes() {
                decision = Some((d.id.clone(), ceiling, available.saturating_sub(margin)));
                break;
            }
            exclusions.push(
                "cpu:0: insufficient currently observed memory after routing headroom".into(),
            );
        }
        let mut explanation = serde_json::json!({"schema":"zerogravity.run.decision.v1","source":selected.source_label(),"target":selected.model_target(),"backend":request.backend.label(),"selected":decision.as_ref().map(|d|&d.0),"context":request.task.context,"resolved_device_policy":format!("{:?}",request.devices),"resolved_headroom":format!("{:?}",request.headroom),"numerical":request.numerical,"max_tokens":request.task.max_tokens,"raw":request.task.raw,"prompt_bytes":request.task.prompt.as_ref().map(String::len),"prefill_chunk":request.task.prefill_chunk,"planning_bytes":selected.planning_bytes(),"planning_scope":"conservative CPU source/state/temporary planning; protect all existing host commitments as potential growth; no OS hard RAM guarantee","host_backing":backing,"availability_observed_unix_ms":inventory.observed_unix_ms,"excluded":exclusions,"speculation":"off (no qualified measured speculative route)","source_verified":false,"model_execution_qualified_by_metadata":false,"provenance":request.provenance.iter().map(|(k,v)|(k,format!("{v:?}"))).collect::<BTreeMap<_,_>>()});

        if request.diagnostics.dry_run {
            explanation["dry_run"] = serde_json::json!(true);
            explanation["reservation"] = serde_json::json!("none");
        }
        Ok(Self {
            request,
            selected,
            decision,
            explanation,
            exclusions,
        })
    }
    pub fn decision(&self) -> &serde_json::Value {
        &self.explanation
    }
    pub fn dry_run(&self) -> bool {
        self.request.diagnostics.dry_run
    }
    pub fn explain(&self) -> bool {
        self.request.diagnostics.explain
    }
    pub fn require_proposal(&self) -> Result<()> {
        if self.decision.is_some() {
            Ok(())
        } else {
            Err(Error::Unsupported(
                "no eligible local device; see dry-run exclusions".into(),
            ))
        }
    }
    pub fn prepare(self) -> Result<Prepared> {
        if self.request.diagnostics.dry_run {
            return Err(Error::State(
                "inspection plan cannot prepare execution".into(),
            ));
        }
        let (_, ceiling, available) = self.decision.ok_or_else(|| {
            Error::Unsupported(format!(
                "no eligible local device: {}",
                self.exclusions.join("; ")
            ))
        })?;
        let source_label = self.selected.source_label().to_owned();
        let source = self
            .selected
            .prepare_on_host(self.request.headroom, ceiling, available)?;
        Ok(Prepared {
            source,
            source_label,
            task: self.request.task,
            diagnostics: self.request.diagnostics,
        })
    }
}
