//! Closed B2/B4/B8 typed body; no source/state/publisher ownership.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
pub(super) fn next_generation() -> Result<u64> {
    NEXT_GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::Limit("batch graph workspace generation exhausted".into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchGraphProfile {
    pub generation: u64,
    pub output: ResidentOutput,
    pub rows: usize,
    pub capacity: usize,
    pub body_nodes: usize,
    pub av_padding: usize,
}

pub(crate) struct BatchGraphReplay<'a> {
    pub profile: BatchGraphProfile,
    pub graph: &'a cubecl::client::Graph,
    pub body: BatchBodyReceipt,
    pub accepted_enqueues: &'a std::sync::atomic::AtomicUsize,
}

impl BatchProgram {
    pub(crate) fn graph_pinned(&self) -> bool {
        Arc::strong_count(&self.graph_pin) != 1
    }
    pub(crate) fn graph_pin(&self) -> Arc<()> {
        Arc::clone(&self.graph_pin)
    }

    pub(crate) fn graph_profile(
        &self,
        parameters: &[RuntimeParameters],
        output: ResidentOutput,
    ) -> Result<BatchGraphProfile> {
        checked_geometry(self.rows, self.inner.plan.prefix_capacity(), parameters)?;
        let count = output.nodes(&self.inner.plan).len();
        let mut attention = 0;
        for node in &self.nodes[..count] {
            match &node.operation {
                PlanOperation::AttentionValue { padding, .. } => {
                    if *padding != 256
                        || parameters
                            .iter()
                            .any(|p| p.valid_tokens == 0 || p.valid_tokens > 256)
                    {
                        return Err(Error::Unsupported(
                            "graph prototype requires AV padded256".into(),
                        ));
                    }
                    attention += 1;
                }
                PlanOperation::Embed { .. }
                | PlanOperation::Project { .. }
                | PlanOperation::View(_)
                | PlanOperation::Contiguous
                | PlanOperation::Rms { .. }
                | PlanOperation::Silu { .. }
                | PlanOperation::Add
                | PlanOperation::SigmoidGate
                | PlanOperation::Conv { .. }
                | PlanOperation::Coefficients { .. }
                | PlanOperation::Recurrent { .. }
                | PlanOperation::KvPrefixGuard { .. }
                | PlanOperation::Rope { .. }
                | PlanOperation::AppendKv
                | PlanOperation::AttentionQk { .. }
                | PlanOperation::AttentionSoftmax { .. }
                | PlanOperation::CheckFinite { .. }
                | PlanOperation::Observe { .. }
                | PlanOperation::ObserveState { .. } => {}
                _ => return Err(Error::Unsupported("unreviewed graph body operator".into())),
            }
        }
        if attention == 0 {
            return Err(Error::Unsupported(
                "graph requires original attention body".into(),
            ));
        }
        Ok(BatchGraphProfile {
            generation: self.graph_generation,
            output,
            rows: self.rows,
            capacity: 512,
            body_nodes: count,
            av_padding: 256,
        })
    }

    pub(crate) fn launch_graph(
        &mut self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        replay: BatchGraphReplay<'_>,
    ) -> Result<ResidentSubmission> {
        let BatchGraphReplay {
            profile,
            graph,
            body,
            accepted_enqueues,
        } = replay;
        self.validate_runtime(client, parameters, bindings)?;
        if self.graph_profile(parameters, profile.output)? != profile
            || !self.graph_pinned()
            || body.dispatches == 0
            || body.fused_dispatches == 0
        {
            return Err(Error::State("stale/unpinned graph body receipt".into()));
        }
        let prefix = self.launch_inputs(client, parameters, bindings, None)?;
        // SAFETY: enclosing physical flight retains original graph and every
        // source/workspace lease; same pinned stream/workspace lock orders IO.
        unsafe { graph.replay() }
            .map_err(|e| Error::State(format!("resident graph enqueue: {e}")))?;
        accepted_enqueues.fetch_add(1, Ordering::AcqRel);
        self.launch_outputs(client, bindings, profile.output, prefix, body)
    }
}

fn checked_geometry(rows: usize, capacity: usize, parameters: &[RuntimeParameters]) -> Result<()> {
    if !matches!(rows, 2 | 4 | 8) || capacity != 512 || parameters.len() != rows {
        return Err(Error::Unsupported(
            "graph prototype requires B2/B4/B8/capacity512".into(),
        ));
    }
    if parameters
        .iter()
        .any(|p| p.position >= 256 || p.valid_tokens != p.position + 1)
    {
        return Err(Error::Unsupported(
            "graph prototype requires exact AV padded256 frontier".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_graph_frontiers_have_one_exact_padding_topology() {
        let pair = |a, b| {
            [
                RuntimeParameters {
                    token: 11,
                    position: a,
                    valid_tokens: a + 1,
                },
                RuntimeParameters {
                    token: 353,
                    position: b,
                    valid_tokens: b + 1,
                },
            ]
        };
        checked_geometry(2, 512, &pair(0, 255)).unwrap();
        checked_geometry(2, 512, &pair(255, 0)).unwrap();
        assert!(checked_geometry(2, 512, &pair(0, 256)).is_err());
        assert!(checked_geometry(4, 512, &pair(0, 1)).is_err());
        assert!(checked_geometry(2, 256, &pair(0, 1)).is_err());
        assert!(checked_geometry(2, 512, &pair(0, 1)[..1]).is_err());
        let mut invalid = pair(0, 1);
        invalid[1].valid_tokens = 1;
        assert!(checked_geometry(2, 512, &invalid).is_err());
        invalid[1].valid_tokens = 0;
        assert!(checked_geometry(2, 512, &invalid).is_err());
    }

    #[test]
    fn each_native_band_retains_exact_independent_parameter_coverage() {
        for rows in [2, 4, 8] {
            let mut params = (0..rows)
                .map(|row| RuntimeParameters {
                    token: 11 + row as u32,
                    position: row * 31,
                    valid_tokens: row * 31 + 1,
                })
                .collect::<Vec<_>>();
            checked_geometry(rows, 512, &params).unwrap();
            params[rows - 1].position = 256;
            params[rows - 1].valid_tokens = 257;
            assert!(checked_geometry(rows, 512, &params).is_err());
            params.pop();
            assert!(checked_geometry(rows, 512, &params).is_err());
        }
    }

    #[test]
    fn workspace_identity_changes_when_prepared_storage_is_replaced() {
        let a = next_generation().unwrap();
        let b = next_generation().unwrap();
        assert_ne!(a, 0);
        assert!(b > a);
        let profile = BatchGraphProfile {
            generation: a,
            output: ResidentOutput::Logits,
            rows: 2,
            capacity: 512,
            body_nodes: 17,
            av_padding: 256,
        };
        assert_ne!(
            profile,
            BatchGraphProfile {
                generation: b,
                ..profile
            }
        );
        assert_ne!(
            profile,
            BatchGraphProfile {
                body_nodes: 18,
                ..profile
            }
        );
        assert_ne!(
            profile,
            BatchGraphProfile {
                av_padding: 512,
                ..profile
            }
        );
        assert_ne!(
            profile,
            BatchGraphProfile {
                output: ResidentOutput::StateOnly,
                ..profile
            }
        );
    }
}
