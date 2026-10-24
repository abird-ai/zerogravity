//! Same original SiLU buffers/certificates; no persistent binding metadata.
use super::*;

#[cfg(feature = "cuda")]
fn ordered_width(rows: usize, elements: usize) -> Result<Option<usize>> {
    if !matches!(rows, 1 | 2 | 4 | 8) || !elements.is_multiple_of(rows) {
        return Err(Error::State("resident SiLU row partition changed".into()));
    }
    let width = elements / rows;
    Ok(crate::ordered_silu::supports_geometry(rows, width).then_some(width))
}

impl ResidentProgram {
    #[cfg(feature = "cuda")]
    fn ordered_silu_binding(
        &self,
        client: &Client,
        node: &PlanNode,
        rows: usize,
    ) -> Result<Option<crate::ordered_silu::PreparedOrderedSilu>> {
        if !crate::activation_quant_fused::is_cuda_service(client) {
            return Ok(None);
        }
        if self.plan.numerical_profile()
            != zerogravity_core::accelerator::program::ProgramNumericalProfile::GgufSourceV1
            || self.plan.is_operator_slice()
        {
            return Err(Error::Unsupported(
                "resident SiLU numerical contract".into(),
            ));
        }
        let PlanOperation::Silu { multiplied } = &node.operation else {
            return Err(Error::State("resident SiLU operation changed".into()));
        };
        let input = |index: usize| -> Result<&DeviceBuffer> {
            node.inputs
                .get(index)
                .and_then(|id| self.values.get(id.0 as usize))
                .ok_or_else(|| Error::State("resident SiLU input binding absent".into()))
        };
        let input = input(0)?;
        let output = node
            .outputs
            .first()
            .and_then(|id| self.values.get(id.0 as usize))
            .ok_or_else(|| Error::State("resident SiLU output binding absent".into()))?;
        let multiplier = if *multiplied {
            Some(
                node.inputs
                    .get(1)
                    .and_then(|id| self.values.get(id.0 as usize))
                    .ok_or_else(|| Error::State("resident SiLU multiplier absent".into()))?,
            )
        } else {
            None
        };
        if output.elements != input.elements
            || multiplier.is_some_and(|value| value.elements != input.elements)
        {
            return Err(Error::State("resident SiLU value geometry changed".into()));
        }
        let Some(width) = ordered_width(rows, input.elements)? else {
            // Only the immutable closed geometry selects the original route.
            // Hardware/range/alias errors from prepare always propagate.
            return Ok(None);
        };
        crate::ordered_silu::PreparedOrderedSilu::prepare(
            client,
            rows,
            width,
            crate::ordered_silu::Bindings {
                input: &input.handle,
                multiplier: multiplier.map(|value| &value.handle),
                output: &output.handle,
                trace: None,
            },
        )
        .map(Some)
    }

    /// Full immutable plan, including StateOnly-skipped tail nodes. Successful
    /// prepare uses only stack bindings and original Arc handle clones.
    #[cfg(feature = "cuda")]
    pub(super) fn preflight_silu(&self, client: &Client, rows: usize) -> Result<()> {
        Self::require_legacy_contract(&self.plan)?;
        for node in self.plan.nodes() {
            if matches!(node.operation, PlanOperation::Silu { .. }) {
                self.ordered_silu_binding(client, node, rows)?;
            }
        }
        Ok(())
    }

    pub(super) fn launch_silu(
        &self,
        client: &Client,
        node: &PlanNode,
        rows: usize,
    ) -> Result<usize> {
        let PlanOperation::Silu { multiplied } = &node.operation else {
            return Err(Error::State("resident SiLU operation changed".into()));
        };
        let input = |i: usize| &self.values[node.inputs[i].0 as usize];
        let output = |i: usize| &self.values[node.outputs[i].0 as usize];
        let legacy = self.legacy()?;
        let s = |i: usize| &legacy.scratch[i];
        let n = input(0).elements;
        // Repeat the exact cold binding before this node's first launch. No
        // mutable origin/range/rows, persistent table or Err fallback exists.
        #[cfg(feature = "cuda")]
        let ordered = self.ordered_silu_binding(client, node, rows)?;
        #[cfg(not(feature = "cuda"))]
        let _ = rows;
        // SAFETY: original typed plan/ranges and owning submission retain the
        // same original input/output/status/scratch leases as before factoring.
        unsafe {
            crate::resident_bindings::certify_interval::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(&input(0).handle, n),
                arg(&self.status, 1),
                -256.0,
                256.0,
                n,
            );
        }
        #[cfg(feature = "cuda")]
        if let Some(ordered) = ordered {
            return Ok(1 + ordered.launch(client)?);
        }
        // Original CUDA-outside-geometry and Vulkan body, byte-equivalent
        // modulo indentation to the pre-integration PlanOperation::Silu arm.
        unsafe {
            for stage in 0..12 {
                crate::silu::silu_step::launch_unchecked(
                    client,
                    grid(n),
                    threads(),
                    arg(&input(0).handle, n),
                    arg(s(0), n * crate::silu::SCRATCH),
                    arg(&output(0).handle, n),
                    n,
                    stage,
                );
            }
            if *multiplied {
                crate::silu::silu_gate::launch_unchecked(
                    client,
                    grid(n),
                    threads(),
                    arg(&input(1).handle, n),
                    arg(&output(0).handle, n),
                    n,
                );
            }
        }
        Ok(13 + usize::from(*multiplied))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests;
