//! Gemma4 sparse expert geometry and original tensor metadata.
//! Numerical routing/execution is not implemented by this contract module.

use zerogravity_core::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeGeometry {
    hidden_size: usize,
    experts: usize,
    selected_experts: usize,
    intermediate_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeTensorSpec {
    pub name: String,
    /// Original Safetensors order, without a transpose or expert permutation.
    pub shape: Vec<usize>,
}

impl MoeGeometry {
    pub fn new(
        hidden_size: usize,
        experts: usize,
        selected_experts: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        if hidden_size == 0
            || experts == 0
            || selected_experts == 0
            || selected_experts > experts
            || intermediate_size == 0
        {
            return Err(Error::Malformed("invalid Gemma4 expert geometry".into()));
        }
        if hidden_size > 131_072 || experts > 65_536 || intermediate_size > 1_048_576 {
            return Err(Error::Limit(
                "Gemma4 expert geometry exceeds contract bounds".into(),
            ));
        }
        experts
            .checked_mul(2)
            .and_then(|n| n.checked_mul(intermediate_size))
            .and_then(|n| n.checked_mul(hidden_size))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Limit("Gemma4 expert tensor byte count overflow".into()))?;
        Ok(Self {
            hidden_size,
            experts,
            selected_experts,
            intermediate_size,
        })
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn experts(&self) -> usize {
        self.experts
    }
    pub fn selected_experts(&self) -> usize {
        self.selected_experts
    }
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Learned MoE arrays in the pinned official HF checkpoint namespace.
    /// Dense MLP and all pre/post norms are separate required family tensors.
    pub fn tensors(&self, layer: usize) -> Vec<MoeTensorSpec> {
        let prefix = format!("model.language_model.layers.{layer}");
        [
            ("router.proj.weight", vec![self.experts, self.hidden_size]),
            ("router.scale", vec![self.hidden_size]),
            ("router.per_expert_scale", vec![self.experts]),
            (
                "experts.gate_up_proj",
                vec![self.experts, 2 * self.intermediate_size, self.hidden_size],
            ),
            (
                "experts.down_proj",
                vec![self.experts, self.hidden_size, self.intermediate_size],
            ),
        ]
        .into_iter()
        .map(|(name, shape)| MoeTensorSpec {
            name: format!("{prefix}.{name}"),
            shape,
        })
        .collect()
    }
}
