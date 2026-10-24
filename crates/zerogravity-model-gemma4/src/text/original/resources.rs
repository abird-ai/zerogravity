//! Allocation layout for the same sequential E2B traversal, not a second schedule.
use super::*;
use std::mem::size_of;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma4CpuInventory {
    pub source_metadata_bytes: u64,
    pub model_metadata_bytes: u64,
    pub initialization_parser_bytes: u64,
    pub retained_parameter_bytes: u64,
    pub state_metadata_bytes: u64,
    pub complete_kv_bytes: u64,
    pub traversal_output_upper_bytes: u64,
    pub source_window_scratch_bytes: u64,
    pub decoder_scratch_bytes: u64,
    pub caller_control_bytes: u64,
    pub input_output_metadata_bytes: u64,
    pub total_upper_bytes: u64,
}
impl OriginalGemma4Text {
    /// Called on authenticated header/config metadata BEFORE binding parameters.
    /// Two independent sessions, one current/proposed traversal at a time.
    pub fn cpu_inventory(
        source: &VerifiedTensorSource,
        config_json: &[u8],
    ) -> Result<Gemma4CpuInventory> {
        let config = Gemma4Config::from_json(config_json)?;
        if config.variant() != crate::Gemma4Variant::E2b {
            return Err(Error::Unsupported("ordinary candidate is E2B only".into()));
        }
        let g = Geometry::from_config(&config)?;
        let parameter_bytes = parameter_names(&g).values().try_fold(0u64, |sum, name| {
            let n = source.parameter(name)?.elements();
            if n == 0 || n > 16384 {
                return Err(Error::Limit("ordinary source parameter extent".into()));
            }
            sum.checked_add(n as u64 * 4)
                .ok_or_else(|| Error::Limit("ordinary parameter byte sum".into()))
        })?;
        let declared = Self::cpu_declared_inventory(config_json)?;
        let actual = layout(&g, source.metadata_reserved_bytes(), parameter_bytes)?;
        if actual != declared {
            return Err(Error::State(
                "ordinary authenticated resource layout differs from source declaration".into(),
            ));
        }
        Ok(actual)
    }
    /// Pure registered source layout; no allocation/Ready/learned permit.
    pub fn cpu_declared_inventory(config_json: &[u8]) -> Result<Gemma4CpuInventory> {
        let config = Gemma4Config::from_json(config_json)?;
        if config.variant() != crate::Gemma4Variant::E2b {
            return Err(Error::Unsupported("ordinary candidate is E2B only".into()));
        }
        let g = Geometry::from_config(&config)?;
        let specs = config.tensor_specs()?;
        let bytes = parameter_names(&g).values().try_fold(0u64, |sum, name| {
            let spec = specs
                .iter()
                .find(|s| &s.name == name)
                .ok_or_else(|| Error::Malformed("registered parameter layout absent".into()))?;
            let elements = spec.shape.iter().try_fold(1usize, |p, n| {
                p.checked_mul(*n)
                    .ok_or_else(|| Error::Limit("parameter shape overflow".into()))
            })?;
            sum.checked_add(elements as u64 * 4)
                .ok_or_else(|| Error::Limit("parameter layout overflow".into()))
        })?;
        layout(&g, 64 * 1024 * 1024, bytes)
    }
}
pub(super) fn layout(
    g: &Geometry,
    source_bytes: u64,
    parameter_bytes: u64,
) -> Result<Gemma4CpuInventory> {
    let mut elements = 2 * g.hidden + 7 * g.layers.len() * g.ple;
    let mut outputs = 9usize;
    for (layer, spec) in g.layers.iter().enumerate() {
        // Sum EVERY producer in one token, conservatively retaining all
        // shadows through its complete layer. Loop locals retire per layer;
        // summing across all35 also bounds their actual shorter lifetimes.
        let q = spec.query_heads * spec.head_dim;
        let kv = if spec.kv_source_layer == layer {
            5 * spec.kv_heads * spec.head_dim
        } else {
            0
        };
        elements = elements
            .checked_add(
                12 * g.hidden
                    + 4 * q
                    + kv
                    + 2 * spec.query_heads * 64
                    + 4 * spec.feed_forward_width
                    + 3 * g.ple,
            )
            .ok_or_else(|| Error::Limit("ordinary activation sum".into()))?;
        outputs += 40;
    }
    elements = elements
        .checked_add(g.hidden + 2 * g.vocab)
        .ok_or_else(|| Error::Limit("ordinary head output sum".into()))?;
    outputs += 3;
    let state_bytes = 15 * 64 * size_of::<(ReferenceTextValue, ReferenceTextValue)>() + 65536;
    // Source produce prices each KV row's full payload plus256B owner.
    let complete_kv = 2 * 64 * (2 * (4608 * 4 + 15 * 256));
    let mut inventory = Gemma4CpuInventory {
        source_metadata_bytes: source_bytes,
        model_metadata_bytes: 2 * 1024 * 1024,
        initialization_parser_bytes: 4 * 1024 * 1024,
        retained_parameter_bytes: parameter_bytes,
        state_metadata_bytes: (3 * state_bytes) as u64,
        complete_kv_bytes: complete_kv as u64,
        traversal_output_upper_bytes: (elements * 4 + outputs * 256) as u64,
        // Largest source window<=1MiB; full-width project scratch<=21504*8,
        // plus output128rows. Extra1MiB bounds RMS/GELU/softmax temporaries.
        source_window_scratch_bytes: 3 * 1024 * 1024 + 21504 * 8 + 128 * 4 + 96,
        decoder_scratch_bytes: 8 * 1024 * 1024 + 65536,
        caller_control_bytes: 3 * 1024 * 1024,
        // Up to8 pretokenized requests +two bounded generated-ID owners.
        input_output_metadata_bytes: 10 * 65536,
        total_upper_bytes: 0,
    };
    inventory.total_upper_bytes = [
        inventory.source_metadata_bytes,
        inventory.model_metadata_bytes,
        inventory.initialization_parser_bytes,
        inventory.retained_parameter_bytes,
        inventory.state_metadata_bytes,
        inventory.complete_kv_bytes,
        inventory.traversal_output_upper_bytes,
        inventory.source_window_scratch_bytes,
        inventory.decoder_scratch_bytes,
        inventory.caller_control_bytes,
        inventory.input_output_metadata_bytes,
    ]
    .into_iter()
    .try_fold(0u64, |sum, n| {
        sum.checked_add(n)
            .ok_or_else(|| Error::Limit("ordinary complete resource sum".into()))
    })?;
    if inventory.total_upper_bytes > 128 * 1024 * 1024 {
        return Err(Error::Limit(
            "ordinary Gemma complete layout exceeds128MiB".into(),
        ));
    }
    Ok(inventory)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_registered_e2b_layout_includes_full_state_and_stays_inside_original_cap() {
        let raw = include_bytes!("../../../tests/fixtures/official-e2b-it-config.json");
        let layout = OriginalGemma4Text::cpu_declared_inventory(raw).unwrap();
        assert_eq!(layout.source_metadata_bytes, 64 * 1024 * 1024);
        assert_eq!(layout.complete_kv_bytes, 5_701_632);
        assert_eq!(layout.decoder_scratch_bytes, 8 * 1024 * 1024 + 65536);
        assert_eq!(layout.input_output_metadata_bytes, 10 * 65536);
        assert!(layout.total_upper_bytes < 128 * 1024 * 1024);
        assert!(
            layout.retained_parameter_bytes > 0
                && layout.traversal_output_upper_bytes > 2 * 262144 * 4
        );
        println!("GEMMA_E2B_DECLARED_RESOURCE_LAYOUT {layout:?}");
    }
    #[test]
    fn foreign_dense_and_moe_resource_profiles_are_not_e2b_permission() {
        for raw in [
            include_bytes!("../../../tests/fixtures/official-31b-it-config.json").as_slice(),
            include_bytes!("../../../tests/fixtures/official-26b-it-config.json").as_slice(),
        ] {
            assert!(matches!(
                OriginalGemma4Text::cpu_declared_inventory(raw),
                Err(Error::Unsupported(_))
            ));
        }
    }
}
