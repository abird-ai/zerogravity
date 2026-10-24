//! Logical hybrid state domains; caller supplies authenticated layer kinds.
use super::*;
pub fn hybrid_state_domains(
    attention_layers: impl IntoIterator<Item = bool>,
) -> Result<Vec<StateSlot>> {
    let mut domains = Vec::new();
    for (layer, attention) in attention_layers.into_iter().enumerate() {
        if layer >= 4096 {
            return Err(Error::Limit("hybrid state layer bound".into()));
        }
        if attention {
            domains.push(StateSlot::Kv(KvDomain(layer)));
        } else {
            domains.extend([
                StateSlot::Conv(ConvDomain(layer)),
                StateSlot::Recurrent(RecurrentDomain(layer)),
            ]);
        }
    }
    if domains.is_empty() {
        return Err(Error::State("hybrid state needs actual layers".into()));
    }
    Ok(domains)
}
