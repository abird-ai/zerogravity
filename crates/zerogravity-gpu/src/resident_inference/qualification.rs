//! Source/physical diagnostics, compiled only by explicit qualification.
use super::*;

pub fn compare_domains(
    expected: &[(StateSlot, Vec<f32>)],
    actual: &[(StateSlot, Vec<f32>)],
) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(Error::State("physical domain coverage".into()));
    }
    for (slot, values) in expected {
        let (_, got) = actual
            .iter()
            .find(|(s, _)| s == slot)
            .ok_or_else(|| Error::State("missing physical state slot".into()))?;
        exact(values, got, &format!("state {slot:?}"))?;
    }
    Ok(())
}

pub fn exact(expected: &[f32], actual: &[f32], name: &str) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(Error::State(format!(
            "{name} length {} != {}",
            expected.len(),
            actual.len()
        )));
    }
    if let Some((index, (e, a))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (e, a))| e.to_bits() != a.to_bits())
    {
        return Err(Error::State(format!(
            "{name} bits differ at {index}: expected {:08x}, got {:08x}",
            e.to_bits(),
            a.to_bits()
        )));
    }
    Ok(())
}
