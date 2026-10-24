use super::io::*;
pub(super) const MIB: u64 = 1024 * 1024;
pub(super) const HOST: u64 = 768 * MIB;
// Original64/H48 geometry. Actual source is checked before session creation.
pub(super) fn derived_upper() -> Proof<u64> {
    let (channels, values, state, gdn) = (10240u64, 6144u64, 48u64 * 128 * 128, 48u64);
    let output = (channels * 5 + state + values * 2) * 4;
    let initial = (channels * 3 * 4).max(64 * 1024) + (state * 4).max(64 * 1024);
    let retained = gdn * output.max(initial) * 2;
    let trace = 66 * MIB;
    let scratch = (state * 5 + channels * 4 + values * 16 + 8192) * 4;
    // Source metadata64, one verified staging64, active operator64, control32,
    // source operators4, session256KiB, trace64KiB. Supersets overlap safely.
    // Added full-fourth-frame source owners across committed and proposal roots.
    let conv4 = gdn * channels * 4 * 4 * 2;
    let upper = retained + conv4 + trace + scratch + (64 + 64 + 64 + 32 + 4) * MIB + 320 * 1024;
    need(
        upper <= HOST,
        "derived whole diagnostic envelope exceeds explicit host limit",
    )?;
    Ok(upper)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_reservations_price_outputs_and_two_roots_without_dead_scratch() {
        assert_eq!(derived_upper().unwrap(), 667_025_408);
        assert!(derived_upper().unwrap() < HOST);
        assert_eq!((10240 * 5 + 48 * 128 * 128 + 6144 * 2) * 4, 3_399_680);
    }
}

// All extra values are original computed owners. Core/gated already share the
// surviving GDN state ticket; norm_gate borrows Z. This rounds the remaining
// nine 1MiB projection/add owners plus three widened17408 FFN owners upward.
pub(super) fn derived_upper_with_layer9(enabled: bool) -> Proof<u64> {
    let upper = derived_upper()? + if enabled { 13 * MIB } else { 0 };
    need(
        upper <= HOST,
        "layer9 trace within same explicit768MiB limit",
    )?;
    Ok(upper)
}
