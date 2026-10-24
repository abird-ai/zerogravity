//! Pin refusals only; no synthetic SourceIdentity is an original source seal.
use super::*;

#[test]
fn exact_2b_opening_is_closed_without_global_source_limit_change() {
    assert!(require_2b_pins(CONFIG2B, INDEX2B, BYTES2B, MIB, 320, 312).is_ok());
    for (config, index, bytes, chunk, text, other) in [
        ("bad", INDEX2B, BYTES2B, MIB, 320, 312),
        (CONFIG2B, "bad", BYTES2B, MIB, 320, 312),
        (CONFIG2B, INDEX2B, BYTES2B - 1, MIB, 320, 312),
        (CONFIG2B, INDEX2B, BYTES2B, 4 * MIB, 320, 312),
        (CONFIG2B, INDEX2B, BYTES2B, MIB, 319, 312),
        (CONFIG2B, INDEX2B, BYTES2B, MIB, 320, 311),
    ] {
        assert!(matches!(
            require_2b_pins(config, index, bytes, chunk, text, other),
            Err(Error::Unsupported(_))
        ));
    }
}
