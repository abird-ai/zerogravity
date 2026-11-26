//! Compact human presentation; the observed byte counts remain unchanged.
pub(super) fn bytes(bytes: u64) -> String {
    const UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut unit = 0;
    let mut divisor = 1u128;
    let bytes = u128::from(bytes);
    while unit + 1 < UNITS.len() && bytes >= divisor * 1024 {
        unit += 1;
        divisor *= 1024;
    }
    if unit == 0 {
        return format!("{bytes} B");
    }
    let mut scaled = (bytes * 100 + divisor / 2) / divisor;
    if scaled >= 102_400 && unit + 1 < UNITS.len() {
        unit += 1;
        divisor *= 1024;
        scaled = (bytes * 100 + divisor / 2) / divisor;
    }
    let whole = scaled / 100;
    let fraction = scaled % 100;
    let number = match fraction {
        0 => whole.to_string(),
        n if n % 10 == 0 => format!("{whole}.{}", n / 10),
        n => format!("{whole}.{n:02}"),
    };
    format!("{number} {}", UNITS[unit])
}
pub(super) fn exact_bytes(value: u64) -> String {
    format!("{} ({value} bytes)", bytes(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_binary_units_and_rounding() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(1023), "1023 B");
        assert_eq!(bytes(1024), "1 KiB");
        assert_eq!(bytes(1536), "1.5 KiB");
        assert_eq!(bytes(32_198_128), "30.71 MiB");
        assert_eq!(bytes(4_571_272_453), "4.26 GiB");
        assert_eq!(bytes(55_586_113_293), "51.77 GiB");
        assert_eq!(bytes(1024 * 1024 - 1), "1 MiB");
        assert_eq!(bytes(u64::MAX), "16 EiB");
    }
}
