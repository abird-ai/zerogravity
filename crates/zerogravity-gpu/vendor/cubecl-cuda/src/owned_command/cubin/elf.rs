//! Bounded ELF64 little-endian CUDA image validation; no fatbin/decompression.
mod rotary_headers;
mod text_headers;
use super::failure;
use cubecl_core::server::ServerError;

const GENERIC_IDENT: [u8; 16] = [0x7f, b'E', b'L', b'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
// Prior original selected images: CUDA OSABI0x33, ABI7, e_version0x80.
const CUDA_IDENT: [u8; 16] = [
    0x7f, b'E', b'L', b'F', 2, 1, 1, 0x33, 7, 0, 0, 0, 0, 0, 0, 0,
];
// Observed captured CUTLASS image only; no generic version123 acceptance.
const CAPTURED_HEADER: [u8; 64] = [
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0xbe, 0x00, 0x7b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xc0, 0xcb, 0x15, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x63, 0x15, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x50, 0x05, 0x50, 0x00, 0x40, 0x00, 0x38, 0x00, 0x04, 0x00, 0x40, 0x00, 0xa1, 0x01, 0x01, 0x00,
];

// Exact selected canonical47 BA resource; no generic CUDA version123 route.
const CANONICAL47_HEADER: [u8; 64] = [
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0xbe, 0x00, 0x7b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xc0, 0xdb, 0x76, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x54, 0x76, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x50, 0x05, 0x50, 0x00, 0x40, 0x00, 0x38, 0x00, 0x04, 0x00, 0x40, 0x00, 0x1d, 0x02, 0x01, 0x00,
];
// Exact selected activation resource; no generic CUDA version129 acceptance.
const ACTIVATION_HEADER: [u8; 64] = [
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0xbe, 0x00, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x40, 0xc1, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x17, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x56, 0x05, 0x56, 0x00, 0x40, 0x00, 0x38, 0x00, 0x04, 0x00, 0x40, 0x00, 0xa9, 0x02, 0x01, 0x00,
];

// Exact Gate35 selected resource headers; full SHA remains sealed separately.
const ATTENTION_GATE_HEADERS: &[(usize, [u8; 64])] = &[
    (
        2679648,
        [
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xbe, 0x00, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x80, 0xe2, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x85,
            0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x56, 0x05, 0x56, 0x00, 0x40, 0x00, 0x38, 0x00,
            0x04, 0x00, 0x40, 0x00, 0x75, 0x01, 0x01, 0x00,
        ],
    ),
    (
        3397824,
        [
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xbe, 0x00, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0xe0, 0xd7, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa0, 0x56,
            0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0x56, 0x05, 0x56, 0x00, 0x40, 0x00, 0x38, 0x00,
            0x04, 0x00, 0x40, 0x00, 0x05, 0x02, 0x01, 0x00,
        ],
    ),
];

// Exact source-classified first-attention selected resource headers.
const ATTENTION_HEADERS: &[(usize, [u8; 64])] = &[
    (
        1813600,
        [
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xbe, 0x00, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x80, 0xab, 0x1b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54,
            0x1b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x56, 0x05, 0x56, 0x00, 0x40, 0x00, 0x38, 0x00,
            0x04, 0x00, 0x40, 0x00, 0x5e, 0x01, 0x01, 0x00,
        ],
    ),
    (
        6317664,
        [
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xbe, 0x00, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x80, 0x65, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x22,
            0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x05, 0x50, 0x00, 0x40, 0x00, 0x38, 0x00,
            0x04, 0x00, 0x40, 0x00, 0x0c, 0x01, 0x01, 0x00,
        ],
    ),
    (
        3596000,
        [
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xbe, 0x00, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0xde, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb3,
            0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x05, 0x50, 0x00, 0x40, 0x00, 0x38, 0x00,
            0x04, 0x00, 0x40, 0x00, 0xac, 0x00, 0x01, 0x00,
        ],
    ),
];

fn u16_at(bytes: &[u8], n: usize) -> u16 {
    u16::from_le_bytes(bytes[n..n + 2].try_into().unwrap())
}
fn u32_at(bytes: &[u8], n: usize) -> u32 {
    u32::from_le_bytes(bytes[n..n + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], n: usize) -> u64 {
    u64::from_le_bytes(bytes[n..n + 8].try_into().unwrap())
}
fn range(offset: u64, size: u64, length: usize) -> Result<std::ops::Range<usize>, ServerError> {
    let end = offset
        .checked_add(size)
        .filter(|end| *end <= length as u64)
        .ok_or_else(|| failure("cubin ELF range overflow/truncation"))?;
    Ok(offset as usize..end as usize)
}

pub(super) fn validate(bytes: &[u8]) -> Result<u8, ServerError> {
    if bytes.len() < 64
        || !((bytes[..16] == GENERIC_IDENT && u32_at(bytes, 20) == 1)
            || (bytes[..16] == CUDA_IDENT && u32_at(bytes, 20) == 0x80)
            || (bytes.len() == 1_428_640 && bytes[..64] == CAPTURED_HEADER)
            || (bytes.len() == 7_789_728 && bytes[..64] == CANONICAL47_HEADER)
            || (bytes.len() == 901_664 && bytes[..64] == ACTIVATION_HEADER)
            || ATTENTION_GATE_HEADERS
                .iter()
                .any(|(n, h)| bytes.len() == *n && bytes[..64] == *h)
            || rotary_headers::matches(bytes)
            || text_headers::matches(bytes)
            || ATTENTION_HEADERS
                .iter()
                .any(|(n, h)| bytes.len() == *n && bytes[..64] == *h))
        || !matches!(u16_at(bytes, 16), 1 | 2)
        || u16_at(bytes, 18) != 190
        || u16_at(bytes, 52) != 64
        || u16_at(bytes, 58) != 64
    {
        return Err(failure(
            "closed cubin requires relocatable/executable CUDA ELF64 LE",
        ));
    }
    let sm = (u32_at(bytes, 48) & 255) as u8;
    if !matches!(sm, 80 | 86) {
        return Err(failure("closed cubin binary SM80/SM86 only"));
    }
    let sections = u16_at(bytes, 60) as usize;
    let names = u16_at(bytes, 62) as usize;
    if sections == 0 || sections > 4096 || names == 0 || names >= sections {
        return Err(failure("closed cubin ELF section count/names"));
    }
    let table = range(u64_at(bytes, 40), (sections as u64) * 64, bytes.len())?;
    let mut extents = Vec::with_capacity(sections);
    for index in 0..sections {
        let h = &bytes[table.start + index * 64..table.start + (index + 1) * 64];
        let kind = u32_at(h, 4);
        let align = u64_at(h, 48);
        if align > 1 && !align.is_power_of_two() {
            return Err(failure("cubin ELF section alignment"));
        }
        let data = if kind == 8 {
            0..0
        } else {
            range(u64_at(h, 24), u64_at(h, 32), bytes.len())?
        };
        extents.push((kind, data));
    }
    if extents[names].0 != 3 {
        return Err(failure("cubin ELF section names table"));
    }
    let strings = &bytes[extents[names].1.clone()];
    if strings.first() != Some(&0) {
        return Err(failure("cubin ELF empty name missing"));
    }
    for index in 0..sections {
        let h = &bytes[table.start + index * 64..table.start + (index + 1) * 64];
        let name = u32_at(h, 0) as usize;
        if name >= strings.len() || !strings[name..].contains(&0) {
            return Err(failure("cubin ELF section name bounds"));
        }
        if matches!(extents[index].0, 2 | 11) {
            let link = u32_at(h, 40) as usize;
            if u64_at(h, 56) != 24
                || link >= sections
                || extents[link].0 != 3
                || !extents[index].1.len().is_multiple_of(24)
            {
                return Err(failure("cubin ELF symbol table/link/extent"));
            }
            let symbols = &bytes[extents[index].1.clone()];
            let strings = &bytes[extents[link].1.clone()];
            for symbol in symbols.chunks_exact(24) {
                let name = u32_at(symbol, 0) as usize;
                let section = u16_at(symbol, 6) as usize;
                if name >= strings.len()
                    || !strings[name..].contains(&0)
                    || (section < 0xff00 && section >= sections)
                {
                    return Err(failure("cubin ELF symbol name/section bounds"));
                }
            }
        }
    }
    let programs = u16_at(bytes, 56) as usize;
    if programs > 1024 || (programs != 0 && u16_at(bytes, 54) != 56) {
        return Err(failure("cubin ELF program table"));
    }
    if programs != 0 {
        let table = range(u64_at(bytes, 32), (programs as u64) * 56, bytes.len())?;
        for index in 0..programs {
            let h = &bytes[table.start + index * 56..table.start + (index + 1) * 56];
            range(u64_at(h, 8), u64_at(h, 32), bytes.len())?;
            if u64_at(h, 32) > u64_at(h, 40) {
                return Err(failure("cubin ELF program file/memory extent"));
            }
        }
    }
    Ok(sm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate129_images_require_exact_headers_extents_and_remaining_elf_guards() {
        for (bytes, header) in ATTENTION_GATE_HEADERS {
            let mut image = vec![0; *bytes];
            image[..64].copy_from_slice(header);
            let table = u64_at(&image, 40) as usize;
            let names = table + 64;
            image[names + 4..names + 8].copy_from_slice(&3u32.to_le_bytes());
            image[names + 24..names + 32].copy_from_slice(&128u64.to_le_bytes());
            image[names + 32..names + 40].copy_from_slice(&1u64.to_le_bytes());
            assert_eq!(validate(&image).unwrap(), 86);
            for offset in [7, 8, 12, 16, 18, 24, 32, 40, 48, 52, 54, 56, 58, 60, 62] {
                let mut bad = image.clone();
                bad[offset] ^= 1;
                assert!(validate(&bad).is_err(), "gate{bytes}/header{offset}");
            }
            let mut supported128 = image.clone();
            supported128[20] = 0x80;
            assert_eq!(
                validate(&supported128).unwrap(),
                86,
                "legacy CUDA128 structural route unchanged; capture issuer still pins full SHA"
            );
            let mut unsupported130 = image.clone();
            unsupported130[20] = 0x82;
            assert!(validate(&unsupported130).is_err());
            assert!(validate(&image[..image.len() - 1]).is_err());
            let mut bad = image.clone();
            bad[names + 4..names + 8].copy_from_slice(&1u32.to_le_bytes());
            assert!(validate(&bad).is_err());
        }
    }
    #[test]
    fn activation129_requires_exact_target_header_extent_and_remaining_elf_guards() {
        // Structural fixture only; closed sealing also requires the full SHA.
        let mut b = vec![0; 901_664];
        b[..64].copy_from_slice(&ACTIVATION_HEADER);
        let names = 857_856 + 64;
        b[names + 4..names + 8].copy_from_slice(&3u32.to_le_bytes());
        b[names + 24..names + 32].copy_from_slice(&128u64.to_le_bytes());
        b[names + 32..names + 40].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(validate(&b).unwrap(), 86);
        for offset in [7, 8, 12, 16, 18, 24, 32, 40, 48, 52, 54, 56, 58, 60, 62] {
            let mut bad = b.clone();
            bad[offset] ^= 1;
            assert!(validate(&bad).is_err(), "activation header byte{offset}");
        }
        let mut supported128 = b.clone();
        supported128[20..24].copy_from_slice(&128u32.to_le_bytes());
        assert_eq!(validate(&supported128).unwrap(), 86);
        let mut wrong_version = b.clone();
        wrong_version[20..24].copy_from_slice(&130u32.to_le_bytes());
        assert!(validate(&wrong_version).is_err());
        assert!(validate(&b[..b.len() - 1]).is_err());
        let mut trailing = b.clone();
        trailing.push(0);
        assert!(validate(&trailing).is_err());
        b[names + 24..names + 32].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(validate(&b).is_err());
        let mut generic129 = image();
        generic129[..16].copy_from_slice(&CUDA_IDENT);
        generic129[20..24].copy_from_slice(&129u32.to_le_bytes());
        assert!(validate(&generic129).is_err());
    }
    fn image() -> Vec<u8> {
        let mut b = vec![0u8; 193];
        b[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
        b[16..18].copy_from_slice(&2u16.to_le_bytes());
        b[18..20].copy_from_slice(&190u16.to_le_bytes());
        b[20..24].copy_from_slice(&1u32.to_le_bytes());
        b[40..48].copy_from_slice(&64u64.to_le_bytes());
        b[48..52].copy_from_slice(&80u32.to_le_bytes());
        b[52..54].copy_from_slice(&64u16.to_le_bytes());
        b[58..60].copy_from_slice(&64u16.to_le_bytes());
        b[60..62].copy_from_slice(&2u16.to_le_bytes());
        b[62..64].copy_from_slice(&1u16.to_le_bytes());
        b[132..136].copy_from_slice(&3u32.to_le_bytes());
        b[152..160].copy_from_slice(&192u64.to_le_bytes());
        b[160..168].copy_from_slice(&1u64.to_le_bytes());
        b
    }
    #[test]
    fn bounded_elf_rejects_truncation_overflow_and_foreign_machine() {
        let original = image();
        assert_eq!(validate(&original).unwrap(), 80);
        let mut relocatable = original.clone();
        relocatable[16..18].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(validate(&relocatable).unwrap(), 80);
        let mut shared_object = original.clone();
        shared_object[16..18].copy_from_slice(&3u16.to_le_bytes());
        assert!(validate(&shared_object).is_err());
        for length in [0, 7, 63, 64, 128, 192] {
            assert!(validate(&original[..length]).is_err());
        }
        for kind in 0..5 {
            let mut b = original.clone();
            match kind {
                0 => b[18] = 62,
                1 => b[48] = 90,
                2 => b[40..48].copy_from_slice(&u64::MAX.to_le_bytes()),
                3 => b[160..168].copy_from_slice(&u64::MAX.to_le_bytes()),
                _ => b[128..132].copy_from_slice(&9u32.to_le_bytes()),
            }
            assert!(validate(&b).is_err(), "case {kind}");
        }
    }

    #[test]
    fn observed_cuda_header_requires_exact_abi_version_and_machine() {
        for (kind, sm) in [(1u16, 80u32), (2, 86)] {
            let mut actual_header = image();
            actual_header[..16].copy_from_slice(&CUDA_IDENT);
            actual_header[16..18].copy_from_slice(&kind.to_le_bytes());
            actual_header[20..24].copy_from_slice(&0x80u32.to_le_bytes());
            actual_header[48..52].copy_from_slice(&sm.to_le_bytes());
            assert_eq!(validate(&actual_header).unwrap(), sm as u8);
            for fault in 0..11 {
                let mut bad = actual_header.clone();
                match fault {
                    0 => bad[7] = 0,
                    1 => bad[8] = 0,
                    2 => bad[8] = 8,
                    3 => bad[6] = 2,
                    4 => bad[9] = 1,
                    5 => bad[20..24].copy_from_slice(&1u32.to_le_bytes()),
                    6 => bad[20..24].copy_from_slice(&0x7fu32.to_le_bytes()),
                    7 => bad[20..24].copy_from_slice(&0x81u32.to_le_bytes()),
                    8 => bad[18..20].copy_from_slice(&62u16.to_le_bytes()),
                    9 => bad[4] = 1,
                    _ => bad[5] = 2,
                }
                assert!(validate(&bad).is_err(), "type{kind}/SM{sm}/fault{fault}");
            }
            actual_header[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(validate(&actual_header).is_err());
        }
    }

    #[test]
    fn captured123_requires_exact_target_header_extent_and_remaining_elf_guards() {
        // Synthetic structural fixture only; the captured seal independently
        // refuses this body because its complete SHA is not the native capture.
        let mut b = vec![0; 1_428_640];
        b[..64].copy_from_slice(&CAPTURED_HEADER);
        let names = 1_401_728 + 64;
        b[names + 4..names + 8].copy_from_slice(&3u32.to_le_bytes());
        b[names + 24..names + 32].copy_from_slice(&128u64.to_le_bytes());
        b[names + 32..names + 40].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(validate(&b).unwrap(), 80);
        for offset in [7, 8, 12, 16, 18, 20, 24, 32, 40, 48, 52, 54, 56, 58, 60, 62] {
            let mut bad = b.clone();
            bad[offset] ^= 1;
            assert!(validate(&bad).is_err(), "captured header byte{offset}");
        }
        assert!(validate(&b[..b.len() - 1]).is_err());
        let mut trailing = b.clone();
        trailing.push(0);
        assert!(validate(&trailing).is_err());
        b[names + 24..names + 32].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(validate(&b).is_err());
        let mut wrong123 = image();
        wrong123[..16].copy_from_slice(&CUDA_IDENT);
        wrong123[20..24].copy_from_slice(&123u32.to_le_bytes());
        assert!(validate(&wrong123).is_err());
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    #[test]
    fn attention129_is_exact_resource_header_extent_not_generic_version() {
        for &(bytes, header) in ATTENTION_HEADERS {
            let mut b = vec![0; bytes];
            b[..64].copy_from_slice(&header);
            let names = u64_at(&b, 40) as usize + 64;
            b[names + 4..names + 8].copy_from_slice(&3u32.to_le_bytes());
            b[names + 24..names + 32].copy_from_slice(&128u64.to_le_bytes());
            b[names + 32..names + 40].copy_from_slice(&1u64.to_le_bytes());
            assert_eq!(validate(&b).unwrap(), header[48]);
            for at in [7, 8, 12, 16, 18, 24, 32, 40, 48, 52, 54, 56, 58, 60, 62] {
                let mut bad = b.clone();
                bad[at] ^= 1;
                assert!(validate(&bad).is_err());
            }
            let mut old = b.clone();
            old[20..24].copy_from_slice(&128u32.to_le_bytes());
            assert_eq!(validate(&old).unwrap(), header[48]);
            let mut future = b.clone();
            future[20..24].copy_from_slice(&130u32.to_le_bytes());
            assert!(validate(&future).is_err());
            assert!(validate(&b[..b.len() - 1]).is_err());
            b.push(0);
            assert!(validate(&b).is_err());
        }
    }
}

#[cfg(test)]
mod fixed_prompt_header_tests {
    use super::*;
    fn structural_fixture() -> Vec<u8> {
        // Structural-only: full-image SHA sealing independently refuses this body.
        let mut image = vec![0; 7_789_728];
        image[..64].copy_from_slice(&CANONICAL47_HEADER);
        let table = u64_at(&image, 40) as usize;
        let names = table + 64;
        image[names + 4..names + 8].copy_from_slice(&3u32.to_le_bytes());
        image[names + 24..names + 32].copy_from_slice(&128u64.to_le_bytes());
        image[names + 32..names + 40].copy_from_slice(&1u64.to_le_bytes());
        image
    }
    #[test]
    fn fixed_prompt_header_matches_only_exact_selected64bytes_and_image_extent() {
        let image = structural_fixture();
        assert_eq!(validate(&image).unwrap(), 80);
        for offset in 0..64 {
            let mut bad = image.clone();
            bad[offset] ^= 1;
            assert!(validate(&bad).is_err(), "header byte{offset}");
        }
        assert!(validate(&image[..image.len() - 1]).is_err());
        let mut trailing = image.clone();
        trailing.push(0);
        assert!(validate(&trailing).is_err());
        let mut generic123 = vec![0; 193];
        generic123[..16].copy_from_slice(&CUDA_IDENT);
        generic123[16..18].copy_from_slice(&2u16.to_le_bytes());
        generic123[18..20].copy_from_slice(&190u16.to_le_bytes());
        generic123[20..24].copy_from_slice(&123u32.to_le_bytes());
        assert!(validate(&generic123).is_err());
    }
    #[test]
    fn fixed_prompt_exact_header_does_not_skip_section_or_program_guards() {
        let image = structural_fixture();
        let table = u64_at(&image, 40) as usize;
        let names = table + 64;
        let programs = u64_at(&image, 32) as usize;
        for fault in 0..5 {
            let mut bad = image.clone();
            match fault {
                0 => bad[names + 4..names + 8].copy_from_slice(&1u32.to_le_bytes()),
                1 => bad[names + 24..names + 32].copy_from_slice(&u64::MAX.to_le_bytes()),
                2 => bad[names + 48..names + 56].copy_from_slice(&3u64.to_le_bytes()),
                3 => bad[128] = 1,
                _ => bad[programs + 32..programs + 40].copy_from_slice(&1u64.to_le_bytes()),
            }
            assert!(validate(&bad).is_err(), "remaining structural guard{fault}");
        }
    }
}
