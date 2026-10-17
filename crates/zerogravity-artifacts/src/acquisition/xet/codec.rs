//! Bounded Xet protocol1.1 chunk decoding into caller-admitted buffers.
//! Frame/block rules: LZ4 ebb370ca83af193212df4dcbadcc5d87bc0de2f0.
//! Xet framing/BG4: xet-core c3c726bed5cf54ded92a63fab892cfb7857c751a.
//! No allocation, I/O, cache or source-identity authority lives here.
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) const HEADER_BYTES: usize = 8;
pub(crate) const MAX_RAW_BYTES: usize = 128 * 1024;
pub(crate) const MAX_COMPRESSED_BYTES: usize = 256 * 1024;
const FRAME_MAGIC: u32 = 0x184d2204;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Codec {
    None,
    Lz4,
    Bg4Lz4,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ChunkHeader {
    codec: Codec,
    compressed: usize,
    raw: usize,
}
impl ChunkHeader {
    pub(crate) fn parse(bytes: [u8; HEADER_BYTES]) -> Result<Self> {
        if bytes[0] != 0 {
            return Err(Error::Unsupported("unsupported Xet chunk version".into()));
        }
        let codec = match bytes[4] {
            0 => Codec::None,
            1 => Codec::Lz4,
            2 => Codec::Bg4Lz4,
            _ => return Err(Error::Unsupported("unsupported Xet chunk codec".into())),
        };
        let compressed = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], 0]) as usize;
        let raw = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], 0]) as usize;
        if compressed > MAX_COMPRESSED_BYTES || raw > MAX_RAW_BYTES {
            return Err(Error::Limit("Xet chunk size bound".into()));
        }
        if codec == Codec::None && compressed != raw {
            return Err(malformed());
        }
        Ok(Self {
            codec,
            compressed,
            raw,
        })
    }
    pub(crate) fn compressed_bytes(self) -> usize {
        self.compressed
    }
    #[cfg(test)]
    pub(crate) fn raw_bytes(self) -> usize {
        self.raw
    }
}
fn malformed() -> Error {
    Error::Malformed("invalid bounded Xet chunk".into())
}
fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::State("Xet decoding canceled".into()))
    } else {
        Ok(())
    }
}

/// `compressed` is exactly this chunk's declared payload. Output and BG4 scratch
/// are borrowed caller buffers, each bounded to MAX_RAW_BYTES. No bytes after
/// the returned raw prefix belong to this chunk. Errors can leave partial output.
pub(crate) fn decode_chunk(
    header: ChunkHeader,
    compressed: &[u8],
    output: &mut [u8],
    scratch: &mut [u8],
    cancel: &AtomicBool,
) -> Result<usize> {
    check_cancel(cancel)?;
    if compressed.len() != header.compressed || output.len() < header.raw {
        return Err(malformed());
    }
    if output.len() > MAX_RAW_BYTES || scratch.len() > MAX_RAW_BYTES {
        return Err(Error::Limit("Xet decoder scratch bound".into()));
    }
    let output = &mut output[..header.raw];
    match header.codec {
        Codec::None => copy_checked(output, compressed, cancel)?,
        Codec::Lz4 => decode_frame(compressed, output, cancel)?,
        Codec::Bg4Lz4 => {
            if scratch.len() < header.raw {
                return Err(malformed());
            }
            let grouped = &mut scratch[..header.raw];
            decode_frame(compressed, grouped, cancel)?;
            let quotient = grouped.len() / 4;
            let remainder = grouped.len() % 4;
            let starts = [
                0,
                quotient + usize::from(remainder > 0),
                2 * quotient + remainder.min(2),
                3 * quotient + remainder.min(3),
            ];
            for (index, byte) in output.iter_mut().enumerate() {
                if index % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                *byte = grouped[starts[index % 4] + index / 4];
            }
        }
    }
    check_cancel(cancel)?;
    Ok(header.raw)
}
fn copy_checked(output: &mut [u8], input: &[u8], cancel: &AtomicBool) -> Result<()> {
    if input.len() != output.len() {
        return Err(malformed());
    }
    for (output, input) in output.chunks_mut(4096).zip(input.chunks(4096)) {
        check_cancel(cancel)?;
        output.copy_from_slice(input);
    }
    check_cancel(cancel)
}
struct Input<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Input<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(malformed)?;
        let data = &self.bytes[self.at..end];
        self.at = end;
        Ok(data)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn word(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| malformed())?,
        ))
    }
}
fn checksum(bytes: &[u8], cancel: &AtomicBool) -> Result<u32> {
    let mut hasher = xxhash_rust::xxh32::Xxh32::new(0);
    for bytes in bytes.chunks(4096) {
        check_cancel(cancel)?;
        hasher.update(bytes);
    }
    check_cancel(cancel)?;
    Ok(hasher.digest())
}
fn decode_frame(compressed: &[u8], output: &mut [u8], cancel: &AtomicBool) -> Result<()> {
    let mut input = Input {
        bytes: compressed,
        at: 0,
    };
    match input.word()? {
        FRAME_MAGIC => {}
        0x184c2102 | 0x184d2a50..=0x184d2a5f => {
            return Err(Error::Unsupported(
                "legacy or skippable Xet LZ4 frame".into(),
            ));
        }
        _ => return Err(malformed()),
    }
    let flags = input.byte()?;
    let block_descriptor = input.byte()?;
    if flags & 0xc0 != 0x40 || flags & 0x02 != 0 || block_descriptor & 0x8f != 0 {
        return Err(malformed());
    }
    let maximum = match (block_descriptor >> 4) & 7 {
        4 => 64 * 1024,
        5 => 256 * 1024,
        6 => 1024 * 1024,
        7 => 4 * 1024 * 1024,
        _ => return Err(malformed()),
    };
    if flags & 1 != 0 {
        return Err(Error::Unsupported(
            "Xet LZ4 dictionary is unsupported".into(),
        ));
    }
    if flags & 8 != 0 {
        let declared = u64::from_le_bytes(input.take(8)?.try_into().map_err(|_| malformed())?);
        if declared != output.len() as u64 {
            return Err(malformed());
        }
    }
    let header_end = input.at;
    if input.byte()? != (checksum(&compressed[4..header_end], cancel)? >> 8) as u8 {
        return Err(malformed());
    }
    let mut written = 0usize;
    loop {
        check_cancel(cancel)?;
        let word = input.word()?;
        if word == 0 {
            break;
        }
        let block_size = (word & 0x7fffffff) as usize;
        if block_size > maximum {
            return Err(malformed());
        }
        let block = input.take(block_size)?;
        if flags & 0x10 != 0 && input.word()? != checksum(block, cancel)? {
            return Err(malformed());
        }
        let block_start = written;
        if word & 0x80000000 != 0 {
            let end = written
                .checked_add(block_size)
                .filter(|end| *end <= output.len())
                .ok_or_else(malformed)?;
            copy_checked(&mut output[written..end], block, cancel)?;
            written = end;
        } else {
            let dictionary_start = if flags & 0x20 != 0 { block_start } else { 0 };
            decode_block(block, output, &mut written, dictionary_start, cancel)?;
        }
        if written - block_start > maximum {
            return Err(malformed());
        }
    }
    if written != output.len() {
        return Err(malformed());
    }
    if flags & 4 != 0 && input.word()? != checksum(output, cancel)? {
        return Err(malformed());
    }
    if input.at != compressed.len() {
        if compressed[input.at..].starts_with(&FRAME_MAGIC.to_le_bytes()) {
            return Err(Error::Unsupported(
                "concatenated Xet LZ4 frames are unsupported".into(),
            ));
        }
        return Err(malformed());
    }
    check_cancel(cancel)
}
fn extended_length(input: &mut Input<'_>, base: usize, cancel: &AtomicBool) -> Result<usize> {
    let mut length = base;
    if base == 15 {
        loop {
            check_cancel(cancel)?;
            let byte = input.byte()?;
            length = length
                .checked_add(byte as usize)
                .filter(|length| *length <= MAX_RAW_BYTES)
                .ok_or_else(malformed)?;
            if byte != 255 {
                break;
            }
        }
    }
    Ok(length)
}
fn decode_block(
    bytes: &[u8],
    output: &mut [u8],
    written: &mut usize,
    dictionary_start: usize,
    cancel: &AtomicBool,
) -> Result<()> {
    let mut input = Input { bytes, at: 0 };
    let mut last_match = None;
    loop {
        check_cancel(cancel)?;
        let token = input.byte()?;
        let literal_length = extended_length(&mut input, (token >> 4) as usize, cancel)?;
        let literals = input.take(literal_length)?;
        let end = written
            .checked_add(literal_length)
            .filter(|end| *end <= output.len())
            .ok_or_else(malformed)?;
        copy_checked(&mut output[*written..end], literals, cancel)?;
        *written = end;
        if input.at == bytes.len() {
            // The final sequence must contain literals only. A match-bearing
            // block also needs five final literals and twelve bytes since the
            // last match start (the standard LZ4 block ending restrictions).
            if last_match.is_some_and(|start| literal_length < 5 || *written - start < 12) {
                return Err(malformed());
            }
            return check_cancel(cancel);
        }
        let offset_bytes = input.take(2)?;
        let distance = u16::from_le_bytes([offset_bytes[0], offset_bytes[1]]) as usize;
        if distance == 0 || distance > written.saturating_sub(dictionary_start) {
            return Err(malformed());
        }
        let match_length = extended_length(&mut input, (token & 15) as usize, cancel)?
            .checked_add(4)
            .ok_or_else(malformed)?;
        let end = written
            .checked_add(match_length)
            .filter(|end| *end <= output.len())
            .ok_or_else(malformed)?;
        last_match = Some(*written);
        while *written != end {
            if (*written).is_multiple_of(4096) {
                check_cancel(cancel)?;
            }
            output[*written] = output[*written - distance];
            *written += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const COMPRESSED: &str = "04224d186040820c0000003b616263030050454e44212100000000";
    fn hex(text: &str) -> Vec<u8> {
        assert!(text.len().is_multiple_of(2));
        text.as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }
    fn header(codec: u8, compressed: usize, raw: usize) -> Result<ChunkHeader> {
        let mut bytes = [0u8; 8];
        bytes[1..4].copy_from_slice(&(compressed as u32).to_le_bytes()[..3]);
        bytes[4] = codec;
        bytes[5..8].copy_from_slice(&(raw as u32).to_le_bytes()[..3]);
        ChunkHeader::parse(bytes)
    }
    fn decode(codec: u8, compressed: &[u8], raw: usize) -> Result<Vec<u8>> {
        let mut output = vec![0u8; raw];
        let mut scratch = vec![0u8; raw];
        let count = decode_chunk(
            header(codec, compressed.len(), raw)?,
            compressed,
            &mut output,
            &mut scratch,
            &AtomicBool::new(false),
        )?;
        assert_eq!(count, raw);
        Ok(output)
    }
    fn frame(flags: u8, blocks: &[(bool, &[u8])], raw: &[u8]) -> Vec<u8> {
        // Test-only frame construction uses the independent locked checksum
        // primitive. Literal full-frame goldens below do not use this builder.
        let mut output = FRAME_MAGIC.to_le_bytes().to_vec();
        output.extend([flags, 0x40]);
        if flags & 8 != 0 {
            output.extend((raw.len() as u64).to_le_bytes());
        }
        output.push((xxhash_rust::xxh32::xxh32(&output[4..], 0) >> 8) as u8);
        for (uncompressed, block) in blocks {
            output.extend(
                ((block.len() as u32) | if *uncompressed { 0x80000000 } else { 0 }).to_le_bytes(),
            );
            output.extend_from_slice(block);
            if flags & 16 != 0 {
                output.extend(xxhash_rust::xxh32::xxh32(block, 0).to_le_bytes());
            }
        }
        output.extend([0; 4]);
        if flags & 4 != 0 {
            output.extend(xxhash_rust::xxh32::xxh32(raw, 0).to_le_bytes());
        }
        output
    }
    fn expected() -> Vec<u8> {
        b"abcabcabcabcabcabcEND!!".to_vec()
    }

    #[test]
    fn header_version_codec_and_exact_sizes_are_bounded() {
        let valid = header(0, 4, 4).unwrap();
        assert_eq!(valid.compressed_bytes(), 4);
        assert_eq!(valid.raw_bytes(), 4);
        assert_eq!(decode(0, b"data", 4).unwrap(), b"data");
        assert_eq!(decode(0, b"", 0).unwrap(), b"");
        assert!(header(0, 4, 3).is_err());
        assert!(header(1, MAX_COMPRESSED_BYTES + 1, 1).is_err());
        assert!(header(1, 1, MAX_RAW_BYTES + 1).is_err());
        for codec in [3, 99, 255] {
            assert!(matches!(header(codec, 0, 0), Err(Error::Unsupported(_))));
        }
        let mut version = [0u8; 8];
        version[0] = 1;
        assert!(matches!(
            ChunkHeader::parse(version),
            Err(Error::Unsupported(_))
        ));
        let mut output = [0u8; 4];
        assert!(
            decode_chunk(valid, b"bad", &mut output, &mut [], &AtomicBool::new(false)).is_err()
        );
    }
    #[test]
    fn frozen_literal_lz4_frame_and_checksum_variants_decode_exactly() {
        // Independent hand-decoded block: literal abc; length15 match at
        // distance3; final literals END!!. Frozen scalar XXH32 frame values.
        for value in [
            COMPRESSED,
            "04224d187040ad0c0000003b616263030050454e4421211c2a08aa00000000",
            "04224d186440a70c0000003b616263030050454e442121000000008cc44c0f",
            "04224d187c401700000000000000f00c0000003b616263030050454e4421211c2a08aa000000008cc44c0f",
        ] {
            assert_eq!(decode(1, &hex(value), 23).unwrap(), expected());
        }
        let all = hex(
            "04224d187c401700000000000000f00c0000003b616263030050454e4421211c2a08aa000000008cc44c0f",
        );
        for index in [6, 14, 31, all.len() - 1] {
            let mut bad = all.clone();
            bad[index] ^= 1;
            assert!(decode(1, &bad, 23).is_err(), "checksum/size offset {index}");
        }
    }
    #[test]
    fn bg4_tail_offsets_are_literal_and_cover_every_remainder() {
        let grouped: &[&[u8]] = &[
            &[],
            &[0],
            &[0, 1],
            &[0, 1, 2],
            &[0, 1, 2, 3],
            &[0, 4, 1, 2, 3],
            &[0, 4, 1, 5, 2, 3],
            &[0, 4, 1, 5, 2, 6, 3],
            &[0, 4, 1, 5, 2, 6, 3, 7],
            &[0, 4, 8, 1, 5, 2, 6, 3, 7],
        ];
        for data in grouped {
            let framed = frame(0x60, &[(true, data)], data);
            assert_eq!(
                decode(2, &framed, data.len()).unwrap(),
                (0..data.len() as u8).collect::<Vec<_>>()
            );
        }
        let framed = frame(0x60, &[(true, b"group")], b"group");
        assert!(
            decode_chunk(
                header(2, framed.len(), 5).unwrap(),
                &framed,
                &mut [0; 5],
                &mut [0; 4],
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    #[test]
    fn linked_blocks_may_refer_back_but_independent_blocks_cannot() {
        let second = b"\x0b\x03\x00\x50END!!";
        let linked = frame(0x40, &[(true, b"abc"), (false, second)], &expected());
        assert_eq!(decode(1, &linked, 23).unwrap(), expected());
        let independent = frame(0x60, &[(true, b"abc"), (false, second)], &expected());
        assert!(decode(1, &independent, 23).is_err());
    }
    #[test]
    fn every_truncation_missing_endmark_and_trailing_data_refuse() {
        let valid = hex(COMPRESSED);
        for length in 0..valid.len() {
            assert!(decode(1, &valid[..length], 23).is_err(), "prefix {length}");
        }
        for tail in [b"x".as_slice(), &FRAME_MAGIC.to_le_bytes()] {
            let mut bad = valid.clone();
            bad.extend_from_slice(tail);
            assert!(decode(1, &bad, 23).is_err());
        }
        let mut concatenated = valid.clone();
        concatenated.extend_from_slice(&valid);
        assert!(matches!(
            decode(1, &concatenated, 23),
            Err(Error::Unsupported(_))
        ));
        assert!(decode(1, &valid, 22).is_err());
        assert!(decode(1, &valid, 24).is_err());
    }
    #[test]
    fn unsupported_frame_profiles_and_reserved_flags_refuse_explicitly() {
        for magic in [0x184c2102u32, 0x184d2a50, 0x184d2a5f] {
            assert!(matches!(
                decode(1, &magic.to_le_bytes(), 0),
                Err(Error::Unsupported(_))
            ));
        }
        for (flags, bd) in [
            (0x20, 0x40),
            (0x62, 0x40),
            (0x60, 0x80),
            (0x60, 0x30),
            (0xe0, 0x40),
        ] {
            let bytes = [4, 0x22, 0x4d, 0x18, flags, bd, 0, 0, 0, 0, 0];
            assert!(decode(1, &bytes, 0).is_err());
        }
        let dict = [4, 0x22, 0x4d, 0x18, 0x61, 0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(matches!(decode(1, &dict, 0), Err(Error::Unsupported(_))));
    }
    #[test]
    fn literal_match_extension_and_block_end_corruption_refuse() {
        let cases: &[&[u8]] = &[
            &[0x00, 0, 0],                               // invalid zero distance
            &[0x00, 1, 0],                               // no preceding byte
            &[0xf0, 255, 255],                           // truncated extended literal length
            &[0x1f, b'a', 1, 0, 255, 255],               // truncated extended match length
            &[0x30, b'a', b'b', b'c', 3, 0],             // no final literal sequence
            &[0x30, b'a', b'b', b'c', 3, 0, 0x10, b'!'], // too few final literals
        ];
        for block in cases {
            assert!(decode(1, &frame(0x60, &[(false, block)], &[]), 23).is_err());
        }
        let mut extension = vec![0xf0];
        extension.extend(std::iter::repeat_n(255, 520));
        assert!(decode(1, &frame(0x60, &[(false, &extension)], &[]), MAX_RAW_BYTES).is_err());
    }
    #[test]
    fn raw_peak_scratch_and_decoded_bombs_stay_within_hard_bounds() {
        let raw = vec![b'x'; MAX_RAW_BYTES];
        let framed = frame(0x64, &[(true, &raw[..65536]), (true, &raw[65536..])], &raw);
        assert_eq!(decode(1, &framed, MAX_RAW_BYTES).unwrap(), raw);
        assert_eq!(decode(0, &raw, MAX_RAW_BYTES).unwrap(), raw);
        let mut oversized = vec![0u8; MAX_RAW_BYTES + 1];
        assert!(
            decode_chunk(
                header(0, 0, 0).unwrap(),
                &[],
                &mut oversized,
                &mut [],
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(
            decode_chunk(
                header(0, 0, 0).unwrap(),
                &[],
                &mut [],
                &mut oversized,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        // The advertised LZ4 maximum never drives output allocation. This
        // valid small block under a4MiB declaration still has exact23byte output.
        let mut large_decl = hex(COMPRESSED);
        large_decl[5] = 0x70;
        large_decl[6] = (xxhash_rust::xxh32::xxh32(&large_decl[4..6], 0) >> 8) as u8;
        assert_eq!(decode(1, &large_decl, 23).unwrap(), expected());
        let mut bomb = vec![0x1f, b'a', 1, 0];
        bomb.extend(std::iter::repeat_n(255, 514));
        bomb.extend([0, 0x50, b'E', b'N', b'D', b'!', b'!']);
        assert!(decode(1, &frame(0x60, &[(false, &bomb)], &[]), MAX_RAW_BYTES).is_err());
    }
    #[test]
    fn cancellation_precedes_any_output_for_all_codecs_and_checksum_work() {
        for codec in [0, 1, 2] {
            let compressed = if codec == 0 {
                b"data".to_vec()
            } else {
                frame(0x7c, &[(true, b"data")], b"data")
            };
            let mut output = [0xa5; 4];
            let mut scratch = [0x5a; 4];
            assert!(
                decode_chunk(
                    header(codec, compressed.len(), 4).unwrap(),
                    &compressed,
                    &mut output,
                    &mut scratch,
                    &AtomicBool::new(true)
                )
                .is_err()
            );
            assert_eq!(output, [0xa5; 4]);
            assert_eq!(scratch, [0x5a; 4]);
        }
        assert!(checksum(&[0; 4097], &AtomicBool::new(true)).is_err());
    }
}
