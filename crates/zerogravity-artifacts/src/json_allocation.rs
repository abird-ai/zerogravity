//! Borrowing structural census and allocation upper bound for one JSON document.
//!
//! The census runs the pinned `serde_json` parser with a borrowing
//! `DeserializeSeed` before any tree is built, so a caller can charge a derived
//! upper bound instead of guessing. It is an estimator: it neither authenticates
//! the bytes nor grants allocation, execution or resource authority.
//!
//! Pinned assumptions: Rust 1.95 x86_64, `serde_json` 1.0.151 (`raw_value`),
//! default 128-level recursion limit, `String` 24B, `Value` 32B, pointer 8B,
//! `BTreeMap` node capacity 11 keys/values and 12 edges. A change to any of
//! these requires revisiting [`node_upper`], [`Counts::tree_upper`] and
//! [`FAILURE_ALLOCATIONS`].
use crate::{Error, Result};
use serde::de::{DeserializeSeed, Error as DeError, MapAccess, SeqAccess, Visitor};
use serde_json::{Deserializer, Value};
use std::mem::{align_of, size_of};

/// Fixed-length diagnostics; refusal paths never format crate/serde errors.
const MSG_SCRATCH: &str = "json census scratch";
const MSG_LAYOUT: &str = "json census layout";
const MSG_ALIGN: &str = "json census alignment";
const MSG_ALIGNED: &str = "json census aligned extent";
const MSG_LEAF: &str = "json census btree leaf";
const MSG_NODE: &str = "json census btree node";
const MSG_STRINGS: &str = "json census string total";
const MSG_ELEMENTS: &str = "json census array elements";
const MSG_CAPACITY: &str = "json census array capacity";
const MSG_CAPACITY_TOTAL: &str = "json census capacity total";
const MSG_NODES: &str = "json census node count";
const MSG_OBJECTS: &str = "json census objects";
const MSG_MEMBERS: &str = "json census members";
const MSG_TREE: &str = "json census tree bound";
const MSG_SENTINEL: &str = "json census private serde key";
const MSG_MALFORMED: &str = "json census malformed input";
const MSG_TRAILING: &str = "json census trailing input";
const MSG_PEAK: &str = "json census parse peak";

/// Longest fixed diagnostic, checked below.
const FIXED_MESSAGE_BOUND: usize = 32;
const _: () = {
    assert!(MSG_SCRATCH.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_LAYOUT.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_ALIGN.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_ALIGNED.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_LEAF.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_NODE.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_STRINGS.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_ELEMENTS.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_CAPACITY.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_CAPACITY_TOTAL.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_NODES.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_OBJECTS.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_MEMBERS.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_TREE.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_SENTINEL.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_MALFORMED.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_TRAILING.len() <= FIXED_MESSAGE_BOUND);
    assert!(MSG_PEAK.len() <= FIXED_MESSAGE_BOUND);
};

/// Pinned `serde_json` 1.0.151 refusal storage, no niche assumption:
/// `ErrorCode` is an 8B tag plus its largest payload (`Box<str>`, 16B) = 24B;
/// `ErrorImpl` adds `line`/`column` (2 * 8B) = 40B. A refusal keeps one fixed
/// custom message boxed string (32B) and the final crate diagnostic (32B).
const SERDE_ERROR_IMPL: usize = 40;
const FAILURE_ALLOCATIONS: usize = SERDE_ERROR_IMPL + 2 * FIXED_MESSAGE_BOUND;

/// How the caller will construct the counted tree from
/// [`JsonAllocation::input`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueConstruction {
    /// Normal `serde_json::Value` parse. Decoded `$serde_json::private::RawValue`
    /// and `$serde_json::private::Number` keys are refused at any depth: with
    /// `raw_value` enabled `Value` would reparse an inner document that this
    /// census did not count.
    SerdeValue,
    /// A reviewed direct recursive value constructor (`zerogravity-cli`
    /// `StrictValue`). Private serde keys stay ordinary JSON keys because that
    /// constructor never classifies them. Never feed this census to
    /// `Value::deserialize`/`from_slice::<Value>`.
    DirectValueTree,
}

/// Census of one borrowed JSON document plus its derived tree bounds.
pub struct JsonAllocation<'a> {
    input: &'a [u8],
    tree_heap_upper: usize,
    parse_extra_peak: usize,
}

impl<'a> JsonAllocation<'a> {
    /// The exact immutable bytes that were censused; parse only these.
    pub fn input(&self) -> &'a [u8] {
        self.input
    }
    /// Retained tree upper bound: `(M+O)*node + 2*C*sizeof(Value) + S`, where
    /// `C` is the sum of per-array capacities (0 empty, else max(4,
    /// next_power_of_two(n))) and the factor 2 covers old+new realloc overlap.
    pub fn tree_heap_upper(&self) -> usize {
        self.tree_heap_upper
    }
    /// Tree upper bound plus the complete pre-census bound: escaped-string
    /// scratch/realloc and bounded refusal allocations.
    pub fn parse_extra_peak(&self) -> usize {
        self.parse_extra_peak
    }
}

/// Checked pre-census bound excluding the caller-owned input: successful
/// escaped-string scratch (raw span plus 3-byte unicode slack, old+new backing
/// of one growth) plus the source-derived refusal allocations in
/// [`FAILURE_ALLOCATIONS`]. Charge this before invoking the census.
pub fn census_scratch_peak(input_len: usize) -> Result<usize> {
    let scratch = input_len
        .checked_add(3)
        .and_then(|span| span.checked_mul(3))
        .and_then(|scratch| scratch.checked_add(8))
        .ok_or_else(|| Error::Limit(MSG_SCRATCH.into()))?;
    scratch
        .checked_add(FAILURE_ALLOCATIONS)
        .ok_or_else(|| Error::Limit(MSG_SCRATCH.into()))
}

/// Source-derived upper bound of one `BTreeMap<String, Value>` node: 11
/// key/value slots and 12 edges plus field/padding slack, aligned.
fn node_upper() -> Result<usize> {
    let (key, val, ptr, align) = (
        size_of::<String>(),
        size_of::<Value>(),
        size_of::<*const ()>(),
        align_of::<String>()
            .max(align_of::<Value>())
            .max(align_of::<*const ()>())
            .max(align_of::<u16>()),
    );
    if key != 24 || val != 32 || ptr != 8 || align != 8 || size_of::<usize>() != 8 {
        return Err(Error::Unsupported(MSG_LAYOUT.into()));
    }
    let leaf = ptr
        .checked_add(2 * size_of::<u16>())
        .and_then(|n| n.checked_add(11 * key))
        .and_then(|n| n.checked_add(11 * val))
        .and_then(|n| n.checked_add(5 * (align - 1)))
        .ok_or_else(|| Error::Limit(MSG_LEAF.into()))?;
    let node = align_up(leaf, align)?
        .checked_add(12 * ptr)
        .and_then(|n| n.checked_add(2 * (align - 1)))
        .ok_or_else(|| Error::Limit(MSG_NODE.into()))?;
    align_up(node, align)
}

fn align_up(value: usize, align: usize) -> Result<usize> {
    if !align.is_power_of_two() {
        return Err(Error::Unsupported(MSG_ALIGN.into()));
    }
    value
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| Error::Limit(MSG_ALIGNED.into()))
}

/// Scalar occurrence counters. Map keys are borrowed, never retained.
#[derive(Default)]
struct Counts {
    members: usize,
    objects: usize,
    array_capacity: usize,
    strings: usize,
}

impl Counts {
    fn try_add(counter: &mut usize, len: usize) -> bool {
        match counter.checked_add(len) {
            Some(next) => {
                *counter = next;
                true
            }
            None => false,
        }
    }
    fn add<E: DeError>(
        counter: &mut usize,
        len: usize,
        msg: &'static str,
    ) -> std::result::Result<(), E> {
        if Self::try_add(counter, len) {
            Ok(())
        } else {
            Err(E::custom(msg))
        }
    }
    fn tree_upper(&self) -> Result<usize> {
        let nodes = self
            .members
            .checked_add(self.objects)
            .ok_or_else(|| Error::Limit(MSG_NODES.into()))?;
        let node_bytes = nodes
            .checked_mul(node_upper()?)
            .ok_or_else(|| Error::Limit(MSG_TREE.into()))?;
        let array_bytes = self
            .array_capacity
            .checked_mul(2)
            .and_then(|n| n.checked_mul(size_of::<Value>()))
            .ok_or_else(|| Error::Limit(MSG_CAPACITY_TOTAL.into()))?;
        node_bytes
            .checked_add(array_bytes)
            .and_then(|n| n.checked_add(self.strings))
            .ok_or_else(|| Error::Limit(MSG_TREE.into()))
    }
}

/// Structural census over the exact borrowed bytes; no `Value` is built.
pub fn census_json(input: &[u8], construction: ValueConstruction) -> Result<JsonAllocation<'_>> {
    let mut counts = Counts::default();
    let mut parser = Deserializer::from_slice(input);
    match (CensusSeed {
        counts: &mut counts,
        construction,
    })
    .deserialize(&mut parser)
    {
        Ok(()) => {}
        Err(error) => {
            // Fixed diagnostics only: drop the serde error and parser (and their
            // escape scratch) before constructing the final crate message.
            drop(error);
            drop(parser);
            return Err(Error::Malformed(MSG_MALFORMED.into()));
        }
    }
    let trailing = parser.end().is_err();
    drop(parser);
    if trailing {
        return Err(Error::Malformed(MSG_TRAILING.into()));
    }
    let tree = counts.tree_upper()?;
    let peak = tree
        .checked_add(census_scratch_peak(input.len())?)
        .ok_or_else(|| Error::Limit(MSG_PEAK.into()))?;
    Ok(JsonAllocation {
        input,
        tree_heap_upper: tree,
        parse_extra_peak: peak,
    })
}

struct CensusSeed<'a> {
    counts: &'a mut Counts,
    construction: ValueConstruction,
}

impl<'a, 'de> DeserializeSeed<'de> for CensusSeed<'a> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> std::result::Result<(), D::Error> {
        de.deserialize_any(CensusVisitor {
            counts: self.counts,
            construction: self.construction,
        })
    }
}

struct CensusVisitor<'a> {
    counts: &'a mut Counts,
    construction: ValueConstruction,
}

impl<'a, 'de> Visitor<'de> for CensusVisitor<'a> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("one JSON document")
    }
    fn visit_bool<E>(self, _: bool) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_i64<E>(self, _: i64) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_u64<E>(self, _: u64) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_f64<E>(self, _: f64) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_unit<E>(self) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_none<E>(self) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_str<E: DeError>(self, value: &str) -> std::result::Result<(), E> {
        Counts::add(&mut self.counts.strings, value.len(), MSG_STRINGS)
    }
    fn visit_borrowed_str<E: DeError>(self, value: &'de str) -> std::result::Result<(), E> {
        Counts::add(&mut self.counts.strings, value.len(), MSG_STRINGS)
    }
    fn visit_string<E: DeError>(self, value: String) -> std::result::Result<(), E> {
        Counts::add(&mut self.counts.strings, value.len(), MSG_STRINGS)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<(), A::Error> {
        let mut elements = 0usize;
        while seq
            .next_element_seed(CensusSeed {
                counts: &mut *self.counts,
                construction: self.construction,
            })?
            .is_some()
        {
            elements = elements
                .checked_add(1)
                .ok_or_else(|| A::Error::custom(MSG_ELEMENTS))?;
        }
        let capacity = if elements == 0 {
            0
        } else {
            elements
                .checked_next_power_of_two()
                .ok_or_else(|| A::Error::custom(MSG_CAPACITY))?
                .max(4)
        };
        Counts::add(
            &mut self.counts.array_capacity,
            capacity,
            MSG_CAPACITY_TOTAL,
        )
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<(), A::Error> {
        Counts::add(&mut self.counts.objects, 1, MSG_OBJECTS)?;
        while map
            .next_key_seed(KeySeed {
                counts: &mut *self.counts,
                construction: self.construction,
            })?
            .is_some()
        {
            Counts::add(&mut self.counts.members, 1, MSG_MEMBERS)?;
            map.next_value_seed(CensusSeed {
                counts: &mut *self.counts,
                construction: self.construction,
            })?;
        }
        Ok(())
    }
}

struct KeySeed<'a> {
    counts: &'a mut Counts,
    construction: ValueConstruction,
}

impl<'a, 'de> DeserializeSeed<'de> for KeySeed<'a> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> std::result::Result<(), D::Error> {
        de.deserialize_str(KeyVisitor {
            counts: self.counts,
            construction: self.construction,
        })
    }
}

struct KeyVisitor<'a> {
    counts: &'a mut Counts,
    construction: ValueConstruction,
}

impl KeyVisitor<'_> {
    fn key<E: DeError>(
        counts: &mut Counts,
        construction: ValueConstruction,
        value: &str,
    ) -> std::result::Result<(), E> {
        if construction == ValueConstruction::SerdeValue
            && (value == "$serde_json::private::RawValue"
                || value == "$serde_json::private::Number")
        {
            return Err(E::custom(MSG_SENTINEL));
        }
        Counts::add(&mut counts.strings, value.len(), MSG_STRINGS)
    }
}

impl<'a, 'de> Visitor<'de> for KeyVisitor<'a> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("one JSON object key")
    }
    fn visit_str<E: DeError>(self, value: &str) -> std::result::Result<(), E> {
        Self::key(self.counts, self.construction, value)
    }
    fn visit_borrowed_str<E: DeError>(self, value: &'de str) -> std::result::Result<(), E> {
        Self::key(self.counts, self.construction, value)
    }
    fn visit_string<E: DeError>(self, value: String) -> std::result::Result<(), E> {
        Self::key(self.counts, self.construction, &value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static TRACK: Cell<bool> = const { Cell::new(false) };
        static LIVE: Cell<isize> = const { Cell::new(0) };
        static PEAK: Cell<usize> = const { Cell::new(0) };
    }
    struct Counting;
    impl Counting {
        fn note(delta: isize) {
            if !TRACK.with(Cell::get) {
                return;
            }
            LIVE.with(|live| {
                let next = live.get() + delta;
                live.set(next);
                if next > 0 {
                    PEAK.with(|peak| {
                        if next as usize > peak.get() {
                            peak.set(next as usize);
                        }
                    });
                }
            });
        }
    }
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                Self::note(layout.size() as isize);
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            Self::note(-(layout.size() as isize));
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let next = unsafe { System.realloc(ptr, layout, new_size) };
            if !next.is_null() {
                Self::note(new_size as isize - layout.size() as isize);
            }
            next
        }
    }
    #[global_allocator]
    static ALLOC: Counting = Counting;

    fn measure<T>(f: impl FnOnce() -> T) -> (T, usize) {
        LIVE.with(|live| live.set(0));
        PEAK.with(|peak| peak.set(0));
        TRACK.with(|track| track.set(true));
        let out = f();
        TRACK.with(|track| track.set(false));
        (out, PEAK.with(Cell::get))
    }

    fn tree(doc: &[u8]) -> usize {
        census_json(doc, ValueConstruction::SerdeValue)
            .unwrap()
            .tree_heap_upper()
    }

    #[test]
    fn counts_nested_structures_and_is_parse_equivalent() {
        let doc = br#"{"a":{"b":[1,2,3,{"c":"d"}]},"a":"\u00e9"}"#;
        let allocation = census_json(doc, ValueConstruction::SerdeValue).unwrap();
        assert_eq!(allocation.input(), doc);
        assert_eq!(
            allocation.parse_extra_peak(),
            allocation.tree_heap_upper() + census_scratch_peak(doc.len()).unwrap()
        );
        assert!(serde_json::from_slice::<Value>(doc).is_ok());
        let parsed: Value = serde_json::from_slice(br#"{"a":{"b":[1,2,3]}}"#).unwrap();
        assert_eq!(parsed["a"]["b"].as_array().unwrap().capacity(), 4);
        let empty = census_json(b"{\"a\":[],\"b\":{}}", ValueConstruction::SerdeValue).unwrap();
        assert_eq!(
            empty.tree_heap_upper(),
            4 * 776 + 2,
            "two objects, two keys"
        );
        assert!(empty.tree_heap_upper() < allocation.tree_heap_upper());
        assert_eq!(census_scratch_peak(0).unwrap(), 17 + FAILURE_ALLOCATIONS);
        assert!(census_scratch_peak(usize::MAX).is_err());
    }

    #[test]
    fn array_capacity_uses_minimum_growth_and_old_new_realloc() {
        for (elements, capacity) in [
            (0usize, 0usize),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 8),
            (8, 8),
            (9, 16),
        ] {
            let doc = format!(
                "[{}]",
                (0..elements)
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let parsed: Value = serde_json::from_slice(doc.as_bytes()).unwrap();
            assert_eq!(
                parsed.as_array().unwrap().capacity(),
                capacity,
                "{elements}"
            );
            assert_eq!(
                tree(doc.as_bytes()),
                2 * capacity * size_of::<Value>(),
                "{elements}"
            );
        }
        let nested = b"[[[[0]]]]";
        let parsed: Value = serde_json::from_slice(nested).unwrap();
        let mut value = &parsed;
        let mut capacity = 0;
        while let Some(next) = value.as_array().and_then(|a| a.first()) {
            capacity += value.as_array().unwrap().capacity();
            value = next;
        }
        assert_eq!(capacity, 4 * 4);
        assert_eq!(tree(nested), 2 * capacity * size_of::<Value>());
    }

    #[test]
    fn duplicate_occurrences_are_charged_and_overflow_is_refused() {
        let one = tree(br#"{"a":1}"#);
        let two = tree(br#"{"a":1,"a":2}"#);
        assert_eq!(two - one, 776 + 1);
        assert!(census_json(b"{}x", ValueConstruction::SerdeValue).is_err());
        assert!(census_json(b"{\"a\":1,}", ValueConstruction::SerdeValue).is_err());
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        assert!(census_json(deep.as_bytes(), ValueConstruction::SerdeValue).is_err());
        let mut max = usize::MAX;
        assert!(!Counts::try_add(&mut max, 1));
        for counts in [
            Counts {
                members: usize::MAX,
                objects: 1,
                ..Counts::default()
            },
            Counts {
                array_capacity: usize::MAX,
                ..Counts::default()
            },
            Counts {
                members: 1,
                strings: usize::MAX,
                ..Counts::default()
            },
        ] {
            assert!(counts.tree_upper().is_err());
        }
    }

    #[test]
    fn private_sentinels_are_refused_only_for_serde_value() {
        for doc in [
            br#"{"$serde_json::private::RawValue":"{}"}"#.as_slice(),
            br#"{"$serde_json::private::Number":1}"#.as_slice(),
            br#"[{"x":{"$serde_json::private::RawValue":"1"}}]"#.as_slice(),
            br#"{"$serde_json::private\u003a\u003aRawValue":"{}"}"#.as_slice(),
        ] {
            assert!(census_json(doc, ValueConstruction::SerdeValue).is_err());
            assert!(census_json(doc, ValueConstruction::DirectValueTree).is_ok());
        }
        assert!(
            census_json(
                br#"{"a":"$serde_json::private::RawValue"}"#,
                ValueConstruction::SerdeValue
            )
            .is_ok()
        );
        for doc in [
            br#""\q""#.as_slice(),
            br#""\ud800""#.as_slice(),
            br#"{"a":1e999}"#.as_slice(),
        ] {
            assert!(census_json(doc, ValueConstruction::DirectValueTree).is_err());
        }
    }

    #[test]
    fn census_and_refusals_stay_under_the_declared_prebound() {
        let escaped = format!("\"{}\"", "\\u00e9".repeat(200));
        let late = format!("{{\"a\":\"{}\",}}", "\\u00e9".repeat(200));
        let sentinel = br#"{"$serde_json::private::Number":1}"#;
        for doc in [
            b"".as_slice(),
            b"{\"a\":1".as_slice(),
            sentinel.as_slice(),
            late.as_bytes(),
        ] {
            let (out, peak) = measure(|| census_json(doc, ValueConstruction::SerdeValue));
            assert!(out.is_err(), "{doc:?}");
            let bound = census_scratch_peak(doc.len()).unwrap();
            assert!(peak <= bound, "{peak} > {bound} for {doc:?}");
        }
        let (out, peak) =
            measure(|| census_json(escaped.as_bytes(), ValueConstruction::SerdeValue));
        assert!(out.is_ok());
        assert!(peak > 0);
        assert!(peak <= census_scratch_peak(escaped.len()).unwrap());
        let (out, peak) = measure(|| census_json(b"[0,{\"a\":[]}]", ValueConstruction::SerdeValue));
        assert!(out.is_ok());
        assert_eq!(peak, 0, "structural census allocates nothing");
    }
}
