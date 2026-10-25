//! Source-derived storage bounds for the approved tokenizer profile.
//!
//! Every row is a typed equation over the pinned crates' own layout rules and
//! the compiled literal descriptor, not a fitted coefficient:
//!
//! * `tokenizers 0.22.2` collections (`RawVec` minimum capacities, `Vec`
//!   doubling, `HashMap` buckets/control bytes, `Encoding` sidecars, BPE
//!   `Word`/`Merge`/`Symbol`, `AddedVocabulary` Aho-Corasick automata).
//! * `regex-automata 0.4.18` engines that this profile can actually reach:
//!   immutable dense/hybrid DFA tables, the four hybrid delegates' fallback
//!   PikeVM/bounded-backtracker heaps, hybrid state caches, program storage.
//!
//! Delegate geometry (NFA state, union-surplus and capture-state counts, byte
//! classes, engine presence, hybrid stride) is the compiled literal descriptor
//! produced from `Regex::debug_print()` of the exact approved expressions; it
//! is transcribed here as constants so production never parses a debug dump.
//!
//! `P` is the pool-stack count of the pinned `meta::Config::get_pool_capacity`
//! (the same cached value the production `meta::Regex` reads), never a fresh
//! parallelism sample and never a caller affinity choice.

use std::mem::size_of;

use regex_automata::meta;

use crate::{Error, Result};
use regex_automata::util::primitives::{PatternID, StateID};
use zerogravity_artifacts::json_allocation::census_scratch_peak;

/// Pinned approved source facts. `F` is the exact `tokenizer.json` length; the
/// decoded string masses below are bounded by it because a JSON string value
/// cannot be longer than the document that encodes it.
pub const APPROVED_TOKENIZER_LEN: u64 = 12_807_982;
pub const VOCAB_ENTRIES: u64 = 248_044;
pub const MERGE_ENTRIES: u64 = 247_587;
pub const ADDED_TOKENS: u64 = 26;
/// `AddPrefix` of the BPE model in the approved source is `""` and the
/// added-token content is at most 20 bytes (approved metadata); longest
/// vocabulary entry is 256 bytes.
pub const MAX_VOCAB_ENTRY_BYTES: u64 = 256;
pub const MAX_ADDED_ENTRY_BYTES: u64 = 20;
/// `BPE::default_cache` capacity (`models/bpe/model.rs`: `with_capacity(10_000)`).
pub const BPE_DEFAULT_CACHE: u64 = 10_000;
/// `BPE` merge scratch string (`String::with_capacity(512)`) and the
/// maximum single merge key (`max_token_len = 255`).
pub const BPE_MERGE_SCRATCH: u64 = 512;
pub const BPE_MAX_KEY: u64 = 255;
/// `AhoCorasick` alphabet width of the added-token automata.
const ALPHABET: u64 = 256;
const AHO_MAX_RK_BUCKETS: u64 = 64;
const AHO_MAX_TEDDY_BUCKETS: u64 = 16;

// ---------------------------------------------------------------------------
// Pinned type layouts.
// ---------------------------------------------------------------------------

const PTR: u64 = size_of::<usize>() as u64;
const U32: u64 = size_of::<u32>() as u64;
const U64: u64 = size_of::<u64>() as u64;
const U8_: u64 = size_of::<u8>() as u64;
const STRING: u64 = size_of::<String>() as u64;
const VEC: u64 = size_of::<Vec<u8>>() as u64;
/// `(usize, usize)`.
const PAIR: u64 = 16;
const OPTION_U32: u64 = size_of::<Option<u32>>() as u64;
/// `Option<NonMaxUsize>` shares the `NonZeroUsize` niche layout.
const SLOT: u64 = size_of::<Option<core::num::NonZeroUsize>>() as u64;

// Public pinned-crate types: asserts bind the equations to the crate actually
// compiled, so a dependency change fails the build instead of the ledger.
const STATE: u64 = size_of::<regex_automata::nfa::thompson::State>() as u64;
const STATE_ID: u64 = size_of::<StateID>() as u64;
const PATTERN_ID: u64 = size_of::<PatternID>() as u64;
const META_CACHE: u64 = size_of::<meta::Cache>() as u64;
/// Inline `inner::Pool::owner_val: UnsafeCell<Option<Cache>>` slot. The
/// optional discriminant is included; the owner cache heap is charged only
/// when the pool has actually been searched.
const POOL_OWNER_SLOT: u64 = size_of::<Option<meta::Cache>>() as u64;

const _: () = {
    assert!(PTR == 8);
    assert!(STRING == 24);
    assert!(VEC == 24);
    assert!(STATE == 24);
    assert!(STATE_ID == 4);
    assert!(PATTERN_ID == 4);
    assert!(SLOT == 8);
    assert!(OPTION_U32 == 8);
    // `meta::Cache` holds one inline cache object per engine; the pinned crate
    // reports 1400 B on x86_64. A different value invalidates the rows below.
    assert!(META_CACHE == 1400);
};

/// Private pinned-crate layouts. Each mirror repeats the exact field order and
/// types of the pinned definition, so `size_of` of the mirror *is* the layout
/// claim; no magic constant is trusted on its own.
#[repr(C)]
#[allow(dead_code)]
struct NormalizedStringMirror {
    original: String,
    normalized: String,
    alignments: Vec<(usize, usize)>,
    original_shift: usize,
}
/// `tokenizers::NormalizedString` (two strings, one vector, one offset).
const NORMALIZED_STRING: u64 = size_of::<NormalizedStringMirror>() as u64;

#[repr(C)]
#[allow(dead_code)]
struct TokenMirror {
    id: u32,
    value: String,
    offsets: (usize, usize),
}
/// `tokenizers::Token` (`id`, `value`, `offsets`).
const TOKEN: u64 = size_of::<TokenMirror>() as u64;

#[repr(C)]
#[allow(dead_code)]
struct SplitMirror {
    normalized: NormalizedStringMirror,
    tokens: Option<Vec<TokenMirror>>,
}
/// `tokenizers::tokenizer::Split`: one inline `NormalizedString` plus its
/// optional token vector.
const SPLIT: u64 = size_of::<SplitMirror>() as u64;

#[repr(C)]
#[allow(dead_code)]
struct EncodingMirror {
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    tokens: Vec<String>,
    words: Vec<Option<u32>>,
    offsets: Vec<(usize, usize)>,
    special_tokens_mask: Vec<u32>,
    attention_mask: Vec<u32>,
    sequence_ranges: std::collections::HashMap<usize, (usize, usize)>,
    overflowing: Vec<Vec<u32>>,
}
/// `tokenizers::Encoding`: seven typed sidecar vectors, the sequence-range map
/// and the overflow list. Each field is charged separately.
const ENCODING: u64 = size_of::<EncodingMirror>() as u64;

#[repr(C)]
#[allow(dead_code)]
struct SymbolMirror {
    c: u32,
    prev: isize,
    next: isize,
    len: usize,
}
/// `tokenizers::models::bpe::word::Symbol` (`u32`, two `isize`, `usize`).
const SYMBOL: u64 = size_of::<SymbolMirror>() as u64;

#[repr(C)]
#[allow(dead_code)]
struct MergeMirror {
    pos: usize,
    rank: u32,
    new_id: u32,
}
/// `tokenizers::models::bpe::word::Merge` (`usize`, two `u32`).
const MERGE: u64 = size_of::<MergeMirror>() as u64;

#[repr(C)]
#[allow(dead_code)]
struct ContentMirror {
    tag: u8,
    payload: String,
}
#[repr(C)]
#[allow(dead_code)]
struct ContentVectorMirror {
    tag: u8,
    payload: Vec<ContentMirror>,
}
/// `serde_core::private::Content`: largest variant is `Map(Vec<(Content,
/// Content)>)`, whose `Vec` payload is 24 B behind an 8-byte tag. A map pair
/// is therefore 2 * 32 B, not `String + Content`.
const CONTENT: u64 = size_of::<ContentVectorMirror>() as u64;
const CONTENT_PAIR: u64 = CONTENT.saturating_mul(2);
/// A typed `(String, String)` merge pair in `MergeType::Tuple`.
const MERGE_TYPED: u64 = STRING * 2;
/// The legacy `Vec<String>` merge alternative in the untagged enum.
const MERGE_LEGACY: u64 = 24;

/// `(Offsets, bool)` per split fragment in `Encoding`.
const OFFSETS_BOOL: u64 = size_of::<((usize, usize), bool)>() as u64;
/// `(char, isize)` symbol pairs in the BPE builder.
const CHAR_ISIZE: u64 = size_of::<(char, isize)>() as u64;
#[repr(C)]
#[allow(dead_code)]
struct AddedTokenMirror {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}
const ADDED_TOKEN: u64 = size_of::<AddedTokenMirror>() as u64;

/// `serde_json::Value` is 32 B; a `BTreeMap` node has 11 key/value slots and
/// 12 edges plus its leaf header and padding. The formula mirrors
/// `json_allocation::node_upper`: leaf = 8 + 2*2 + 11*24 + 11*32 + 5*7,
/// aligned to 8; node = leaf + 12*8 + 14, aligned to 8.
const VALUE: u64 = 32;
const BTREE_NODE: u64 = 776;
/// Unaligned leaf row from `json_allocation::node_upper`'s slot/edge terms.
const BTREE_LEAF_RAW: u64 = 8 + 2 * 2 + 11 * 24 + 11 * 32 + 5 * 7 + 7;
const BTREE_LEAF: u64 = BTREE_LEAF_RAW / 8 * 8;
const BTREE_NODE_RAW: u64 = 774;
const _: () = {
    assert!(CONTENT_PAIR == 64);
    assert!(VALUE == 32);
    assert!(BTREE_LEAF + 12 * 8 + 14 == BTREE_NODE_RAW);
    assert!(BTREE_NODE_RAW.div_ceil(8) * 8 == BTREE_NODE);
};

/// `regex_automata::nfa::thompson::backtrack` `Frame`:
/// `Step { sid, at }` or `RestoreCapture { slot, offset }`; tag + 16 B.
const FRAME: u64 = 24;
/// `pikevm::FollowEpsilon`: `Change{slot,offset}` / `RestoreCapture{slot}` /
/// `RestoreCaptureFrom{slot,offset}`; tag + capture slot + offset.
const FOLLOW_EPSILON: u64 = 24;
/// `thompson::Transition`: `start: u8`, `end: u8`, `next: StateID`.
const TRANSITION: u64 = 8;
/// `hybrid::LazyStateID` (`u32`).
const LAZY_STATE_ID: u64 = 4;
/// `hybrid::State` header: `Arc<[u8]>` fat pointer.
const HYBRID_STATE_HEADER: u64 = 16;
/// `hybrid::Cache` state-map entry: `(State, LazyStateID)` aligned to 8 B.
const HYBRID_MAP_ENTRY: u64 = 24;
/// `hybrid::Cache` payload per cached state: `13 + 4 * patterns + 5 * NFA
/// states` (transitions are premultiplied by the stride).
const HYBRID_STATE_FIXED: u64 = 13;
/// `fancy_regex::vm::Branch` (`usize` pc pair + `isize` next) and `Save`.
const BRANCH: u64 = 24;
const SAVE: u64 = 16;
/// `fancy_regex::vm::MAX_STACK` (`1_000_000`).
const VM_STACK: u64 = 1_000_000;

/// `BoundedBacktracker` visited set: configured capacity is *heap bytes*; the
/// bitset is `8 * capacity` bits, blocks of `8 * size_of::<usize>()` bits.
const VISITED_CAPACITY_BYTES: u64 = 256 * (1 << 10);
const VISITED_BLOCK_BITS: u64 = 8 * PTR;
/// `Start::len() * 2` for one pattern with per-pattern starts enabled.
const HYBRID_START_SLOTS: u64 = 18;
/// Hybrid DFA byte-class stride (`1 << 7`) in the pinned descriptor.
#[cfg(test)]
const HYBRID_STRIDE: u64 = 1 << 7;
/// Hybrid DFA cache capacity (`meta::Config::get_hybrid_cache_capacity`).
const HYBRID_CACHE_CAPACITY: u64 = 2_097_152;
/// `Instruction` enum: widest variant is `Repeat([usize; 4])` behind a tag.
const INSTRUCTION: u64 = 40;
/// One `meta::Regex` delegate: `Arc`/info/prefilter/engine wrappers.
const DELEGATE_WRAPPERS: u64 = 512;
/// `inner::Pool` fields excluding its inline owner slot: boxed `create`
/// closure, `stacks` vector header and owner `AtomicUsize`.
const POOL_HEADER: u64 = 48;
/// One `CacheLine<Mutex<Vec<Box<T>>>>` stack header: the inner
/// `Mutex<Vec<Box<T>>>` is 24 B, padded by `repr(align(64))`.
const POOL_STACK: u64 = 64;
/// NFA per-state transition upper bound: at most 256 distinct byte ranges.
const NFA_TRANSITIONS_PER_STATE: u64 = 256;
/// One boxed slice/`Box<[T]>` header.
const BOX_SLICE: u64 = PTR.saturating_mul(2);

// ---------------------------------------------------------------------------
// Collection primitives (pinned `RawVec`/`Vec`/`HashMap` rules).
// ---------------------------------------------------------------------------

/// `RawVec::MIN_NON_ZERO_CAP`: 8 elements for 1-byte, 4 for <= 1024 B.
const fn min_elements(size: u64) -> u64 {
    if size == 1 {
        8
    } else if size <= 1024 {
        4
    } else {
        1
    }
}

/// Retained capacity of one vector grown to `n` elements: doubling keeps a
/// power-of-two capacity of at least `2n`, floored by the minimum non-zero
/// capacity of the element type (`max(size, align)` bytes).
fn reserve(size: u64, n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let doubling = 2u128 * n as u128 * size as u128;
    let minimum = min_elements(size) as u128 * size as u128;
    u64::try_from(doubling.max(minimum)).unwrap_or(u64::MAX)
}

/// Old and new backing of one growth overlap: `2 * reserve`.
fn regrow(size: u64, n: u64) -> u64 {
    reserve(size, n).saturating_mul(2)
}

/// `k` separate vectors of total length `len`: each pays its own doubling plus
/// its own minimum capacity.
fn fragments(size: u64, len: u64, k: u64) -> u64 {
    let elements = 2u128 * len as u128 * size as u128;
    let minimum = k as u128 * min_elements(size) as u128 * size as u128;
    u64::try_from(elements + minimum).unwrap_or(u64::MAX)
}

/// Exactly reserved vector (`Vec::with_capacity`/`reserve_exact`).
fn exact(size: u64, n: u64) -> u64 {
    size.saturating_mul(n)
}

const fn align_up(value: u64, align: u64) -> u64 {
    let align = if align == 0 { 1 } else { align };
    value.div_ceil(align) * align
}

fn next_pow2(value: u128) -> u128 {
    let mut buckets = 1u128;
    while buckets < value {
        buckets *= 2;
    }
    buckets
}

/// One `HashMap`/`AHashMap` table capacity for `n` live entries: hashbrown
/// rounds the 7/8-load bucket count up to a power of two and never allocates
/// fewer than four buckets, with one control byte per bucket.
fn buckets_of(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let wanted = (n as u128 * 8).div_ceil(7);
    let buckets = next_pow2(wanted).max(4);
    u64::try_from(buckets).unwrap_or(u64::MAX)
}

/// One hash table: bucket nodes of `entry` bytes, control bytes and the group
/// header. `entry` must already be padded to the node alignment.
fn table(entry: u64, n: u64) -> u64 {
    let buckets = buckets_of(n);
    if buckets == 0 {
        return 0;
    }
    buckets
        .saturating_mul(entry)
        .saturating_add(buckets)
        .saturating_add(16)
}

// ---------------------------------------------------------------------------
// Checked constructor arithmetic (u128 intermediate, u64 result).
// ---------------------------------------------------------------------------

fn arithmetic() -> Error {
    Error::Limit("tokenizer constructor arithmetic overflow".into())
}

fn cadd(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(arithmetic)
}

fn cmul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(arithmetic)
}

fn csum(parts: &[u64]) -> Result<u64> {
    parts
        .iter()
        .try_fold(0u64, |total, &part| cadd(total, part))
}

fn checked_align_up(value: u64, align: u64) -> Result<u64> {
    let align = if align == 0 { 1 } else { align };
    let remainder = value % align;
    if remainder == 0 {
        Ok(value)
    } else {
        cadd(value, align - remainder)
    }
}

/// `E_t(n)=t*n` (exact initialized vector).
fn exact_bytes(size: u64, n: u64) -> Result<u64> {
    cmul(size, n)
}

/// `R_t(n)`: one retained RawVec backing with doubling and `min_t` floor.
fn reserve_bytes(size: u64, n: u64) -> Result<u64> {
    if n == 0 {
        return Ok(0);
    }
    let elements = (n as u128)
        .checked_mul(2)
        .ok_or_else(arithmetic)?
        .max(min_elements(size) as u128);
    let bytes = elements.checked_mul(size as u128).ok_or_else(arithmetic)?;
    u64::try_from(bytes).map_err(|_| arithmetic())
}

/// `G_t(n)=2*R_t(n)`: old and new backing overlap.
fn checked_regrow(size: u64, n: u64) -> Result<u64> {
    cmul(2, reserve_bytes(size, n)?)
}

/// `F_t(total,k)=t*(2*total + k*min_t)`: `k` independent nonempty fragments.
fn fragment_bytes(size: u64, total: u64, k: u64) -> Result<u64> {
    let elements = (total as u128)
        .checked_mul(2)
        .and_then(|n| n.checked_add((k as u128).checked_mul(min_elements(size) as u128)?))
        .ok_or_else(arithmetic)?;
    let bytes = elements.checked_mul(size as u128).ok_or_else(arithmetic)?;
    u64::try_from(bytes).map_err(|_| arithmetic())
}

/// `FG_t(total,k)=2*F_t(total,k)`.
fn fragment_regrow_bytes(size: u64, total: u64, k: u64) -> Result<u64> {
    cmul(2, fragment_bytes(size, total, k)?)
}

/// Checked version of [`table`].
fn checked_table(entry: u64, n: u64) -> Result<u64> {
    let buckets = buckets_of(n);
    if buckets == 0 {
        return Ok(0);
    }
    let nodes = cmul(buckets, entry)?;
    cadd(cadd(nodes, buckets)?, 16)
}

/// Hash table whose `(key,value)` node is padded to the 8-byte pair alignment
/// (a `(String, u32)` node is 32 B, not 28 B).
fn checked_hash_of(key: u64, value: u64, n: u64) -> Result<u64> {
    checked_table(checked_align_up(cadd(key, value)?, 8)?, n)
}

/// Deliberately loose BTree node bound: `(2*m+1)*align8(16+11*(k+v)+12*8+32)`.
fn checked_btree(key: u64, value: u64, members: u64) -> Result<u64> {
    let payload = cadd(cadd(16, cmul(11, cadd(key, value)?)?)?, cadd(12 * 8, 32)?)?;
    let node = checked_align_up(payload, 8)?;
    cmul(cadd(cmul(2, members)?, 1)?, node)
}

/// `Arc_a(x)=align_a(align_a(16)+x)`.
fn checked_arc(align: u64, payload: u64) -> Result<u64> {
    let align = if align == 0 { 1 } else { align };
    checked_align_up(cadd(checked_align_up(16, align)?, payload)?, align)
}

fn checked_census_scratch(source_len: u64) -> Result<u64> {
    let len = usize::try_from(source_len).map_err(|_| arithmetic())?;
    u64::try_from(census_scratch_peak(len)?).map_err(|_| arithmetic())
}

// ---------------------------------------------------------------------------
// Compiled delegate descriptor.
// ---------------------------------------------------------------------------

/// Per-direction Thompson NFA geometry of one delegate: state count, union
/// edges after the first alternative and capture-state count.
#[derive(Clone, Copy)]
struct Nfa {
    states: u64,
    surplus: u64,
    capture_states: u64,
}

/// One compiled delegate of an approved literal expression.
#[derive(Clone, Copy)]
struct Delegate {
    /// Instruction-program offset (descriptor cross-reference only).
    #[allow(dead_code)]
    insn: u64,
    /// A `core` delegate owns its engines; `pre` never runs a search and only
    /// contributes its prefilter.
    core: bool,
    forward: Nfa,
    reverse: Nfa,
    /// `byte_classes.alphabet_len()` of this delegate.
    classes: u64,
    /// Dense DFA state lengths (forward, reverse), `0` when not dense.
    dense: (u64, u64),
    /// Hybrid DFA stride2 (`0` when the delegate is not hybrid).
    stride2: u64,
}

const NO_NFA: Nfa = Nfa {
    states: 0,
    surplus: 0,
    capture_states: 0,
};

macro_rules! nfa {
    ($s:expr, $u:expr, $c:expr) => {
        Nfa {
            states: $s,
            surplus: $u,
            capture_states: $c,
        }
    };
}
macro_rules! delegate {
    ($insn:expr, $fwd:expr, $rev:expr, $classes:expr, $dense:expr, $stride2:expr) => {
        Delegate {
            insn: $insn,
            core: true,
            forward: $fwd,
            reverse: $rev,
            classes: $classes,
            dense: $dense,
            stride2: $stride2,
        }
    };
}
macro_rules! prefilter_only {
    ($insn:expr) => {
        Delegate {
            insn: $insn,
            core: false,
            forward: NO_NFA,
            reverse: NO_NFA,
            classes: 0,
            dense: (0, 0),
            stride2: 0,
        }
    };
}

/// `APPROVED_SPLIT_REGEX` (122 B, sha256 `00431aed..e9ba`): 17 delegates
/// (15 core, 2 prefilter-only), 61 instructions, forward NFA 1191 states /
/// reverse 4608, 116 max byte classes, 11 dense and 4 hybrid delegates.
///
/// Every delegate has `groups=[1,1]`, `captures_len=1` and no Unicode
/// word-boundary assertion; the dense DFA's default quit set therefore stays
/// empty, so the eleven dense delegates' anchored searches cannot reach a
/// PikeVM or bounded-backtracker heap. Only the four hybrid delegates own
/// fallback heaps.
#[rustfmt::skip]
const SPLIT_DELEGATES: [Delegate; 17] = [
    delegate!( 6, nfa!(  8,1,2), nfa!(   9,3,0),  12, ( 11,  7), 0),
    delegate!( 9, nfa!(  7,1,2), nfa!(   5,1,0),   8, (  9,  6), 0),
    delegate!(12, nfa!(  8,1,2), nfa!(   6,1,0),  12, ( 11,  7), 0),
    delegate!(15, nfa!(  8,1,2), nfa!(   6,1,0),  12, ( 11,  7), 0),
    delegate!(18, nfa!(  7,1,2), nfa!(   5,1,0),   8, (  9,  6), 0),
    delegate!(21, nfa!(  8,1,2), nfa!(   6,1,0),   8, ( 11,  7), 0),
    delegate!(23, nfa!(  7,1,2), nfa!(   5,1,0),   8, (  9,  6), 0),
    delegate!(27, nfa!(336,1,2), nfa!(1380,932,0), 115, (0, 0), 7),
    delegate!(28, nfa!(311,1,2), nfa!(1369,936,0), 109, (0, 0), 7),
    delegate!(32, nfa!( 97,1,2), nfa!( 288,149,0),  86, (0, 0), 7),
    delegate!(37, nfa!(335,1,2), nfa!(1404,950,0), 116, (0, 0), 7),
    prefilter_only!(40),
    delegate!(45, nfa!( 13,1,2), nfa!(  22,10,0),  27, ( 21, 13), 0),
    prefilter_only!(47),
    delegate!(51, nfa!( 13,1,2), nfa!(  22,10,0),  27, ( 21, 13), 0),
    delegate!(54, nfa!( 20,1,2), nfa!(  59,25,0),  36, ( 35, 18), 0),
    delegate!(57, nfa!( 13,1,2), nfa!(  22,10,0),  27, ( 21, 13), 0),
];

/// ByteLevel default `RE` (74 B, sha256 `eeb55ba7..1c36`): 6 core delegates,
/// 54 instructions, forward 783 / reverse 3009 states, 3 dense and 3 hybrid.
///
/// `byte_level.rs:122` initializes this `Lazy` regex even with
/// `use_regex=false`, but the `false` branch never searches it and the decoder
/// uses `CHAR_BYTES`. The compiled program is therefore a *process-global*
/// owner once, with uninitialized pool cache slots; no per-tokenizer search
/// cache exists for this profile.
#[rustfmt::skip]
const BYTE_LEVEL_DELEGATES: [Delegate; 6] = [
    delegate!(28, nfa!(303,1,2), nfa!(1240,827,0), 107, (0, 0), 7),
    delegate!(34, nfa!( 97,1,2), nfa!( 288,149,0),  86, (0, 0), 7),
    delegate!(40, nfa!(337,1,2), nfa!(1378,936,0), 115, (0, 0), 7),
    delegate!(44, nfa!( 13,1,2), nfa!(  22, 10,0),  27, ( 21, 13), 0),
    delegate!(47, nfa!( 20,1,2), nfa!(  59, 25,0),  36, ( 35, 18), 0),
    delegate!(50, nfa!( 13,1,2), nfa!(  22, 10,0),  27, ( 21, 13), 0),
];

const SPLIT_INSTRUCTIONS: u64 = 61;
const SPLIT_LITERAL_BYTES: u64 = 122;
const SPLIT_LITERAL_STRING_PAYLOAD: u64 = 1;
const BYTE_LEVEL_INSTRUCTIONS: u64 = 54;
const BYTE_LEVEL_LITERAL_BYTES: u64 = 74;
const BYTE_LEVEL_LITERAL_STRING_PAYLOAD: u64 = 20;
/// `captures_len()` of both literals: two save slots, one group.
const CAPTURE_SLOTS: u64 = 2;
const PATTERNS: u64 = 1;

/// Instruction program: the enum vector (widest variant `Repeat`), the string
/// literal payloads and the top-level `Arc` header.
fn instruction_program(count: u64, literal: u64, payload: u64) -> u64 {
    reserve(INSTRUCTION, count)
        .saturating_add(literal)
        .saturating_add(payload)
        .saturating_add(24)
}

// ---------------------------------------------------------------------------
// Regex program (per literal, shared by every pool stack).
// ---------------------------------------------------------------------------

/// Thompson NFA: state vector, per-state sparse transition backing, union
/// edges and group metadata, plus the `Arc` header.
fn nfa_bytes(nfa: Nfa) -> u64 {
    if nfa.states == 0 {
        return 0;
    }
    reserve(STATE, nfa.states)
        .saturating_add(fragments(
            TRANSITION,
            nfa.states.saturating_mul(NFA_TRANSITIONS_PER_STATE),
            nfa.states,
        ))
        .saturating_add(reserve(STATE_ID, nfa.surplus.max(nfa.states)))
        .saturating_add(exact(PTR, 2).saturating_mul(2))
        .saturating_add(64)
}

/// Immutable dense DFA tables: premultiplied `u32` transition rows (padded to
/// the power-of-two stride), start table, byte-class map.
fn dense_bytes(classes: u64, state_length: u64) -> u64 {
    if state_length == 0 {
        return 0;
    }
    let stride = next_pow2(classes as u128 + 1);
    let transitions = u64::try_from(stride * state_length as u128).unwrap_or(u64::MAX);
    exact(U32, transitions)
        .saturating_add(exact(STATE_ID, HYBRID_START_SLOTS))
        .saturating_add(exact(U8_, classes.saturating_add(1)))
}

/// Hybrid DFA program: fixed object plus alphabet/class table. Determinized
/// state tables live in the cache, not here.
fn hybrid_program_bytes(classes: u64) -> u64 {
    const HYBRID_DFA: u64 = 720;
    HYBRID_DFA
        .saturating_add(exact(U8_, classes.saturating_add(1)))
        .saturating_add(reserve(U8_, classes.saturating_add(1)))
}

fn delegate_program(delegate: &Delegate) -> u64 {
    if !delegate.core {
        // Prefilter-only delegate: needle plus its own wrapper. Pool headers
        // and the eager owner cache are priced by `pool_bytes`/`idle_pool_bytes`.
        return DELEGATE_WRAPPERS.saturating_add(64);
    }
    let mut bytes = DELEGATE_WRAPPERS
        .saturating_add(nfa_bytes(delegate.forward))
        .saturating_add(nfa_bytes(delegate.reverse));
    if delegate.dense.0 != 0 {
        bytes = bytes
            .saturating_add(dense_bytes(delegate.classes, delegate.dense.0))
            .saturating_add(dense_bytes(delegate.classes, delegate.dense.1));
    }
    if delegate.stride2 != 0 {
        bytes = bytes
            .saturating_add(hybrid_program_bytes(delegate.classes))
            .saturating_add(hybrid_program_bytes(delegate.classes));
    }
    bytes
}

fn literal_program(delegates: &[Delegate], instructions: u64, literal: u64, payload: u64) -> u64 {
    delegates.iter().fold(
        instruction_program(instructions, literal, payload),
        |total, delegate| total.saturating_add(delegate_program(delegate)),
    )
}

// ---------------------------------------------------------------------------
// Per-cache heap of one reachable engine.
// ---------------------------------------------------------------------------

/// `Captures` owned by every cache object: the group slot vector.
fn captures_heap() -> u64 {
    reserve(SLOT, CAPTURE_SLOTS).saturating_add(exact(PTR, 2))
}

/// PikeVM cache, forward only (`meta::Cache` owns one forward PikeVM; a
/// reverse PikeVM does not exist). Four `StateID` sparse sets, the two slot
/// tables `n*g + max(g, 2q)` and the epsilon-closure stack `1 + u + c`.
fn pike_heap(nfa: Nfa) -> u64 {
    if nfa.states == 0 {
        return 0;
    }
    let slots = nfa
        .states
        .saturating_mul(CAPTURE_SLOTS)
        .saturating_add(CAPTURE_SLOTS.max(2 * PATTERNS));
    reserve(STATE_ID, nfa.states).saturating_mul(4)
        + reserve(SLOT, slots).saturating_mul(2)
        + reserve(
            FOLLOW_EPSILON,
            1u64.saturating_add(nfa.surplus)
                .saturating_add(nfa.capture_states),
        )
}

/// Accepted haystack span of the bounded backtracker: the configured visited
/// capacity in bits over the NFA state count, minus one.
fn max_haystack(nfa: Nfa) -> u64 {
    if nfa.states == 0 {
        return 0;
    }
    let bits = VISITED_CAPACITY_BYTES.saturating_mul(8);
    bits.checked_div(nfa.states.max(1))
        .unwrap_or(0)
        .saturating_sub(1)
}

/// Bounded-backtracker cache, forward only. `backtrack.rs` visits each
/// (state, position) pair at most once, so the frame stack is bounded by
/// `1 + (h + 1) * (u + c)` and the visited bitmap by the required
/// `ceil(n * (h + 1) / 64)` words.
fn backtrack_heap(nfa: Nfa) -> u64 {
    if nfa.states == 0 {
        return 0;
    }
    let h = max_haystack(nfa);
    let positions = h.saturating_add(1);
    let words = nfa.states.saturating_mul(positions).div_ceil(8 * PTR);
    let visited_cap = VISITED_CAPACITY_BYTES
        .saturating_mul(8)
        .div_ceil(VISITED_BLOCK_BITS);
    let visited = reserve(PTR, words.min(visited_cap));
    let frames = reserve(
        FRAME,
        1u64.saturating_add(
            positions.saturating_mul(nfa.surplus.saturating_add(nfa.capture_states)),
        ),
    );
    visited.saturating_add(frames)
}

/// One hybrid DFA direction at its configured cache capacity.
///
/// `K = floor(H / (4 * d))` mandatory 512-byte transition rows; `H` bounds the
/// aggregate live payload bytes; `W = 13 + 4q + 5n` is the widest payload.
/// Both directions are initialized eagerly even though the anchored
/// `search_half` only searches forward, so both are retained.
fn hybrid_heap(nfa: Nfa, stride2: u64) -> u64 {
    if nfa.states == 0 || stride2 == 0 {
        return 0;
    }
    let stride = 1u64 << stride2;
    let k = HYBRID_CACHE_CAPACITY / (4 * stride);
    let w = HYBRID_STATE_FIXED
        .saturating_add(4 * PATTERNS)
        .saturating_add(nfa.states.saturating_mul(5));
    reserve(LAZY_STATE_ID, k.saturating_mul(stride))
        .saturating_add(reserve(LAZY_STATE_ID, HYBRID_START_SLOTS))
        .saturating_add(reserve(HYBRID_STATE_HEADER, k))
        .saturating_add(table(HYBRID_MAP_ENTRY, k))
        .saturating_add(HYBRID_CACHE_CAPACITY)
        .saturating_add(HYBRID_STATE_HEADER.saturating_mul(k))
        .saturating_add(reserve(STATE_ID, nfa.states).saturating_mul(4))
        .saturating_add(reserve(STATE_ID, 1u64.saturating_add(nfa.surplus)))
        .saturating_add(reserve(U8_, w))
        .saturating_add(HYBRID_STATE_HEADER.saturating_add(reserve(U8_, w)))
}

/// The heap owned by one cache object of one delegate.
///
/// A dense-only delegate is searched exclusively through its immutable dense
/// DFA: the anchored `search_half`, the empty default quit set and the absent
/// Unicode word-boundary assertion leave `search_half_nofail`, the PikeVM and
/// the bounded backtracker unreachable, so those caches are never allocated.
/// The four hybrid delegates keep their fallback heaps because a lazy-cache
/// failure can fall through to them.
fn cache_heap(delegate: &Delegate) -> u64 {
    if !delegate.core {
        // `Pre<P>::create_cache` still owns `Captures::all(group_info)` and
        // four `none()` engine caches; only its search path is a prefilter.
        return captures_heap();
    }
    let mut bytes = captures_heap();
    if delegate.stride2 != 0 {
        bytes = bytes
            .saturating_add(pike_heap(delegate.forward))
            .saturating_add(backtrack_heap(delegate.forward))
            .saturating_add(hybrid_heap(delegate.forward, delegate.stride2))
            .saturating_add(hybrid_heap(delegate.reverse, delegate.stride2));
    }
    bytes
}

/// One delegate pool: header, `P` cache-line stack headers with their boxed
/// cache slots, and the `P + 1` cache objects (the inline owner plus one per
/// stack index), each holding its own `meta::Cache` and heap. A prefilter-only
/// strategy still owns that cache object even though its search never touches
/// a regex engine.
fn pool_bytes(delegate: &Delegate, stacks: u64) -> u64 {
    let object = META_CACHE.saturating_add(cache_heap(delegate));
    POOL_HEADER
        .saturating_add(stacks.saturating_mul(POOL_STACK))
        .saturating_add(stacks.saturating_mul(reserve(PTR, 1)))
        .saturating_add(stacks.saturating_mul(object))
        // The inline owner slot is part of every `inner::Pool` allocation; its
        // heap cache is created on the first `get`.
        .saturating_add(POOL_OWNER_SLOT.saturating_add(cache_heap(delegate)))
}

/// Pool headers of a compiled-but-never-searched literal: the outer boxed
/// inner pool (including its inline uninitialised owner slot), `P` empty
/// cache-line stack headers and no cache object. ByteLevel's `RE` is never
/// searched with `use_regex=false`, so the owner cache and stack boxes remain
/// absent.
fn idle_pool_bytes(_delegate: &Delegate, stacks: u64) -> u64 {
    POOL_HEADER
        .saturating_add(stacks.saturating_mul(POOL_STACK))
        .saturating_add(POOL_OWNER_SLOT)
}

fn literal_caches(delegates: &[Delegate], stacks: u64) -> u64 {
    delegates.iter().fold(0u64, |total, delegate| {
        total.saturating_add(pool_bytes(delegate, stacks))
    })
}

fn literal_idle_pools(delegates: &[Delegate], stacks: u64) -> u64 {
    delegates.iter().fold(0u64, |total, delegate| {
        total.saturating_add(idle_pool_bytes(delegate, stacks))
    })
}

// ---------------------------------------------------------------------------
// BPE model: retained inventory, cold-constructor peak, request scratch.
// ---------------------------------------------------------------------------

/// `AddedVocabulary` Aho-Corasick geometry: `P` patterns and `L` total
/// pattern bytes. The approved profile is `P=26, L=520`.
fn aho_shape(patterns: u64, pattern_bytes: u64) -> Result<(u64, u64, u64)> {
    // `s` is the trie state count (one per pattern byte plus the root) and the
    // per-state row width used by the pinned builder's sparse/dense tables.
    let s = cadd(pattern_bytes, 4)?;
    let e = cadd(1, cmul(ALPHABET, s)?)?;
    let m = cadd(1, cmul(patterns, s)?)?;
    Ok((s, e, m))
}

/// Builder prefilter rows shared by the retained DFA and builder NFA: packed
/// pattern needles, Rabin-Karp and Teddy prefilters, temporary search tables,
/// start/rare byte sets and the memmem fallback.
fn aho_prefilter_build(patterns: u64, pattern_bytes: u64) -> Result<u64> {
    let pat = csum(&[
        checked_regrow(24, patterns)?,
        checked_regrow(4, patterns)?,
        fragment_regrow_bytes(1, pattern_bytes, patterns)?,
        checked_regrow(4, patterns)?,
        checked_arc(8, 80)?,
    ])?;
    let rabin_karp = cadd(
        exact_bytes(24, AHO_MAX_RK_BUCKETS)?,
        fragment_regrow_bytes(16, patterns, patterns.min(AHO_MAX_RK_BUCKETS))?,
    )?;
    let teddy_payload = cadd(cadd(8, cmul(16 * 24, 1)?)?, cmul(4 * 2 * 32, 1)?)?;
    let teddy = cadd(
        checked_arc(32, checked_align_up(teddy_payload, 32)?)?,
        fragment_regrow_bytes(4, patterns, patterns.min(AHO_MAX_TEDDY_BUCKETS))?,
    )?;
    let temporary = csum(&[
        exact_bytes(24, 16)?,
        checked_btree(16, 8, patterns)?,
        cmul(4, patterns)?,
    ])?;
    // Actual Teddy mask builder elements are two 32 B vectors (`2 * 4 *
    // size_of::<u64>()`), not a scalar u64: `G64(4)`.
    let masks = checked_regrow(2 * 4 * U64, 4)?;
    let packed_header = checked_arc(8, 104)?;
    let start_rare = cadd(checked_arc(8, 3)?, checked_arc(8, 259)?)?;
    let memmem = checked_regrow(U8_, 20)?;
    csum(&[
        cmul(2, pat)?,
        rabin_karp,
        teddy,
        temporary,
        masks,
        packed_header,
        start_rare,
        memmem,
    ])
}

fn aho_prefilter_retained(patterns: u64, pattern_bytes: u64) -> Result<u64> {
    // `Prefilter::build` chooses one strategy; the packed builder path is an
    // upper bound and retains the same rows.
    aho_prefilter_build(patterns, pattern_bytes)
}

/// Retained DFA chosen for the approved pattern count, including one complete
/// `Prefilter` owner per automaton. `s` is the source-derived state capacity.
fn aho_dfa_retained(patterns: u64, pattern_bytes: u64) -> Result<u64> {
    let (s, _e, _m) = aho_shape(patterns, pattern_bytes)?;
    let hd = checked_align_up(
        csum(&[
            cmul(3, 24)?,
            8,
            cadd(32, 8)?,
            cmul(3, 8)?,
            ALPHABET,
            cmul(2, 8)?,
            cmul(4, 4)?,
        ])?,
        8,
    )?;
    let header = checked_arc(8, hd)?;
    let transitions = exact_bytes(4, cmul(ALPHABET, s)?)?;
    let epsilon = exact_bytes(24, s)?;
    let match_rows = fragment_bytes(4, cmul(patterns, s)?, s)?;
    let pattern_ids = exact_bytes(4, patterns)?;
    let prefilter = aho_prefilter_retained(patterns, pattern_bytes)?;
    csum(&[
        header,
        transitions,
        epsilon,
        match_rows,
        pattern_ids,
        prefilter,
    ])
}

/// Builder-live noncontiguous NFA: state/transition vectors, dense rows for
/// shallow states, match/pattern tables, start-state rows and the same
/// prefilter owner. This is scratch, not retained.
fn aho_nfa_build(patterns: u64, pattern_bytes: u64) -> Result<u64> {
    let (s, e, m) = aho_shape(patterns, pattern_bytes)?;
    csum(&[
        checked_regrow(20, s)?,
        checked_regrow(9, e)?,
        checked_regrow(4, e)?,
        checked_regrow(8, m)?,
        checked_regrow(4, patterns)?,
        cmul(2, checked_regrow(4, s)?)?,
        aho_prefilter_build(patterns, pattern_bytes)?,
    ])
}

/// One normalized and one non-normalized `AhoCorasick`; both use
/// `MatchKind::LeftmostLongest`, which rejects overlapping/stream searches, so
/// no overlapping-match vectors are live. Retained storage is two DFAs.
fn split_automata_retained() -> Result<u64> {
    let patterns = ADDED_TOKENS;
    let bytes = cmul(patterns, MAX_ADDED_ENTRY_BYTES)?;
    let first = aho_dfa_retained(patterns, bytes)?;
    let second = aho_dfa_retained(0, 0)?;
    cadd(cadd(first, second)?, checked_regrow(4, patterns)?)
}

/// Constructor peak adds the builder NFA, a second complete DFA and the
/// non-normalized empty automaton while the earlier refresh may still own two
/// tries; the retained row is charged in addition by the retained ledger.
fn split_automata_peak() -> Result<u64> {
    let patterns = ADDED_TOKENS;
    let bytes = cmul(patterns, MAX_ADDED_ENTRY_BYTES)?;
    let first = aho_dfa_retained(patterns, bytes)?;
    let empty = aho_dfa_retained(0, 0)?;
    let build = aho_nfa_build(patterns, bytes)?;
    let build_empty = aho_nfa_build(0, 0)?;
    csum(&[
        cmul(2, first)?,
        cmul(2, empty)?,
        build,
        build_empty,
        checked_regrow(16, patterns)?,
        checked_regrow(8, patterns)?,
        checked_regrow(4, patterns)?,
    ])
}

/// Retained `BPE` model storage: forward and reverse vocabularies (one owned
/// `String` mass each), merge table, added-token vocabulary (maps, ordered
/// token vectors, special-token set, split automata) and the two empty affix
/// strings. The default merge cache is shrunk to zero capacity before use.
fn bpe_retained() -> Result<u64> {
    let vocab_forward = checked_hash_of(STRING, U32, VOCAB_ENTRIES)?;
    let vocab_reverse = checked_hash_of(U32, STRING, VOCAB_ENTRIES)?;
    let merge_key = cmul(2, U32)?;
    let merges = checked_hash_of(merge_key, merge_key, MERGE_ENTRIES)?;
    let strings = cmul(2, APPROVED_TOKENIZER_LEN)?;
    let added = csum(&[
        checked_hash_of(STRING, U32, ADDED_TOKENS)?,
        checked_hash_of(U32, ADDED_TOKEN, ADDED_TOKENS)?,
        checked_regrow(ADDED_TOKEN, ADDED_TOKENS)?,
        checked_regrow(ADDED_TOKEN, ADDED_TOKENS)?,
        checked_table(STRING, ADDED_TOKENS)?,
        cmul(exact_bytes(MAX_ADDED_ENTRY_BYTES, ADDED_TOKENS)?, 3)?,
        split_automata_retained()?,
    ])?;
    csum(&[
        vocab_forward,
        vocab_reverse,
        merges,
        strings,
        added,
        cmul(2, STRING)?,
    ])
}

/// Untagged `serde_core::Content` image of the approved model document:
/// vocabulary as `Vec<(Content, Content)>` (both members inline in each 64 B
/// pair), merges as a `Vec<Content>` whose two-element child sequences are
/// independently allocated, and one `Content` per scalar. This is a source
/// fallback only; the accepted constructor charges the censused tree `C`.
fn content_tree() -> Result<u64> {
    let vocab_pairs = cadd(
        checked_regrow(CONTENT_PAIR, VOCAB_ENTRIES)?,
        APPROVED_TOKENIZER_LEN,
    )?;
    let merge_outer = checked_regrow(CONTENT, MERGE_ENTRIES)?;
    let merge_children = fragment_bytes(CONTENT, cmul(2, MERGE_ENTRIES)?, MERGE_ENTRIES)?;
    let merges = csum(&[merge_outer, merge_children, APPROVED_TOKENIZER_LEN])?;
    let added = checked_regrow(CONTENT, cmul(ADDED_TOKENS, 8)?)?;
    csum(&[vocab_pairs, merges, added])
}

/// Flattened `serde_json::Value` twin of the model fragment that survives
/// while the BPE `MergeType` untagged buffer and the typed merges are built.
/// `members` counts the vocabulary and small model fields; `arrays` is the
/// merges outer array plus one minimum-capacity child per merge.
fn value_tree() -> Result<u64> {
    let members = cadd(VOCAB_ENTRIES, 8)?;
    let objects = 2;
    let nodes = cadd(members, objects)?;
    let arrays = cadd(
        u64::try_from(next_pow2(MERGE_ENTRIES as u128).max(4)).map_err(|_| arithmetic())?,
        cmul(4, MERGE_ENTRIES)?,
    )?;
    let nodes_bytes = cmul(nodes, BTREE_NODE)?;
    let arrays_bytes = cmul(arrays, 2 * VALUE)?;
    csum(&[nodes_bytes, arrays_bytes, APPROVED_TOKENIZER_LEN])
}

/// `(B,J,F)`: extracted byte length, syntax-tree node capacity and fancy HIR
/// node count for a literal of `l` bytes.
struct SyntaxCounts {
    b: u64,
    j: u64,
    f: u64,
}

fn syntax_counts(literal_bytes: u64) -> Result<SyntaxCounts> {
    let b = cmul(16, cadd(literal_bytes, 1)?)?;
    let j = cadd(cmul(4, b)?, 16)?;
    let f = cadd(cmul(2, literal_bytes)?, 8)?;
    Ok(SyntaxCounts { b, j, f })
}

/// `FANCY + AST + 3*HIR + CLASS` from the pinned regex-syntax builders.
fn syntax_scratch(literal_bytes: u64) -> Result<u64> {
    let SyntaxCounts { b, j, f } = syntax_counts(literal_bytes)?;
    let fancy = csum(&[
        exact_bytes(48, f)?,
        fragment_regrow_bytes(48, f, f)?,
        fragment_regrow_bytes(64, f, f)?,
        fragment_regrow_bytes(1, b, f)?,
        checked_regrow(1, b)?,
    ])?;
    let ast = csum(&[
        exact_bytes(512, j)?,
        fragment_regrow_bytes(256, j, j)?,
        cmul(2, checked_regrow(512, j)?)?,
        cmul(2, checked_regrow(64, j)?)?,
    ])?;
    let hir = csum(&[
        exact_bytes(128, j)?,
        exact_bytes(128, j)?,
        fragment_regrow_bytes(64, j, j)?,
        checked_regrow(128, j)?,
        checked_regrow(64, j)?,
    ])?;
    let class = cmul(cmul(4, cadd(literal_bytes, 1)?)?, checked_regrow(8, 2048)?)?;
    csum(&[fancy, ast, cmul(3, hir)?, class])
}

/// Two added-vocabulary extraction vectors and their borrowed slices.
fn extract_scratch() -> Result<u64> {
    let borrowed = checked_regrow(32, 250)?;
    let slices = cmul(250, checked_regrow(1, 200)?)?;
    cmul(24, cadd(borrowed, slices)?)
}

/// Literal-trie extraction scratch for at most 50 000 candidates.
fn trie_extract_scratch() -> Result<u64> {
    csum(&[
        checked_regrow(24, 50_001)?,
        checked_regrow(8, 50_001)?,
        fragment_regrow_bytes(16, 50_000, 50_001)?,
    ])
}

/// Forward and reverse Thompson construction, including the configured NFA
/// limit envelope, its optional second independent deterministic conversion,
/// the 256-bucket UTF-8 compiler scratch and the literal trie.
fn thompson_scratch(literal_bytes: u64) -> Result<u64> {
    let SyntaxCounts { b, j, .. } = syntax_counts(literal_bytes)?;
    let limit = 10 * (1 << 20);
    let overshoot = cmul(8, ALPHABET)?.max(cmul(4, j)?);
    let b1 = cadd(cadd(limit, 32)?, overshoot)?;
    let s1 = cadd(b1 / 32, 1)?;
    let grow = cmul(6, b1)?;
    let build = csum(&[
        exact_bytes(4, s1)?,
        checked_regrow(8, s1)?,
        exact_bytes(1, s1)?,
    ])?;
    let final_conversion = cmul(6, b1)?;
    let start = csum(&[
        checked_regrow(4, 1)?,
        checked_regrow(24, 1)?,
        checked_regrow(16, 1)?,
    ])?;
    let utf8 = csum(&[
        checked_regrow(40, 10_000)?,
        cmul(10_000, checked_regrow(8, ALPHABET)?)?,
        checked_regrow(32, 4)?,
        cmul(4, checked_regrow(8, ALPHABET)?)?,
    ])?;
    let reverse = checked_regrow(16, 1_000)?;
    let literals = cadd(b, 1)?;
    let trie = csum(&[
        checked_regrow(48, literals)?,
        cmul(2, fragment_regrow_bytes(16, b, literals)?)?,
        checked_regrow(4, literals)?,
        checked_regrow(8, literals)?,
    ])?;
    csum(&[grow, build, final_conversion, start, utf8, reverse, trie])
}

/// One complete dense DFA build, including old/new state rows, the 24-byte
/// hash table, byte-class rows, acceleration tables and three BTree nodes.
fn dense_scratch(nfa_states: u64, half_states: u64) -> Result<u64> {
    if half_states == 0 {
        return Ok(0);
    }
    const DENSE: u64 = 10_240;
    let states = cadd(DENSE / 4, half_states)?;
    let capacity = cadd(DENSE / (4 * half_states), 1)?;
    let width = cadd(17, cmul(5, nfa_states)?)?;
    csum(&[
        checked_regrow(4, states)?,
        checked_regrow(16, capacity)?,
        cmul(2, checked_table(24, capacity)?)?,
        cadd(DENSE, width)?,
        cmul(24, capacity)?,
        cmul(4, checked_regrow(4, nfa_states)?)?,
        checked_regrow(4, cadd(1, cmul(ALPHABET, nfa_states)?)?)?,
        cmul(2, checked_regrow(1, width)?)?,
        checked_regrow(4, capacity)?,
        checked_regrow(8, 257)?,
        cmul(3, checked_btree(4, 24, capacity)?)?,
        fragment_regrow_bytes(4, capacity, capacity)?,
        cmul(3, checked_regrow(4, capacity)?)?,
        checked_regrow(8, capacity)?,
        checked_regrow(4, 18)?,
    ])
}

/// Compiler scratch before the installed program. `DENSE` is a maximum over
/// the one delegate whose lazy build borrows the scratch; the other rows are
/// live sums. ByteLevel's compiler is not part of the per-source constructor.
fn compiler_scratch(literal_bytes: u64, delegates: &[Delegate], instructions: u64) -> Result<u64> {
    let mut dense = 0u64;
    for delegate in delegates {
        if !delegate.core || delegate.dense.0 == 0 {
            continue;
        }
        let half =
            u64::try_from(next_pow2(delegate.classes as u128 + 1)).map_err(|_| arithmetic())?;
        // `dense.0/1` are DFA state counts and only gate the build. The
        // deterministic construction consumes the ORIGINAL Thompson NFA
        // state counts, e.g. delegate 54 has reverse NFA 59, not DFA 18.
        let forward = dense_scratch(delegate.forward.states, half)?;
        let reverse = dense_scratch(delegate.reverse.states, half)?;
        dense = dense.max(cadd(forward, reverse)?);
    }
    csum(&[
        syntax_scratch(literal_bytes)?,
        extract_scratch()?,
        trie_extract_scratch()?,
        thompson_scratch(literal_bytes)?,
        dense,
        checked_regrow(40, instructions)?,
    ])
}

/// Non-Value constructor rows (vocabularies, typed and legacy merges, cache,
/// added tokens, the split automata construction peak and one compiler
/// envelope). The Value/Content trees are charged separately.
fn bpe_constructor_static(plan: &TokenizerPlan) -> Result<u64> {
    let vocab_forward = checked_hash_of(STRING, U32, VOCAB_ENTRIES)?;
    let vocab_reverse = checked_hash_of(U32, STRING, VOCAB_ENTRIES)?;
    let merge_key = cmul(2, U32)?;
    let merges = checked_hash_of(merge_key, merge_key, MERGE_ENTRIES)?;
    let cached = checked_hash_of(STRING, VEC, BPE_DEFAULT_CACHE)?;
    let typed_merges = checked_regrow(MERGE_TYPED, MERGE_ENTRIES)?;
    let legacy_merges = checked_regrow(MERGE_LEGACY, MERGE_ENTRIES)?;
    let reverse_merges = checked_hash_of(merge_key, merge_key, MERGE_ENTRIES)?;
    // Both `(String,String)` masses plus the legacy copy that may still be
    // alive while the typed merges are built.
    let owned_strings = cmul(4, APPROVED_TOKENIZER_LEN)?;
    let added = csum(&[
        checked_hash_of(STRING, U32, ADDED_TOKENS)?,
        checked_hash_of(U32, ADDED_TOKEN, ADDED_TOKENS)?,
        cmul(exact_bytes(MAX_ADDED_ENTRY_BYTES, ADDED_TOKENS)?, 3)?,
        split_automata_retained()?,
        split_automata_peak()?,
    ])?;
    csum(&[
        vocab_forward,
        vocab_reverse,
        merges,
        cached,
        typed_merges,
        legacy_merges,
        reverse_merges,
        owned_strings,
        added,
        BPE_MERGE_SCRATCH,
        checked_regrow(U8_, BPE_MAX_KEY)?,
        compiler_scratch(SPLIT_LITERAL_BYTES, &SPLIT_DELEGATES, SPLIT_INSTRUCTIONS)?,
        plan.split_program_bytes,
    ])
}

/// Three complete censused tree images (flattened Value, ModelHelper Content,
/// MergeType Content) plus the pre-census scratch. The accepted constructor
/// supplies `C` from the actual source.
pub fn constructor_tree_bytes(census_tree: u64, source_len: u64) -> Result<u64> {
    csum(&[cmul(3, census_tree)?, checked_census_scratch(source_len)?])
}

/// Source fallback for an API without the inspected bytes. The real
/// `constructor_of_arbiter` path uses the censused `C`, never this envelope.
fn bpe_constructor_value() -> Result<u64> {
    let c = value_tree()?.max(content_tree()?);
    constructor_tree_bytes(c, APPROVED_TOKENIZER_LEN)
}

// ---------------------------------------------------------------------------
// Request phases.
// ---------------------------------------------------------------------------

/// Rendered bytes `B`; `N <= 3B` NFC-normalized bytes, `E <= 6B` ByteLevel
/// bytes, `S <= 3B` split fragments, `T <= 3B` BPE tokens, `K <= 2B` symbols
/// and `O <= B + 4S` summed original fragment bytes.
///
/// The peak is the simultaneous sum of the named owners. Every phase keeps the
/// live input, the transformed `NormalizedString` bodies, the attached token
/// arrays and the `Encoding` sidecars alive, so phase maxima cannot replace the
/// sum. Only the *engine* term is a maximum over the single delegate whose
/// cache the search borrows.
fn admitted_prompt_peak_impl(rendered: u64) -> u64 {
    let b = rendered;
    let n = b.saturating_mul(3);
    let e = b.saturating_mul(6);
    let k = b.saturating_mul(2);
    let t = b.saturating_mul(3);
    let s = b.saturating_mul(3);
    let o = b.saturating_add(s.saturating_mul(4));

    // Base: the request copy, the rendered `String`, the pretokenized
    // fragment owners and the attached token arrays/contents.
    let base = reserve(U8_, b)
        .saturating_add(STRING)
        .saturating_add(reserve(SPLIT, s))
        .saturating_add(fragments(U8_, o, s))
        .saturating_add(fragments(U8_, e, s))
        .saturating_add(fragments(PAIR, e, s))
        .saturating_add(fragments(TOKEN, t, s))
        .saturating_add(fragments(U8_, e.saturating_add(b), t));

    // NFC: cloned input, replaced chars, two destination/splice strings and
    // alignments, the spilled decomposition/recomposition buffers with their
    // old+new overlap, and stable-sort workspace.
    let nfc = reserve(U8_, b)
        .saturating_add(regrow(CHAR_ISIZE, b))
        .saturating_add(regrow(U8_, n).saturating_mul(2))
        .saturating_add(regrow(PAIR, n).saturating_mul(2))
        .saturating_add(regrow(U8_, k))
        .saturating_add(regrow(CHAR_ISIZE, k))
        .saturating_add(regrow(PAIR, k))
        .saturating_add(NORMALIZED_STRING.saturating_mul(2));

    // Split: match/offset-bool vectors, intermediate normal strings, old/new
    // outer arrays, copied fragment bodies and two alignment generations, plus
    // the two added-vocabulary extraction vectors.
    let split = reserve(OFFSETS_BOOL, n.saturating_add(1))
        .saturating_mul(2)
        .saturating_add(reserve(NORMALIZED_STRING, s))
        .saturating_add(reserve(SPLIT, s).saturating_mul(2))
        .saturating_add(fragments(U8_, o, s))
        .saturating_add(regrow(PAIR, n).saturating_mul(2))
        .saturating_add(reserve(OPTION_U32.saturating_add(OFFSETS_BOOL), s))
        .saturating_add(reserve(NORMALIZED_STRING.saturating_add(VEC), s));

    // ByteLevel: one `(char, isize)` transform per normalized byte, replaced
    // chars, the mapped string and splice generation, and their alignments.
    let byte_level = reserve(CHAR_ISIZE, n)
        .saturating_add(regrow(CHAR_ISIZE, n))
        .saturating_add(regrow(U8_, e).saturating_mul(2))
        .saturating_add(regrow(PAIR, e).saturating_mul(2))
        .saturating_add(reserve(NORMALIZED_STRING, 1))
        .saturating_add(reserve(SPLIT, s));

    // BPE: `Word` reserves one Symbol per mapped byte, merges are at most
    // `3N` heap entries with old+new overlap, plus the two affix strings at
    // their source length and the discarded <=255-byte cache key.
    let bpe = reserve(SYMBOL, e)
        .saturating_add(regrow(MERGE, n.saturating_mul(3)))
        .saturating_add(exact(2, 2))
        .saturating_add(reserve(U8_, BPE_MAX_KEY));

    // Encoding: seven typed backing arrays over `T` tokens, the merge output
    // backing with its collected offsets intermediate, the input token arrays,
    // the ids copy and the sequence-range map.
    let encoding = reserve(U32, t)
        .saturating_mul(2)
        .saturating_add(reserve(STRING, t))
        .saturating_add(reserve(OPTION_U32, t))
        .saturating_add(reserve(PAIR, t))
        .saturating_add(reserve(U32, t))
        .saturating_add(reserve(PAIR, t))
        .saturating_add(reserve(U32, t))
        .saturating_add(reserve(U32, t))
        .saturating_add(reserve(PAIR, t).saturating_mul(2))
        .saturating_add(table(align_up(PTR.saturating_add(PAIR), 8), 1))
        .saturating_add(ENCODING)
        .saturating_add(VEC);

    base.saturating_add(nfc)
        .saturating_add(split)
        .saturating_add(byte_level)
        .saturating_add(bpe)
        .saturating_add(encoding)
}

/// One prompt/encode request at `rendered` bytes without the serialized-VM
/// owner; `None` above the priced prompt cap. The actual `Retained` source
/// owner prepays `serialized_vm_bytes`, so serving callers use this bound.
/// Generic callers that do not hold that owner use [`prompt_peak`].
pub fn admitted_prompt_peak(rendered: u64) -> Option<u64> {
    (rendered <= MAX_PRICED_PROMPT_BYTES).then(|| admitted_prompt_peak_impl(rendered))
}

/// Fancy-regex VM request scratch only, never multiplied by `P`. This is a
/// checked named plan row and is prepaid by the source `Retained` owner.
fn serialized_vm_bytes() -> Result<u64> {
    let save = VM_STACK
        .checked_mul(2)
        .and_then(|n| n.checked_add(2))
        .ok_or_else(arithmetic)?;
    csum(&[
        exact_bytes(PTR, 2)?,
        checked_regrow(BRANCH, VM_STACK)?,
        checked_regrow(SAVE, save)?,
    ])
}

/// Full request peak including the serialized VM; used by callers that do not
/// hold the source `Retained` owner.
fn prompt_peak_impl(rendered: u64) -> u64 {
    admitted_prompt_peak_impl(rendered).saturating_add(serialized_vm_bytes().unwrap_or(u64::MAX))
}

/// Largest cache growth one request can cause: only delegates whose fallback
/// engine is reachable own a growable heap.
fn request_growth(delegate: &Delegate) -> u64 {
    if !delegate.core {
        return 0;
    }
    cache_heap(delegate)
}

/// One `decode` call over `tokens` output ids: the reversed id vector, the
/// per-token strings, the ByteLevel reverse map, the lossy UTF-8 output and its
/// copies, and the final join.
fn decode_peak_impl(tokens: u64) -> u64 {
    let per_token = MAX_VOCAB_ENTRY_BYTES.saturating_mul(3);
    let text = per_token.saturating_mul(tokens);
    reserve(U32, tokens)
        .saturating_add(reserve(STRING, tokens))
        .saturating_add(exact(MAX_VOCAB_ENTRY_BYTES, tokens))
        .saturating_add(regrow(U8_, text))
        .saturating_add(regrow(U8_, per_token))
        .saturating_add(exact(U8_, text))
        .saturating_add(regrow(U8_, text))
        .saturating_add(reserve(STRING, 1))
        .saturating_add(BPE_MAX_KEY)
        .saturating_add(ENCODING_SIDECAR)
}

/// `Encoding`-sidecar struct headers kept live while decoding one token.
const ENCODING_SIDECAR: u64 = 8 * VEC;

// ---------------------------------------------------------------------------
// Public plan.
// ---------------------------------------------------------------------------

/// Retained, constructor, bootstrap and per-request storage of the approved
/// profile at `pool_stacks` `meta::Regex` pool stacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenizerPlan {
    /// `meta::Config::get_pool_capacity()` read at admission.
    pub pool_stacks: u64,
    /// Retained Split regex program plus every delegate pool at `P`.
    pub split_retained_bytes: u64,
    /// Process-global ByteLevel program and idle pool headers (one owner for
    /// the process, not one per tokenizer).
    pub byte_level_retained_bytes: u64,
    /// Retained BPE model and added-token inventory.
    pub bpe_retained_bytes: u64,
    /// Cold-constructor peak, released once the model is installed. This is
    /// the source fallback envelope; callers holding the exact bytes charge
    /// `3*C + census_scratch_peak(F)` for the censused tree instead.
    pub bpe_constructor_peak_bytes: u64,
    /// Non-Value constructor rows, including the complete Split compiler
    /// envelope. ByteLevel's compiler belongs to the process bootstrap.
    pub bpe_constructor_static_bytes: u64,
    /// Fallback Value/Content twin when the caller cannot hold the exact
    /// source bytes; the exact path does not use this row.
    pub bpe_constructor_value_bytes: u64,
    /// Process-lifetime ByteLevel bootstrap (`RE`, `LazyBYTES_CHAR`,
    /// `CHAR_BYTES`) retained by the process, not by a router or source.
    pub bootstrap_bytes: u64,
    /// Once-per-process bootstrap peak: ByteLevel compiler scratch plus the
    /// permanent bootstrap owner while the statics are being published.
    pub bootstrap_peak_bytes: u64,
    /// `split + bpe` retained per source owner; the ByteLevel program is a
    /// process-global owner and is charged by the bootstrap lease instead.
    pub retained_bytes: u64,
    /// Reserved application capacity for one source tokenizer's largest
    /// reachable cache reallocation overlap. Charged once per actual tokenizer
    /// owner (`G`), shared by its aliases, not once per request. This is
    /// reserved capacity, not an RSS claim.
    pub request_cache_growth_bytes: u64,
    /// Serialized fancy-regex VM request scratch (`V`), charged once per actual
    /// tokenizer owner and shared by its aliases, not once per request.
    pub serialized_vm_bytes: u64,
    /// Retained Split program bytes (constructor scratch input).
    pub split_program_bytes: u64,
    /// Retained ByteLevel program bytes (constructor scratch input).
    pub byte_level_program_bytes: u64,
}

/// Sanity caps: the adapter rejects longer prompts and wider token batches
/// before it prices them, so the saturating arithmetic below cannot understate
/// a request the server would actually admit.
pub const MAX_PRICED_PROMPT_BYTES: u64 = 2 * (1 << 20);
pub const MAX_PRICED_DECODE_TOKENS: u64 = 1 << 16;
const MAX_POOL_STACKS: usize = 1 << 12;

/// `P` from the pinned meta configuration; `1` if the crate reports zero.
pub fn plan(pool_stacks: usize) -> Result<TokenizerPlan> {
    if pool_stacks > MAX_POOL_STACKS {
        return Err(Error::Limit(format!(
            "tokenizer pool capacity {pool_stacks} exceeds {MAX_POOL_STACKS}"
        )));
    }
    let stacks = pool_stacks.max(1) as u64;
    let split_program = literal_program(
        &SPLIT_DELEGATES,
        SPLIT_INSTRUCTIONS,
        SPLIT_LITERAL_BYTES,
        SPLIT_LITERAL_STRING_PAYLOAD,
    );
    let byte_level_program = literal_program(
        &BYTE_LEVEL_DELEGATES,
        BYTE_LEVEL_INSTRUCTIONS,
        BYTE_LEVEL_LITERAL_BYTES,
        BYTE_LEVEL_LITERAL_STRING_PAYLOAD,
    );
    let split_retained = split_program.saturating_add(literal_caches(&SPLIT_DELEGATES, stacks));
    // ByteLevel is compiled once per process and never searched by this
    // profile: program plus idle pool headers, one owner.
    let byte_level_retained =
        byte_level_program.saturating_add(literal_idle_pools(&BYTE_LEVEL_DELEGATES, stacks));
    let bpe_retained = bpe_retained()?;
    // Bootstrap: the process-global ByteLevel compiler scratch, its permanent
    // program, idle pool headers and the two 256-entry alphabet maps. The
    // compiler scratch is released before the permanent owner is published.
    let bootstrap = byte_level_retained
        .saturating_add(exact(BOX_SLICE, 1))
        .saturating_add(table(CHAR_ISIZE, 256))
        .saturating_add(table(U32, 256))
        .saturating_add(regrow(U8_, 256))
        .saturating_add(regrow(U32, 256));
    let bootstrap_peak = csum(&[
        compiler_scratch(
            BYTE_LEVEL_LITERAL_BYTES,
            &BYTE_LEVEL_DELEGATES,
            BYTE_LEVEL_INSTRUCTIONS,
        )?,
        bootstrap,
    ])?;
    let mut plan = TokenizerPlan {
        pool_stacks: stacks,
        split_retained_bytes: split_retained,
        byte_level_retained_bytes: byte_level_retained,
        bpe_retained_bytes: bpe_retained,
        bpe_constructor_peak_bytes: 0,
        bpe_constructor_static_bytes: 0,
        bpe_constructor_value_bytes: 0,
        bootstrap_bytes: bootstrap,
        bootstrap_peak_bytes: bootstrap_peak,
        retained_bytes: split_retained.saturating_add(bpe_retained),
        request_cache_growth_bytes: 0,
        serialized_vm_bytes: 0,
        split_program_bytes: split_program,
        byte_level_program_bytes: byte_level_program,
    };
    plan.bpe_constructor_static_bytes = bpe_constructor_static(&plan)?;
    plan.bpe_constructor_value_bytes = bpe_constructor_value()?;
    plan.bpe_constructor_peak_bytes = cadd(
        plan.bpe_constructor_static_bytes,
        plan.bpe_constructor_value_bytes,
    )?;
    plan.request_cache_growth_bytes = SPLIT_DELEGATES
        .iter()
        .fold(0u64, |peak, delegate| peak.max(request_growth(delegate)));
    plan.serialized_vm_bytes = serialized_vm_bytes()?;
    Ok(plan)
}

/// One prompt/encode request at `rendered` bytes; `None` above the cap.
pub fn prompt_peak(rendered: u64) -> Option<u64> {
    (rendered <= MAX_PRICED_PROMPT_BYTES).then(|| prompt_peak_impl(rendered))
}

/// One `decode` of `tokens` ids; `None` above the cap.
pub fn decode_peak(tokens: u64) -> Option<u64> {
    (tokens <= MAX_PRICED_DECODE_TOKENS).then(|| decode_peak_impl(tokens))
}

/// Does the source subtotal plus `requests` simultaneous requests fit a host
/// budget in MiB? The source subtotal is `bootstrap + R + G + V`, charged once
/// per actual tokenizer; only admitted prompt (without `V`) and decode are per
/// request. Used by ledger tests only.
#[cfg(test)]
pub fn fits(plan: &TokenizerPlan, host_mib: u64, rendered: u64, requests: u64) -> bool {
    let request = admitted_prompt_peak(rendered)
        .unwrap()
        .saturating_add(decode_peak(512).unwrap());
    let source = plan
        .bootstrap_bytes
        .saturating_add(plan.retained_bytes)
        .saturating_add(plan.request_cache_growth_bytes)
        .saturating_add(plan.serialized_vm_bytes);
    source.saturating_add(request.saturating_mul(requests)) <= host_mib.saturating_mul(1 << 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn totals(delegates: &[Delegate]) -> (u64, u64, u64, u64, u64) {
        let mut core = 0;
        let mut fwd = 0;
        let mut rev = 0;
        let mut dense = 0;
        let mut hybrid = 0;
        delegates.iter().for_each(|delegate| {
            if delegate.core {
                core += 1;
                fwd += delegate.forward.states;
                rev += delegate.reverse.states;
                if delegate.dense.0 != 0 {
                    dense += 1;
                }
                if delegate.stride2 != 0 {
                    hybrid += 1;
                }
            }
        });
        (core, fwd, rev, dense, hybrid)
    }

    #[test]
    fn descriptor_tables_match_the_compiled_literals() {
        assert_eq!(totals(&SPLIT_DELEGATES), (15, 1191, 4608, 11, 4));
        assert_eq!(totals(&BYTE_LEVEL_DELEGATES), (6, 783, 3009, 3, 3));
        assert_eq!(SPLIT_DELEGATES.len(), 17);
        assert_eq!(
            SPLIT_DELEGATES.iter().filter(|d| !d.core).count(),
            2,
            "two prefilter-only delegates"
        );
        assert_eq!(SPLIT_DELEGATES[7].reverse.surplus, 932);
        assert_eq!(SPLIT_DELEGATES[10].reverse.surplus, 950);
        assert_eq!(SPLIT_DELEGATES.iter().map(|d| d.classes).max(), Some(116));
        assert_eq!(
            BYTE_LEVEL_DELEGATES.iter().map(|d| d.classes).max(),
            Some(115)
        );
        // Every Split core delegate is either dense or hybrid, never both, and
        // carries a single group.
        for delegate in SPLIT_DELEGATES.iter().filter(|d| d.core) {
            assert_ne!(delegate.dense.0 != 0, delegate.stride2 != 0);
            assert_eq!(delegate.forward.capture_states, 2);
            assert_eq!(delegate.forward.surplus, 1);
        }
    }

    #[test]
    fn dense_delegates_own_no_fallback_heap() {
        let dense = SPLIT_DELEGATES
            .iter()
            .find(|d| d.core && d.dense.0 != 0)
            .unwrap();
        // Captures only: no PikeVM slot tables, no backtracker frames.
        assert_eq!(cache_heap(dense), captures_heap());
        // The hybrid delegate keeps its forward PikeVM/backtracker heaps and
        // both hybrid directions.
        let hybrid = SPLIT_DELEGATES
            .iter()
            .find(|d| d.core && d.stride2 != 0)
            .unwrap();
        assert_eq!(
            cache_heap(hybrid),
            captures_heap()
                + pike_heap(hybrid.forward)
                + backtrack_heap(hybrid.forward)
                + hybrid_heap(hybrid.forward, hybrid.stride2)
                + hybrid_heap(hybrid.reverse, hybrid.stride2)
        );
        assert_eq!(hybrid_heap(hybrid.forward, 0), 0);
    }

    #[test]
    fn collection_primitives_follow_raw_vec_minimum_capacities() {
        // u8 first push: 8 bytes, not 2.
        assert_eq!(reserve(U8_, 1), 8);
        // 8-byte pointer first push: 4 elements = 32 bytes, not 16.
        assert_eq!(reserve(PTR, 1), 32);
        // (String, u32) node is 32 bytes: 24 + 4 padded to the pair alignment.
        assert_eq!(align_up(STRING + U32, 8), 32);
        // A one-entry map still owns four buckets, never two.
        assert_eq!(buckets_of(1), 4);
        assert_eq!(buckets_of(3), 4);
        // Load factor 7/8: 524288 entries need 1048576 buckets.
        assert_eq!(buckets_of(524_288), 1_048_576);
        assert_eq!(table(32, 524_288) - table(28, 524_288), 1_048_576 * 4);
    }

    #[test]
    fn typed_element_sizes_match_the_pinned_declarations() {
        assert_eq!(SYMBOL, 32);
        assert_eq!(MERGE, 16);
        assert_eq!(OFFSETS_BOOL, 24);
        assert_eq!(CHAR_ISIZE, 16);
        assert_eq!(NORMALIZED_STRING, 80);
        assert_eq!(TOKEN, 48);
        assert_eq!(SPLIT, 104);
        assert_eq!(INSTRUCTION, 40);
        assert_eq!(HYBRID_STRIDE * LAZY_STATE_ID, 512);
        assert_eq!(HYBRID_CACHE_CAPACITY / (4 * HYBRID_STRIDE), 4096);
    }

    #[test]
    fn pool_headers_match_the_pinned_inner_pool_and_cache_line_layout() {
        // `Pool` = boxed create closure (16) + stacks Vec header (24) + owner
        // AtomicUsize (8); each `CacheLine<Mutex<Vec<Box<T>>>>` is padded to
        // 64 B by `repr(align(64))`.
        assert_eq!(POOL_HEADER, 48);
        assert_eq!(POOL_STACK, 64);
        let delegate = SPLIT_DELEGATES[0];
        let idle = idle_pool_bytes(&delegate, 16);
        assert_eq!(idle, POOL_HEADER + 16 * POOL_STACK + POOL_OWNER_SLOT);
        let active = pool_bytes(&delegate, 16);
        assert!(active > idle);
        assert_eq!(
            active - idle,
            16 * reserve(PTR, 1) + 16 * META_CACHE + 17 * cache_heap(&delegate)
        );
    }

    #[test]
    fn backtracker_span_and_frames_follow_the_proven_state_position_bound() {
        // n = 8: h = floor(2 MiB bits / 8) - 1 = 262143 positions.
        assert_eq!(max_haystack(nfa!(8, 1, 2)), 262_143);
        // n = 336: the configured capacity only admits 6242 positions.
        assert_eq!(max_haystack(nfa!(336, 1, 2)), 6240);
        let hybrid = SPLIT_DELEGATES[7];
        let h = max_haystack(hybrid.forward);
        let frames = reserve(
            FRAME,
            1 + (h + 1) * (hybrid.forward.surplus + hybrid.forward.capture_states),
        );
        assert!(backtrack_heap(hybrid.forward) >= frames);
    }

    #[test]
    fn plan_rows_are_reported_for_every_production_pool_capacity() {
        for stacks in [1, 4, 8, 16, 32] {
            let plan = plan(stacks).unwrap();
            println!(
                "P={} retained={} split={} byte_level_global={} bpe={} ctor={} boot={} boot_peak={} growth={} vm={} split_program={}",
                plan.pool_stacks,
                plan.retained_bytes,
                plan.split_retained_bytes,
                plan.byte_level_retained_bytes,
                plan.bpe_retained_bytes,
                plan.bpe_constructor_peak_bytes,
                plan.bootstrap_bytes,
                plan.bootstrap_peak_bytes,
                plan.request_cache_growth_bytes,
                plan.serialized_vm_bytes,
                plan.split_program_bytes
            );
        }
        let plan = plan(16).unwrap();
        println!(
            "P={} retained={} split={} byte_level_global={} bpe={} ctor={} boot={} boot_peak={} growth={} vm={} split_program={}",
            plan.pool_stacks,
            plan.retained_bytes,
            plan.split_retained_bytes,
            plan.byte_level_retained_bytes,
            plan.bpe_retained_bytes,
            plan.bpe_constructor_peak_bytes,
            plan.bootstrap_bytes,
            plan.bootstrap_peak_bytes,
            plan.request_cache_growth_bytes,
            plan.serialized_vm_bytes,
            plan.split_program_bytes
        );
        let plan = super::plan(16).unwrap();
        println!(
            "P=16 MiB retained={:.1} split={:.1} byte_level_global={:.1} bpe={:.1} ctor={:.1} boot={:.1} prompt(47tok/512B)={:.1} decode(8)={:.3}",
            plan.retained_bytes as f64 / MIB as f64,
            plan.split_retained_bytes as f64 / MIB as f64,
            plan.byte_level_retained_bytes as f64 / MIB as f64,
            plan.bpe_retained_bytes as f64 / MIB as f64,
            plan.bpe_constructor_peak_bytes as f64 / MIB as f64,
            plan.bootstrap_bytes as f64 / MIB as f64,
            prompt_peak(512).unwrap() as f64 / MIB as f64,
            decode_peak(8).unwrap() as f64 / MIB as f64,
        );
        for delegate in SPLIT_DELEGATES.iter().filter(|d| d.core) {
            println!(
                "split insn={} dense={} hybrid={} heap={} pool16={}",
                delegate.insn,
                delegate.dense.0,
                delegate.stride2,
                cache_heap(delegate),
                pool_bytes(delegate, 16)
            );
        }
    }

    #[test]
    fn whole_profile_fits_the_4096_mib_host_at_p16_with_active_eight() {
        let plan = plan(16).unwrap();
        assert_eq!(plan.pool_stacks, 16);
        assert!(plan.split_retained_bytes > plan.bpe_retained_bytes);
        assert!(plan.bpe_constructor_peak_bytes > 0);
        assert!(
            plan.bpe_constructor_peak_bytes < 4096 * MIB,
            "source fallback constructor envelope {} B must fit the 4 GiB host",
            plan.bpe_constructor_peak_bytes
        );
        assert!(plan.request_cache_growth_bytes > 0);
        // The production profile is 47 prompt tokens / 8 output tokens. This
        // 512-byte rendered bound is a conservative priced envelope, not a
        // claim that token count implies 512 bytes; the source fixture's actual
        // prompt byte lengths are measured by the tokenizer source test.
        // `fits` charges the source subtotal (`bootstrap + R + G + V`) once and
        // only admitted prompt (without `V`) / decode per request.
        assert!(
            fits(&plan, 4096, 512, 8),
            "P16 profile must fit 4096 MiB: retained {} MiB, prompt {} MiB",
            plan.retained_bytes / MIB,
            prompt_peak(512).unwrap() / MIB
        );
    }

    #[test]
    fn serialized_vm_is_a_named_checked_plan_row_and_is_not_per_request() {
        let plan = plan(16).unwrap();
        assert_eq!(plan.serialized_vm_bytes, 224_000_144);
        assert_eq!(
            admitted_prompt_peak(512)
                .unwrap()
                .checked_add(plan.serialized_vm_bytes)
                .unwrap(),
            prompt_peak(512).unwrap(),
            "generic prompt peak is admitted input plus the prepaid VM row"
        );
        assert_eq!(prompt_peak(MAX_PRICED_PROMPT_BYTES + 1), None);
        assert_eq!(admitted_prompt_peak(MAX_PRICED_PROMPT_BYTES + 1), None);
    }

    #[test]
    fn request_phase_is_monotone_in_the_rendered_length() {
        let small = prompt_peak(1 << 10).unwrap();
        let large = prompt_peak(1 << 20).unwrap();
        assert!(large > small);
        assert!(decode_peak(512).unwrap() > decode_peak(1).unwrap());
        assert!(decode_peak(0).unwrap() < decode_peak(1).unwrap());
        assert_eq!(prompt_peak(MAX_PRICED_PROMPT_BYTES + 1), None);
        assert_eq!(decode_peak(MAX_PRICED_DECODE_TOKENS + 1), None);
        assert!(plan(MAX_POOL_STACKS + 1).is_err());
    }

    #[test]
    fn constructor_tree_is_three_census_rows_plus_census_scratch() {
        let census_tree = 1_234_567;
        let source_len = APPROVED_TOKENIZER_LEN;
        let scratch = census_scratch_peak(source_len as usize).unwrap() as u64;
        assert_eq!(
            constructor_tree_bytes(census_tree, source_len).unwrap(),
            3 * census_tree + scratch
        );
        // A checked constructor must refuse rather than wrap a huge tree.
        assert!(constructor_tree_bytes(u64::MAX / 2, source_len).is_err());
        assert!(checked_regrow(u64::MAX / 2, u64::MAX).is_err());
    }

    #[test]
    fn merge_children_are_priced_as_independent_two_element_vectors() {
        // Each `[left, right]` child is a separate RawVec, not one aggregate
        // `Vec<Content>` of `2*m` elements. The fragment row records exactly
        // that independently allocated minimum capacity.
        let one_child = checked_regrow(CONTENT, 2).unwrap();
        for children in 1..=16 {
            assert_eq!(
                fragment_bytes(CONTENT, 2 * children, children).unwrap(),
                one_child * children
            );
        }
        assert_eq!(one_child, 256);
    }

    #[test]
    fn aho_geometry_and_prefilter_follow_the_fixed_equations() {
        let pattern_bytes = ADDED_TOKENS * MAX_ADDED_ENTRY_BYTES;
        assert_eq!(pattern_bytes, 520);
        assert_eq!(
            aho_shape(ADDED_TOKENS, pattern_bytes).unwrap(),
            (524, 134_145, 13_625)
        );
        // `Arc32(align32(8+16*24+4*2*32))` = Arc32(672) = 704 B.
        assert_eq!(checked_arc(32, 672).unwrap(), 704);
        assert!(aho_dfa_retained(ADDED_TOKENS, pattern_bytes).unwrap() > 0);
        assert!(aho_dfa_retained(0, 0).unwrap() > 0);
        assert!(
            split_automata_peak().unwrap() > split_automata_retained().unwrap(),
            "constructor peak includes retained automata plus build rows"
        );
    }

    #[test]
    fn compiler_rows_derive_from_literal_counts_not_a_fitted_envelope() {
        let counts = syntax_counts(SPLIT_LITERAL_BYTES).unwrap();
        assert_eq!((counts.b, counts.j, counts.f), (1_968, 7_888, 252));
        let syntax = syntax_scratch(SPLIT_LITERAL_BYTES).unwrap();
        let thompson = thompson_scratch(SPLIT_LITERAL_BYTES).unwrap();
        // Thompson includes two independent 6*B1 envelopes plus the 256-bucket
        // UTF-8 compiler scratch; the literal counts are visible in syntax.
        assert!(syntax > 0);
        assert!(thompson > 2 * 6 * (10 << 20));
        let dense = dense_scratch(336, 128).unwrap();
        assert!(dense > 10_240);
        assert!(
            compiler_scratch(SPLIT_LITERAL_BYTES, &SPLIT_DELEGATES, SPLIT_INSTRUCTIONS).unwrap()
                >= thompson + syntax
        );
    }
}
