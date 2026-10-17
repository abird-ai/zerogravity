//! Additive full-path Unicode FnmatchCaseV1 selection; legacy Files is unchanged.

use super::candidates::{CandidateRequest, MetadataCalls};
use super::http::{HttpTransport, check_cancel};
use super::quant::GgufQuantRequest;
use super::{ExternalSidecars, Provider, PullLimits, ResolvedPullPlan};
use crate::cache::{FileRole, Manifest, SnapshotLease, TransferAdmission};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;

const MAX_PATTERNS: usize = 4096;
const MAX_PATTERN_BYTES: usize = 4096;
const MAX_INPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_WORK: usize = 16 * 1024 * 1024;
pub(super) const CONTROL_BYTES: u64 = 16 * 1024 * 1024;

/// Full-path Unicode-scalar, case-exact FnmatchCaseV1. Slash and initial dots
/// are ordinary characters; backslash is literal and malformed '[' is literal.
/// This additive selection never changes the old Files byte/component grammar.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FilterRequest {
    source: CandidateRequest,
    quant: Option<GgufQuantRequest>,
    includes: Vec<String>,
    excludes: Vec<String>,
}
impl FilterRequest {
    pub fn new(
        source: CandidateRequest,
        includes: Vec<String>,
        excludes: Vec<String>,
    ) -> Result<Self> {
        if includes.is_empty() {
            return Err(Error::Malformed("filter includes are required".into()));
        }
        validate_patterns(&includes, &excludes)?;
        Ok(Self {
            source,
            quant: None,
            includes,
            excludes,
        })
    }
    pub fn for_quant(quant: GgufQuantRequest, excludes: Vec<String>) -> Result<Self> {
        validate_patterns(&[], &excludes)?;
        Ok(Self {
            source: quant.source().clone(),
            quant: Some(quant),
            includes: Vec::new(),
            excludes,
        })
    }
    pub fn source(&self) -> &CandidateRequest {
        &self.source
    }
    pub fn includes(&self) -> &[String] {
        &self.includes
    }
    pub fn excludes(&self) -> &[String] {
        &self.excludes
    }
    pub fn quant_request(&self) -> Option<&GgufQuantRequest> {
        self.quant.as_ref()
    }
    fn relative<'a>(&self, path: &'a str) -> Option<&'a str> {
        match self.source.subfolder() {
            None => Some(path),
            Some(folder) => path.strip_prefix(folder)?.strip_prefix('/'),
        }
    }
}
fn bound() -> Error {
    Error::Limit("filter work or input bounds exceeded".into())
}
fn validate_patterns(includes: &[String], excludes: &[String]) -> Result<usize> {
    if includes
        .len()
        .checked_add(excludes.len())
        .is_none_or(|n| n > MAX_PATTERNS)
    {
        return Err(bound());
    }
    let mut bytes = 0usize;
    for pattern in includes.iter().chain(excludes) {
        if pattern.is_empty()
            || pattern.len() > MAX_PATTERN_BYTES
            || pattern.chars().any(char::is_control)
        {
            return Err(Error::Malformed("invalid filter pattern".into()));
        }
        bytes = bytes
            .checked_add(pattern.len())
            .filter(|n| *n <= MAX_INPUT_BYTES)
            .ok_or_else(bound)?;
    }
    Ok(bytes)
}
struct Work<'a> {
    left: usize,
    cancel: &'a AtomicBool,
}
impl<'a> Work<'a> {
    fn new(cancel: &'a AtomicBool) -> Self {
        Self {
            left: MAX_WORK,
            cancel,
        }
    }
    fn charge(&mut self, count: usize) -> Result<()> {
        let before = self.left;
        self.left = self.left.checked_sub(count).ok_or_else(bound)?;
        if count >= 4096 || before / 4096 != self.left / 4096 || before == MAX_WORK {
            check_cancel(self.cancel)?;
        }
        Ok(())
    }
}
#[derive(Debug)]
enum Token {
    Literal(char),
    Any,
    Star,
    Class {
        negative: bool,
        ranges: Vec<(char, char)>,
    },
}
struct Pattern {
    tokens: Vec<Token>,
}
impl Pattern {
    fn compile(pattern: &str, work: &mut Work<'_>) -> Result<Self> {
        if pattern.len() > MAX_PATTERN_BYTES {
            return Err(bound());
        }
        work.charge(pattern.len())?;
        let chars: Vec<_> = pattern.chars().collect();
        let mut tokens = Vec::with_capacity(chars.len());
        let mut i = 0;
        while i < chars.len() {
            work.charge(1)?;
            let c = chars[i];
            i += 1;
            match c {
                '*' => {
                    if !matches!(tokens.last(), Some(Token::Star)) {
                        tokens.push(Token::Star);
                    }
                }
                '?' => tokens.push(Token::Any),
                '[' => {
                    let mut end = i;
                    if chars.get(end) == Some(&'!') {
                        end += 1;
                    }
                    if chars.get(end) == Some(&']') {
                        end += 1;
                    }
                    while end < chars.len() && chars[end] != ']' {
                        work.charge(1)?;
                        end += 1;
                    }
                    if end == chars.len() {
                        tokens.push(Token::Literal('['));
                    } else {
                        tokens.push(Self::class(&chars[i..end], work)?);
                        i = end + 1;
                    }
                }
                _ => tokens.push(Token::Literal(c)),
            }
        }
        Ok(Self { tokens })
    }
    // Same range normalization as CPython v3.14.0, represented as scalar
    // intervals instead of regex. Literal hyphens are distinct from connectors.
    fn class(chars: &[char], work: &mut Work<'_>) -> Result<Token> {
        let mut chunks = Vec::<Vec<char>>::new();
        let mut start = 0usize;
        let mut search = if chars.first() == Some(&'!') { 2 } else { 1 };
        while search < chars.len() {
            work.charge(1)?;
            if chars[search] == '-' {
                work.charge(search - start)?;
                chunks.push(chars[start..search].to_vec());
                start = search + 1;
                search += 3;
            } else {
                search += 1;
            }
        }
        if start < chars.len() {
            work.charge(chars.len() - start)?;
            chunks.push(chars[start..].to_vec());
        } else if let Some(last) = chunks.last_mut() {
            last.push('-');
        }
        for at in (1..chunks.len()).rev() {
            work.charge(1)?;
            if chunks[at - 1].last() > chunks[at].first() {
                work.charge(chunks[at].len() + chunks.len() - at)?;
                let tail = chunks.remove(at);
                chunks[at - 1].pop();
                chunks[at - 1].extend_from_slice(&tail[1..]);
            }
        }
        let mut units = Vec::<(char, bool)>::new();
        for (index, chunk) in chunks.iter().enumerate() {
            work.charge(chunk.len() + 1)?;
            if index != 0 {
                units.push(('-', true));
            }
            units.extend(chunk.iter().map(|ch| (*ch, false)));
        }
        let negative = units.first() == Some(&('!', false));
        let mut i = usize::from(negative);
        let mut ranges = Vec::new();
        while i < units.len() {
            work.charge(1)?;
            if i + 2 < units.len() && units[i + 1].1 {
                ranges.push((units[i].0, units[i + 2].0));
                i += 3;
            } else {
                ranges.push((units[i].0, units[i].0));
                i += 1;
            }
        }
        Ok(Token::Class { negative, ranges })
    }
    fn epsilon(&self, row: &mut [bool], work: &mut Work<'_>) -> Result<()> {
        for (i, token) in self.tokens.iter().enumerate() {
            work.charge(1)?;
            if matches!(token, Token::Star) {
                row[i + 1] |= row[i];
            }
        }
        Ok(())
    }
    fn matches(&self, name: &str, work: &mut Work<'_>) -> Result<bool> {
        work.charge(self.tokens.len() + 1)?;
        let mut current = vec![false; self.tokens.len() + 1];
        let mut next = vec![false; current.len()];
        current[0] = true;
        self.epsilon(&mut current, work)?;
        for ch in name.chars() {
            next.fill(false);
            for (i, token) in self.tokens.iter().enumerate() {
                work.charge(1)?;
                if !current[i] {
                    continue;
                }
                let yes = match token {
                    Token::Star => {
                        next[i] = true;
                        false
                    }
                    Token::Any => true,
                    Token::Literal(literal) => *literal == ch,
                    Token::Class { negative, ranges } => {
                        let mut found = false;
                        for (low, high) in ranges {
                            work.charge(1)?;
                            found |= *low <= ch && ch <= *high;
                        }
                        found != *negative
                    }
                };
                next[i + 1] |= yes;
            }
            self.epsilon(&mut next, work)?;
            std::mem::swap(&mut current, &mut next);
        }
        Ok(current[self.tokens.len()])
    }
}
/// The only retained matcher result needed by closure. Bounded to4096 keys and
/// <=8MiB original path bytes. Together with <=512KiB scratch and tree node/key
/// overhead this fits the conservative extra16MiB admitted control envelope.
#[derive(Default)]
pub(super) struct Exclusions {
    paths: BTreeSet<String>,
}
impl Exclusions {
    pub(super) fn allows(&self, path: &str) -> bool {
        !self.paths.contains(path)
    }
    pub(super) fn required(&self, path: &str) -> Result<()> {
        if !self.allows(path) {
            return Err(Error::Unsupported(
                "filter excludes a required shard, index or sidecar".into(),
            ));
        }
        Ok(())
    }
}
fn union(
    patterns: &[String],
    entries: &BTreeMap<String, bool>,
    request: &FilterRequest,
    regular_only: bool,
    work: &mut Work<'_>,
) -> Result<BTreeSet<String>> {
    let mut matched = BTreeSet::new();
    for pattern in patterns {
        let compiled = Pattern::compile(pattern, work)?;
        for (path, regular) in entries {
            work.charge(1)?;
            if regular_only && !regular || matched.contains(path) {
                continue;
            }
            if let Some(relative) = request.relative(path)
                && compiled.matches(relative, work)?
            {
                matched.insert(path.clone());
            }
        }
    }
    Ok(matched)
}
pub(super) fn select(
    entries: &BTreeMap<String, bool>,
    request: &FilterRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
) -> Result<(BTreeSet<String>, Exclusions)> {
    super::quant::validate_inventory(entries, limits, cancel)?;
    let mut bytes = validate_patterns(&request.includes, &request.excludes)?;
    for path in entries.keys() {
        check_cancel(cancel)?;
        bytes = bytes
            .checked_add(path.len())
            .filter(|n| *n <= MAX_INPUT_BYTES)
            .ok_or_else(bound)?;
    }
    let mut work = Work::new(cancel);
    let exclusions = Exclusions {
        paths: union(&request.excludes, entries, request, false, &mut work)?,
    };
    let selected = if let Some(quant) = &request.quant {
        super::quant::select_excluding(entries, quant, limits, cancel, &exclusions)?
    } else {
        let mut selected = union(&request.includes, entries, request, true, &mut work)?;
        selected.retain(|path| exclusions.allows(path));
        selected
    };
    if selected.is_empty() {
        return Err(Error::Unsupported(
            "filter selection is empty; no fallback".into(),
        ));
    }
    check_cancel(cancel)?;
    Ok((selected, exclusions))
}
pub fn resolve(
    transport: &impl HttpTransport,
    request: &FilterRequest,
    limits: &PullLimits,
    admission: &TransferAdmission,
    cancel: &AtomicBool,
) -> Result<ResolvedPullPlan> {
    check_cancel(cancel)?;
    let _control = super::quant::metadata_admission_extra(limits, admission, CONTROL_BYTES)?;
    let transport = MetadataCalls::new(transport, limits.max_pages);
    match request.source.provider() {
        Provider::HuggingFace => super::huggingface::resolve_filtered_from(
            &transport,
            request,
            limits,
            cancel,
            "https://huggingface.co/",
        ),
        Provider::GitHub => super::github::resolve_filtered(&transport, request, limits, cancel),
    }
}
/// Eligibility of exactly the retained closed snapshot, not a filtered view or
/// proof of remote-global uniqueness, freshness, authorization or native Xet.
pub fn validate_cached_selection(
    lease: &SnapshotLease,
    request: &FilterRequest,
    limits: &PullLimits,
    cancel: &AtomicBool,
) -> Result<()> {
    limits.validate()?;
    if limits.unknown_length_cap.is_some() {
        return Err(Error::Unsupported(
            "filters require exact source lengths".into(),
        ));
    }
    check_cancel(cancel)?;
    super::components::validate_cached_file_source(
        lease,
        request.source.provider(),
        request.source.repository(),
        request.source.revision(),
    )?;
    Manifest::new(
        lease.manifest().source().clone(),
        lease.manifest().entries().to_vec(),
        &limits.cache,
    )?;
    super::executor::reject_cached_pointers(lease, cancel)?;
    let entries = lease
        .manifest()
        .entries()
        .iter()
        .map(|e| (e.path().to_owned(), true))
        .collect();
    let (mut selected, exclusions) = select(&entries, request, limits, cancel)?;
    let mut metadata = limits.max_metadata_bytes;
    let mut pages = limits.max_pages;
    super::selection::close_model_selection_filtered(
        &entries,
        &mut selected,
        limits,
        ExternalSidecars::default(),
        &exclusions,
        &mut |path| {
            check_cancel(cancel)?;
            pages = pages.checked_sub(1).ok_or_else(bound)?;
            let entry = lease
                .manifest()
                .entries()
                .iter()
                .find(|e| e.path() == path)
                .ok_or_else(|| Error::Malformed("cached filter index absent".into()))?;
            let size = usize::try_from(entry.bytes())
                .ok()
                .filter(|n| *n <= metadata)
                .ok_or_else(bound)?;
            metadata -= size;
            let bytes = lease.read_verified(path, size as u64)?;
            check_cancel(cancel)?;
            Ok(bytes)
        },
    )?;
    if selected.len() != lease.manifest().entries().len()
        || lease
            .manifest()
            .entries()
            .iter()
            .any(|e| !selected.contains(e.path()))
    {
        return Err(Error::Unsupported(
            "cached snapshot contains files outside the requested filter closure".into(),
        ));
    }
    for entry in lease.manifest().entries() {
        check_cancel(cancel)?;
        let expected = if entry.path().ends_with(".gguf") || entry.path().ends_with(".safetensors")
        {
            Some(FileRole::Weights)
        } else if entry.path().ends_with(".safetensors.index.json") {
            Some(FileRole::Index)
        } else {
            match entry.path().rsplit('/').next().unwrap_or(entry.path()) {
                "config.json" => Some(FileRole::Config),
                "tokenizer.json" => Some(FileRole::Tokenizer),
                _ => None,
            }
        };
        if expected.is_some_and(|role| role != entry.role()) {
            return Err(Error::Malformed(
                "cached filter required role mismatch".into(),
            ));
        }
    }
    check_cancel(cancel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    // Frozen before Rust implementation from CPython v3.14.0 Lib/fnmatch.py.
    // Oracle Python3.13 only adapts strict-end spelling \z to synonymous \Z.
    #[test]
    fn pinned_fnmatchcase_class_unicode_and_path_literals() {
        let cases = [
            ("?", "é", true),
            ("??", "é", false),
            ("*", "a/b", true),
            ("a?b", "a/b", true),
            ("a*b", "a/b", true),
            ("**", "a/b", true),
            (".*", ".hidden", true),
            ("*", ".hidden", true),
            ("A*", "abc", false),
            ("é", "é", false),
            ("[a-z]", "m", true),
            ("[a-z]", "A", false),
            ("[!a-z]", "é", true),
            ("[!a-z]", "a", false),
            ("[é-ê]", "ê", true),
            ("[é-ê]", "e", false),
            ("[^a]", "^", true),
            ("[^a]", "b", false),
            ("[]a]", "]", true),
            ("[]a]", "a", true),
            ("[!]]", "a", true),
            ("[!]]", "]", false),
            ("[a-]", "-", true),
            ("[-a]", "-", true),
            ("[a-]", "a", true),
            ("[z-a]", "a", false),
            ("[!z-a]", "x", true),
            ("[!z-a]", "/", true),
            ("[a--b]", "b", true),
            ("[a--b]", "a", false),
            ("[a--b]", "-", false),
            ("[a-b-c]", "-", true),
            ("[a-b-c]", "c", true),
            ("[a-b-c]", "b", true),
            ("[a-b-c]", "d", false),
            ("[a-b-c-d]", "d", true),
            ("[a-b-c-d]", "-", true),
            ("[!--]", "a", true),
            ("[!--]", "-", false),
            ("[--a]", "_", true),
            ("[--a]", "b", false),
            ("[a&&b]", "&", true),
            ("[a~~b]", "~", true),
            ("[a||b]", "|", true),
            ("[[]", "[", true),
            ("[[]a", "[a", true),
            ("[abc", "[abc", true),
            ("[!abc", "[!abc", true),
            ("[]", "[]", true),
            ("[!]", "[!]", true),
            ("a\\b", "a\\b", true),
            ("[\\]", "\\", true),
            ("[*]", "*", true),
            ("[?]", "?", true),
            ("a[[]b", "a[b", true),
            ("[[]*", "[anything", true),
            ("a/b", "a/b", true),
            ("a/*", "a/b/c", true),
            ("[!a]", "/", true),
            ("[/]", "/", true),
            ("[a-b-c-d-e]", "e", true),
            ("[z-a-c]", "-", true),
            ("[z-a-c]", "c", true),
            ("[a-z-a]", "-", true),
            ("[a-z-a]", "a", true),
            ("[a--c-e]", "b", false),
            ("[a--c-e]", "e", true),
            ("[!a--b]", "b", false),
            ("[!a--b]", "x", true),
            ("[---]", "-", true),
            ("[---]", ".", false),
            ("[!-]", "-", false),
            ("[!-]", "x", true),
        ];
        let cancel = AtomicBool::new(false);
        let mut work = Work::new(&cancel);
        for (pattern, name, expected) in cases {
            let compiled = Pattern::compile(pattern, &mut work).unwrap();
            assert_eq!(
                compiled.matches(name, &mut work).unwrap(),
                expected,
                "{pattern:?} {name:?}"
            );
        }
    }
    #[test]
    fn bounded_exhaustive_short_patterns_match_frozen_reference_digest() {
        fn words(alphabet: &[char], maximum: usize) -> Vec<String> {
            let mut all = vec![String::new()];
            let mut previous = all.clone();
            for _ in 0..maximum {
                let mut next = Vec::new();
                for prefix in &previous {
                    for ch in alphabet {
                        next.push(format!("{prefix}{ch}"));
                    }
                }
                all.extend(next.iter().cloned());
                previous = next;
            }
            all
        }
        let patterns = words(&['a', 'b', '[', ']', '!', '-', '*', '?'], 3);
        let names = words(&['a', 'b', ']', '-', '!', 'é', '/'], 2);
        assert_eq!(patterns.len() * names.len(), 33345);
        let cancel = AtomicBool::new(false);
        let mut work = Work::new(&cancel);
        let mut hash = Sha256::new();
        for pattern in &patterns {
            let compiled = Pattern::compile(pattern, &mut work).unwrap();
            for name in &names {
                hash.update([u8::from(compiled.matches(name, &mut work).unwrap())]);
            }
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            "c9f3b0c631f703ae44a4f2dafb5c8c71cb2715bfaa3249751cd299c73a40445f"
        );
    }
}

#[cfg(test)]
mod adversarial {
    use super::*;

    #[test]
    fn shared_budget_exhaustion_is_limit_not_no_match_or_recursion() {
        let cancel = AtomicBool::new(false);
        let mut work = Work::new(&cancel);
        assert!(work.charge(MAX_WORK - 1).is_ok());
        assert!(work.charge(2).is_err());
        let names: BTreeMap<_, _> = (0..256)
            .map(|i| (format!("{i:04}{}", "a".repeat(64)), true))
            .collect();
        let request = FilterRequest::new(
            CandidateRequest::new(Provider::HuggingFace, "owner/repo", "main", None).unwrap(),
            vec!["?".repeat(64); 4096],
            vec![],
        )
        .unwrap();
        assert!(matches!(
            select(&names, &request, &PullLimits::default(), &cancel),
            Err(Error::Limit(_))
        ));
    }
    #[test]
    fn cancellation_and_unclosed_class_work_are_bounded() {
        let cancelled = AtomicBool::new(true);
        let mut work = Work::new(&cancelled);
        assert!(Pattern::compile("[a-]", &mut work).is_err());
        let live = AtomicBool::new(false);
        let mut work = Work::new(&live);
        let pattern = Pattern::compile(&"[".repeat(MAX_PATTERN_BYTES), &mut work).unwrap();
        assert!(
            pattern
                .matches(&"[".repeat(4096), &mut work)
                .unwrap_err()
                .to_string()
                .contains("bounds")
        );
    }
}
