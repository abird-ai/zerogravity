//! One typed binding order for CUDA source signatures, checked handle binding,
//! and PinnedKernel byte/access metadata. No device action or semantic permit.
use super::*;
use BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    GdnSource(usize),
    GdnDestination(usize),
    PriorK(usize),
    PriorV(usize),
    NativeCache,
    CandidateK(usize),
    CandidateV(usize),
    Base(usize),
    End(usize),
    Block(usize),
    Status,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Arg {
    slot: Slot,
    pub bytes: usize,
    pub access: BufferIOAttr,
}

impl Arg {
    const fn new(slot: Slot, bytes: usize, access: BufferIOAttr) -> Self {
        Self {
            slot,
            bytes,
            access,
        }
    }

    /// All row field populations were already checked by TypedRow::require_combination.
    pub(super) fn handle<'a>(
        &self,
        rows: &'a [TypedRow<'a>],
        status: &'a Handle,
    ) -> Result<&'a Handle> {
        let row = |index| {
            rows.get(index)
                .ok_or_else(|| Error::Malformed("typed ABI row out of range".into()))
        };
        let missing = || Error::State("typed ABI required row field absent".into());
        Ok(match self.slot {
            Slot::GdnSource(i) => {
                let r = row(i)?;
                r.prior.unwrap_or(r.native)
            }
            Slot::GdnDestination(i) => {
                let r = row(i)?;
                r.candidate.unwrap_or(r.native)
            }
            Slot::PriorK(i) => row(i)?.prior.ok_or_else(missing)?,
            Slot::PriorV(i) => {
                let r = row(i)?;
                r.second_prior.or(r.second).ok_or_else(missing)?
            }
            Slot::NativeCache => row(0)?.native,
            Slot::CandidateK(i) => row(i)?.candidate.ok_or_else(missing)?,
            Slot::CandidateV(i) => row(i)?.second.ok_or_else(missing)?,
            Slot::Base(i) => row(i)?.controls.ok_or_else(missing)?[0],
            Slot::End(i) => row(i)?.controls.ok_or_else(missing)?[1],
            Slot::Block(i) => row(i)?.controls.ok_or_else(missing)?[2],
            Slot::Status => status,
        })
    }

    #[cfg(test)]
    fn c_declaration(self) -> (String, &'static str) {
        match self.slot {
            Slot::GdnSource(i) => (format!("s{i}"), "const ZG_INPUT*"),
            Slot::GdnDestination(i) => (format!("d{i}"), "ZG_OUTPUT*"),
            Slot::PriorK(i) => (format!("pk{i}"), "const unsigned int*"),
            Slot::PriorV(i) => (format!("pv{i}"), "const unsigned int*"),
            Slot::NativeCache => ("native_cache".into(), "unsigned short*"),
            Slot::CandidateK(i) => (format!("ck{i}"), "unsigned int*"),
            Slot::CandidateV(i) => (format!("cv{i}"), "unsigned int*"),
            Slot::Base(i) => (format!("base{i}"), "const long long*"),
            Slot::End(i) => (format!("end{i}"), "const int*"),
            Slot::Block(i) => (format!("block{i}"), "const int*"),
            Slot::Status => ("status".into(), "unsigned int*"),
        }
    }
}

// B4 native export has at most 30 pointer slots. A fixed stack descriptor
// avoids new heap allocations during every hot token/transport binding.
// Native kernel metadata receives the same descriptor once at cold prepare.
const MAX_ARGS: usize = 30;
pub(super) struct Abi {
    pub entry: &'static str,
    args: [Arg; MAX_ARGS],
    len: usize,
}
impl Abi {
    fn new(entry: &'static str) -> Self {
        Self {
            entry,
            args: [Arg::new(Slot::Status, 4, R); MAX_ARGS],
            len: 0,
        }
    }
    fn push(&mut self, slot: Slot, bytes: usize, access: BufferIOAttr) -> Result<()> {
        let next = self.args.get_mut(self.len).ok_or_else(|| {
            Error::Limit("grouped B4 ABI exceeds its certified 30-argument capacity".into())
        })?;
        *next = Arg::new(slot, bytes, access);
        self.len += 1;
        Ok(())
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = &Arg> {
        self.args[..self.len].iter()
    }
}

pub(super) fn gdn(kind: usize) -> Result<Abi> {
    let entry = *[
        "zg_cohort_conv_in_rows",
        "zg_cohort_conv_out_rows",
        "zg_cohort_h_in_rows",
        "zg_cohort_h_out_rows",
    ]
    .get(kind)
    .ok_or_else(|| Error::Unsupported("grouped GDN kernel kind".into()))?;
    let direction = if kind.is_multiple_of(2) {
        TypedDirection::Import
    } else {
        TypedDirection::Export
    };
    let layout = if kind < 2 {
        TypedLayout::Conv
    } else {
        TypedLayout::RecurrentH
    };
    let (source, destination) = gdn_extents(layout, direction)?;
    let mut args = Abi::new(entry);
    for i in 0..ROWS {
        args.push(Slot::GdnSource(i), source, R)?;
    }
    for i in 0..ROWS {
        args.push(Slot::GdnDestination(i), destination, W)?;
    }
    args.push(Slot::Status, 4, RW)?;
    Ok(args)
}

pub(super) fn kv(incoming: bool) -> Result<Abi> {
    let mut args = Abi::new(if incoming {
        "zg_cohort_kv_in_rows"
    } else {
        "zg_cohort_kv_out_rows"
    });
    for i in 0..ROWS {
        args.push(Slot::PriorK(i), KV_CANONICAL_BYTES, R)?;
    }
    for i in 0..ROWS {
        args.push(Slot::PriorV(i), KV_CANONICAL_BYTES, R)?;
    }
    args.push(
        Slot::NativeCache,
        KV_CACHE_BYTES,
        if incoming { RW } else { R },
    )?;
    if !incoming {
        for i in 0..ROWS {
            args.push(Slot::CandidateK(i), KV_CANONICAL_BYTES, W)?;
        }
        for i in 0..ROWS {
            args.push(Slot::CandidateV(i), KV_CANONICAL_BYTES, W)?;
        }
    }
    // CUDA signature: row-major base/end/block (8/4/4 bytes) for each row.
    for i in 0..ROWS {
        args.push(Slot::Base(i), 8, R)?;
        args.push(Slot::End(i), 4, R)?;
        args.push(Slot::Block(i), 4, R)?;
    }
    args.push(Slot::Status, 4, RW)?;
    Ok(args)
}

pub(super) fn metadata(args: &Abi) -> Arc<[(usize, BufferIOAttr)]> {
    args.iter().map(|arg| (arg.bytes, arg.access)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The CUDA signature remains literal native source. Inspect its complete
    // preprocessor-selected argument list against the exact Rust ABI roles.
    // This source-only test is independent from the GPU binary's binding checks.
    fn cuda_parameters(source: &str, incoming: Option<bool>) -> Vec<(String, String)> {
        let header = source
            .split_once("void ZG_ENTRY(")
            .expect("native CUDA entry")
            .1
            .split_once(") {")
            .expect("native CUDA parameter closing")
            .0;
        let mut omit = false;
        let mut joined = String::new();
        for line in header.lines() {
            match line.trim() {
                "#if !ZG_IN" => {
                    omit = incoming == Some(true);
                    continue;
                }
                "#endif" => {
                    omit = false;
                    continue;
                }
                _ => {}
            }
            if !omit {
                joined.push(' ');
                joined.push_str(line.trim());
            }
        }
        joined
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                let s = s.trim();
                let (kind, name) = s.rsplit_once(' ').expect("typed CUDA parameter");
                (name.to_string(), kind.trim().to_string())
            })
            .collect()
    }

    #[test]
    fn native_cuda_source_signatures_match_the_single_binding_table() {
        for kind in 0..4 {
            let args = gdn(kind).unwrap();
            let declared = cuda_parameters(STATE_SOURCE, None);
            let expected = args
                .iter()
                .copied()
                .map(Arg::c_declaration)
                .map(|(name, kind)| (name, kind.into()))
                .collect::<Vec<_>>();
            assert_eq!(declared, expected, "GDN kind {kind}");
        }
        for incoming in [true, false] {
            let declared = cuda_parameters(KV_SOURCE, Some(incoming));
            let expected = kv(incoming)
                .unwrap()
                .iter()
                .copied()
                .map(Arg::c_declaration)
                .map(|(name, kind)| (name, kind.into()))
                .collect::<Vec<_>>();
            assert_eq!(declared, expected, "KV import={incoming}");
        }
    }
}
