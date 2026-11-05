//! Borrowed source operations; admission, flights and roots remain caller-owned.
use super::{
    cold::{Cold, view},
    *,
};
use cubecl::{prelude::*, server::Handle};
use zerogravity_core::accelerator::program::{SourceCallMode, SourceGdnGeometry, SourceParameter};

pub(super) fn arg(h: &Handle, n: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), n) }
}
pub(super) fn grid(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(32), 1, 1)
}
pub(super) fn dim() -> CubeDim {
    CubeDim::new_1d(32)
}
impl Cold {
    pub(super) fn s(&self, index: usize, words: usize) -> Result<Handle> {
        view(
            &self.scratch[index],
            0,
            words
                .checked_mul(4)
                .ok_or_else(|| Error::Limit("source-call scratch words".into()))?,
        )
    }
    pub(super) fn table(&self, index: usize, bytes: usize) -> Result<Handle> {
        view(&self.tables[index], 0, bytes)
    }
    pub(super) fn trig_table(&self) -> Result<Handle> {
        self.table(2, std::mem::size_of_val(&crate::source_trig::cpu::TABLES))
    }
    pub(super) fn parameter(&self, p: &SourceParameter) -> Result<Handle> {
        if self.layout.parameters.get(&p.name) != Some(p) {
            return Err(Error::State(
                "source-call learned parameter descriptor changed".into(),
            ));
        }
        let n = p
            .shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(usize::try_from(*b).ok()?))
            .ok_or_else(|| Error::Limit("source-call parameter words".into()))?;
        view(
            self.parameters
                .get(&p.name)
                .ok_or_else(|| Error::State("source-call missing parameter owner".into()))?,
            0,
            n * 4,
        )
    }
    pub(super) fn status_view(&self) -> Result<Handle> {
        view(&self.status, 0, 4)
    }
    pub(super) fn zero_view(&self) -> Result<Handle> {
        view(&self.zero, 0, 4)
    }
    pub(super) fn rms(
        &self,
        input: &Handle,
        gamma: &Handle,
        output: &Handle,
        width: usize,
        groups: usize,
        mode: crate::source_rms::Gamma,
    ) -> Result<usize> {
        let (bytes, tiles) = rms_tiles(width, groups)?;
        // Validate the complete borrowed vectors before submitting any tile.
        subview(input, 0, bytes)?;
        subview(output, 0, bytes)?;
        let lengths =
            crate::source_rms::scratch_words(width, groups.min(crate::source_rms::MAX_GROUPS))?;
        let scratch = lengths
            .into_iter()
            .enumerate()
            .map(|(i, n)| self.s(layout::RMS.start + i, n))
            .collect::<Result<Vec<_>>>()?;
        let mut count = 0;
        for (offset, bytes, groups) in tiles {
            count += crate::source_rms::launch(
                self.service.client(),
                &scratch,
                &subview(input, offset, bytes)?,
                Some(gamma),
                &subview(output, offset, bytes)?,
                &self.status_view()?,
                &self.zero_view()?,
                width,
                groups,
                mode,
            )?;
        }
        Ok(count)
    }
    pub(super) fn sigmoid(
        &self,
        input: &Handle,
        output: &Handle,
        elements: usize,
    ) -> Result<usize> {
        let table = self.table(1, 72 * 8)?;
        let shifted = self.s(layout::SIGMOID_SHIFT, elements * 2)?;
        let status = self.status_view()?;
        unsafe {
            crate::source_sigmoid::call_shift::launch_unchecked(
                self.service.client(),
                grid(elements),
                dim(),
                arg(input, elements),
                arg(&table, 72),
                arg(&shifted, elements),
                elements as u32,
            );
            crate::source_sigmoid::call::launch_unchecked(
                self.service.client(),
                grid(elements),
                dim(),
                arg(input, elements),
                arg(&table, 72),
                arg(&shifted, elements),
                arg(output, elements),
                arg(&status, 1),
                elements as u32,
            );
        }
        Ok(2)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn(
        &self,
        activated: &Handle,
        gate: &Handle,
        ba: &Handle,
        prior: &Handle,
        next: &Handle,
        output: &Handle,
        parameters: &[SourceParameter],
        geometry: SourceGdnGeometry,
        rows: usize,
        mode: SourceCallMode,
    ) -> Result<usize> {
        let kh = geometry.key_heads;
        let vh = geometry.value_heads;
        let width = vh * 128;
        let channels = (2 * kh + vh) * 128;
        let n = rows * width;
        let coeff = rows * vh;
        let state = vh * 16384;
        let client = self.service.client();
        let status = self.status_view()?;
        let a = self.parameter(&parameters[0])?;
        let bias = self.parameter(&parameters[1])?;
        let gamma = self.parameter(&parameters[2])?;
        let normal = self.table(0, 256 * 4)?;
        let a_exp = self.s(layout::A_EXP, vh)?;
        let beta_input = self.s(layout::BETA_INPUT, coeff)?;
        let beta = self.s(layout::BETA, coeff)?;
        let decay_log = self.s(layout::LOG_DECAY, coeff)?;
        let core = self.s(layout::CORE, n)?;
        unsafe {
            crate::source_standalone_exp::normal_vector::launch_unchecked(
                client,
                grid(vh),
                dim(),
                arg(&a, vh),
                arg(&normal, 256),
                arg(&a_exp, vh),
                arg(&status, 1),
                vh,
            );
            super::kernels::coefficients::launch_unchecked(
                client,
                grid(coeff),
                dim(),
                arg(ba, coeff * 2),
                arg(&bias, vh),
                arg(&a_exp, vh),
                arg(&beta_input, coeff),
                arg(&decay_log, coeff),
                arg(&status, 1),
                rows as u32,
                vh,
            );
        }
        let mut count = 2 + self.sigmoid(&beta_input, &beta, coeff)?;
        match mode {
            SourceCallMode::Prefill => {
                let q = self.s(layout::QUERY, n)?;
                let k = self.s(layout::KEY, n)?;
                let v = self.s(layout::VALUE, n)?;
                unsafe {
                    super::kernels::repeat_qkv::launch_unchecked(
                        client,
                        grid(n),
                        dim(),
                        arg(activated, rows * channels),
                        arg(&q, n),
                        arg(&k, n),
                        arg(&v, n),
                        rows as u32,
                        kh,
                        vh,
                    );
                }
                let g = crate::source_gdn::chunk::execution::Geometry::new(rows, vh)?;
                let out = g
                    .output_words()
                    .into_iter()
                    .enumerate()
                    .map(|(i, n)| match i {
                        17 => Ok(next.clone()),
                        18 => Ok(core.clone()),
                        _ => self.s(layout::CHUNK.start + i, n),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let temp = g
                    .scratch_words()
                    .into_iter()
                    .enumerate()
                    .map(|(i, n)| self.s(layout::CHUNK_SCRATCH.start + i, n))
                    .collect::<Result<Vec<_>>>()?;
                let rare = self.table(4, 128 * 8)?;
                count += 1 + crate::source_gdn::chunk::execution::launch(
                    client,
                    g,
                    [&q, &k, &v, &beta, &decay_log, prior],
                    std::array::from_fn(|i| &out[i]),
                    [&temp[0], &temp[1]],
                    &normal,
                    &rare,
                    &status,
                )?;
            }
            SourceCallMode::Decode => {
                if rows != 1 {
                    return Err(Error::State("source-call cached GDN row count".into()));
                }
                let decay = self.s(layout::DECAY, vh)?;
                let q = self.s(layout::QUERY, width)?;
                let k = self.s(layout::KEY, width)?;
                let decayed = self.s(layout::DECAYED, state)?;
                let memory = self.s(layout::MEMORY, width)?;
                let delta = self.s(layout::DELTA, width)?;
                use crate::source_gdn::recurrent as r;
                unsafe {
                    crate::source_standalone_exp::normal_vector::launch_unchecked(
                        client,
                        grid(vh),
                        dim(),
                        arg(&decay_log, vh),
                        arg(&normal, 256),
                        arg(&decay, vh),
                        arg(&status, 1),
                        vh,
                    );
                    r::normalize_qk::launch_unchecked(
                        client,
                        grid(vh),
                        dim(),
                        arg(activated, channels),
                        arg(&q, width),
                        arg(&k, width),
                        arg(&status, 1),
                        kh,
                        vh,
                        true,
                    );
                    r::decayed::launch_unchecked(
                        client,
                        grid(state),
                        dim(),
                        arg(prior, state),
                        arg(&decay, vh),
                        arg(&decayed, state),
                        arg(&status, 1),
                        state,
                        false,
                    );
                    r::key_reduce::launch_unchecked(
                        client,
                        grid(width),
                        dim(),
                        arg(&decayed, state),
                        arg(&k, width),
                        arg(&memory, width),
                        arg(&status, 1),
                        width,
                        false,
                    );
                    r::delta::launch_unchecked(
                        client,
                        grid(width),
                        dim(),
                        arg(activated, channels),
                        arg(&memory, width),
                        arg(&beta, vh),
                        arg(&delta, width),
                        arg(&status, 1),
                        kh,
                        width,
                    );
                    r::state_update::launch_unchecked(
                        client,
                        grid(state),
                        dim(),
                        arg(&decayed, state),
                        arg(&k, width),
                        arg(&delta, width),
                        arg(activated, channels),
                        arg(&beta, vh),
                        arg(next, state),
                        arg(&status, 1),
                        kh,
                        state,
                        false,
                    );
                    r::key_reduce::launch_unchecked(
                        client,
                        grid(width),
                        dim(),
                        arg(next, state),
                        arg(&q, width),
                        arg(&core, width),
                        arg(&status, 1),
                        width,
                        true,
                    );
                }
                count += 7;
            }
        }
        let normalized = self.s(layout::NORMAL, n)?;
        let bf16_gamma = parameters[2].kind
            == zerogravity_core::encoding::StoredTensorEncoding::Scalar(
                zerogravity_core::encoding::ScalarDType::Bf16,
            );
        count += self.rms(
            &core,
            &gamma,
            &normalized,
            128,
            rows * vh,
            crate::source_rms::Gamma::DirectF32AfterBf16,
        )?;
        for row in 0..rows {
            let normalized_row = subview(&normalized, row * width * 4, width * 4)?;
            let gate_row = subview(gate, row * width * 4, width * 4)?;
            let output_row = subview(output, row * width * 4, width * 4)?;
            unsafe {
                crate::source_activation::gated::launch_unchecked(
                    client,
                    grid(width),
                    dim(),
                    arg(&normalized_row, width),
                    arg(&gate_row, width),
                    arg(&output_row, width),
                    arg(&status, 1),
                    width,
                    bf16_gamma,
                );
            }
            count += 1;
        }
        Ok(count)
    }
}
// Flatten only independent RMS groups; each launch retains the existing bound.
fn rms_tiles(
    width: usize,
    groups: usize,
) -> Result<(usize, impl Iterator<Item = (usize, usize, usize)>)> {
    let bound = crate::source_rms::MAX_GROUPS;
    crate::source_rms::scratch_words(width, groups.min(bound))?;
    let bytes = width
        .checked_mul(groups)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Limit("source-call RMS complete byte extent".into()))?;
    let tiles = (0..groups).step_by(bound).map(move |first| {
        let count = (groups - first).min(bound);
        (first * width * 4, count * width * 4, count)
    });
    Ok((bytes, tiles))
}

pub(super) fn subview(handle: &Handle, offset: usize, bytes: usize) -> Result<Handle> {
    let total = handle
        .size()
        .checked_sub(handle.offset_start.unwrap_or(0))
        .and_then(|n| n.checked_sub(handle.offset_end.unwrap_or(0)))
        .ok_or_else(|| Error::State("source-call parent view".into()))?;
    let end = offset
        .checked_add(bytes)
        .filter(|n| bytes > 0 && *n as u64 <= total)
        .ok_or_else(|| Error::State("source-call nested view extent".into()))?;
    Ok(handle
        .clone()
        .offset_start(offset as u64)
        .offset_end(total - end as u64))
}

#[cfg(test)]
mod tests {
    use super::rms_tiles;

    #[test]
    fn rms_tiles_cover_independent_head_rows_with_bounded_complete_groups() {
        for width in [32, 128, 256, 1024, 5120] {
            for rows in 1..=64 {
                for heads in [1, 2, 3, 4, 8, 16, 24, 32, 48, 64] {
                    let groups = rows * heads;
                    let (bytes, tiles) = rms_tiles(width, groups).unwrap();
                    let (mut offset, mut seen) = (0, 0);
                    for (start, length, count) in tiles {
                        assert_eq!(start, offset);
                        assert_eq!(length, count * width * 4);
                        assert!(crate::source_rms::scratch_words(width, count).is_ok());
                        offset += length;
                        seen += count;
                    }
                    assert_eq!(seen, groups);
                    assert_eq!(offset, bytes);
                    assert_eq!(bytes, groups * width * 4);
                }
            }
        }
    }

    #[test]
    fn rms_tiles_refuse_empty_invalid_and_overflowing_extents() {
        for (width, groups) in [
            (0, 1),
            (31, 1),
            (5121, 1),
            (128, 0),
            (128, usize::MAX),
            (5120, usize::MAX / 5120),
        ] {
            assert!(rms_tiles(width, groups).is_err());
        }
    }
}
