use super::{MAX_SYSTEMS, ORDER, WIDTH, solve_unit_lower};
use crate::numerics::torch_cpu::{reduction, transcendentals};
use crate::{Error, Result, accelerator::SourceIdentity};

/// Already grouped/repeated BF16 Q/K/V in `[row,head,128]`; beta/log-decay
/// `[row,head]`. Prior state is full FP32 `[head,key128,value128]` or absent.
/// This invokes chunk arithmetic even for one row; it is not cached recurrence.
pub struct ChunkInputs<'a> {
    pub source: SourceIdentity,
    pub rows: usize,
    pub heads: usize,
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub beta: &'a [f32],
    pub log_decay: &'a [f32],
    pub previous_state: Option<&'a [f32]>,
}

pub struct ChunkCompleted {
    pub output_bf16: Vec<f32>,
    pub state: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkStage {
    ScaledQuery,
    NormalizedKey,
    BetaKey,
    BetaValue,
    CumulativeDecay,
    PairwiseDecay,
    TriangularSystem,
    IntraChunkAttention,
    DecayedBetaKey,
    NewValues,
    CumulativeKey,
    DecayedQuery,
    DecayedKey,
    ChunkDecay,
    ValueCorrection,
    InterChunkAttention,
    CoreF32,
    FinalState,
    Output,
}

/// Borrowed computed values only; caller admits any observer copies.
pub type ChunkObserver<'a> = dyn FnMut(ChunkStage, &[f32]) -> Result<()> + 'a;

fn observe(
    observer: &mut Option<&mut ChunkObserver<'_>>,
    stage: ChunkStage,
    values: &[f32],
) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite source chunk intermediate".into(),
        ));
    }
    if let Some(observer) = observer {
        observer(stage, values)?;
    }
    Ok(())
}

/// One original padded64 GDN chunk. Caller admits aggregate storage and retains
/// source; private outputs grant neither StateRoot publication nor native support.
pub fn core(input: ChunkInputs<'_>) -> Result<ChunkCompleted> {
    core_observed(input, None)
}

pub fn core_observed(
    input: ChunkInputs<'_>,
    mut observer: Option<&mut ChunkObserver<'_>>,
) -> Result<ChunkCompleted> {
    let ChunkInputs {
        source,
        rows,
        heads,
        query,
        key,
        value,
        beta,
        log_decay,
        previous_state,
    } = input;
    if !source.is_bound()
        || !(1..=ORDER).contains(&rows)
        || heads == 0
        || heads > MAX_SYSTEMS
        || !heads.is_multiple_of(16)
        || query.len() != rows * heads * WIDTH
        || key.len() != query.len()
        || value.len() != query.len()
        || beta.len() != rows * heads
        || log_decay.len() != beta.len()
        || previous_state.is_some_and(|v| v.len() != heads * WIDTH * WIDTH)
    {
        return Err(Error::Malformed(
            "source chunk grouped input/state geometry".into(),
        ));
    }
    if query
        .iter()
        .chain(key)
        .chain(value)
        .chain(beta)
        .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
        || beta.iter().any(|v| !(0.0..=1.0).contains(v))
        || log_decay.iter().any(|v| !v.is_finite() || *v > 0.0)
        || previous_state.is_some_and(|v| v.iter().any(|v| !v.is_finite()))
    {
        return Err(Error::Malformed(
            "source chunk BF16/decay/state operands".into(),
        ));
    }
    let normalize = |values: &[f32], scale: f32| -> Result<Vec<f32>> {
        let normalized = reduction::l2_norm_128(source, values, 1e-6)?;
        let mut padded = vec![0.0; heads * ORDER * WIDTH];
        for h in 0..heads {
            for t in 0..rows {
                for d in 0..WIDTH {
                    padded[(h * ORDER + t) * WIDTH + d] =
                        normalized[(t * heads + h) * WIDTH + d] * scale;
                }
            }
        }
        Ok(padded)
    };
    let q = normalize(query, f32::from_bits(0x3db5_04f3))?; // RNE(128^-0.5)
    let k = normalize(key, 1.0)?;
    let mut kb = vec![0.0; q.len()];
    let mut vb = vec![0.0; q.len()];
    let mut cumulative = vec![0.0; heads * ORDER];
    for h in 0..heads {
        let mut running = 0.0f64; // Original CPU cumsum uses a double accumulator.
        for t in 0..ORDER {
            if t < rows {
                running += f64::from(log_decay[t * heads + h]);
                for d in 0..WIDTH {
                    let out = (h * ORDER + t) * WIDTH + d;
                    kb[out] = k[out] * beta[t * heads + h];
                    vb[out] = value[(t * heads + h) * WIDTH + d] * beta[t * heads + h];
                }
            } else {
                running += 0.0;
            }
            cumulative[h * ORDER + t] = running as f32;
        }
    }
    for (stage, values) in [
        (ChunkStage::ScaledQuery, &q),
        (ChunkStage::NormalizedKey, &k),
        (ChunkStage::BetaKey, &kb),
        (ChunkStage::BetaValue, &vb),
        (ChunkStage::CumulativeDecay, &cumulative),
    ] {
        observe(&mut observer, stage, values)?;
    }
    let mut pairwise = vec![f32::NEG_INFINITY; heads * ORDER * ORDER];
    for h in 0..heads {
        for t in 0..ORDER {
            for s in 0..=t {
                pairwise[(h * ORDER + t) * ORDER + s] =
                    cumulative[h * ORDER + t] - cumulative[h * ORDER + s];
            }
        }
    }
    let pairwise = transcendentals::exp_nonpositive_f32_contiguous16(source, &pairwise)?;
    observe(&mut observer, ChunkStage::PairwiseDecay, &pairwise)?;
    let mut system = matmul(&kb, &k, heads, (ORDER, ORDER, WIDTH), (false, true));
    let mut intra = matmul(&q, &k, heads, (ORDER, ORDER, WIDTH), (false, true));
    for ((a, b), decay) in system.iter_mut().zip(&mut intra).zip(pairwise) {
        *a *= decay;
        *b *= decay;
    }
    observe(&mut observer, ChunkStage::TriangularSystem, &system)?;
    observe(&mut observer, ChunkStage::IntraChunkAttention, &intra)?;
    let exponents = transcendentals::exp_nonpositive_f32_contiguous16(source, &cumulative)?;
    let mut decayed_beta = kb;
    for (row, &decay) in decayed_beta.chunks_exact_mut(WIDTH).zip(&exponents) {
        for value in row {
            *value *= decay;
        }
    }
    observe(&mut observer, ChunkStage::DecayedBetaKey, &decayed_beta)?;
    let new_values = solve_unit_lower(source, &system, &vb)?;
    let cumulative_key = solve_unit_lower(source, &system, &decayed_beta)?;
    observe(&mut observer, ChunkStage::NewValues, &new_values)?;
    observe(&mut observer, ChunkStage::CumulativeKey, &cumulative_key)?;
    let mut decayed_q = q;
    for (row, &decay) in decayed_q.chunks_exact_mut(WIDTH).zip(&exponents) {
        for value in row {
            *value *= decay;
        }
    }
    let mut final_differences = vec![0.0; cumulative.len()];
    let mut last = vec![0.0; heads];
    for h in 0..heads {
        last[h] = cumulative[(h + 1) * ORDER - 1];
        for t in 0..ORDER {
            final_differences[h * ORDER + t] = last[h] - cumulative[h * ORDER + t];
        }
    }
    let final_exponents =
        transcendentals::exp_nonpositive_f32_contiguous16(source, &final_differences)?;
    let chunk_decay = transcendentals::exp_nonpositive_f32_contiguous16(source, &last)?;
    let mut decayed_k = k;
    for (row, decay) in decayed_k.chunks_exact_mut(WIDTH).zip(final_exponents) {
        for value in row {
            *value *= decay;
        }
    }
    observe(&mut observer, ChunkStage::DecayedQuery, &decayed_q)?;
    observe(&mut observer, ChunkStage::DecayedKey, &decayed_k)?;
    observe(&mut observer, ChunkStage::ChunkDecay, &chunk_decay)?;
    let empty = if previous_state.is_none() {
        vec![0.0; heads * WIDTH * WIDTH]
    } else {
        Vec::new()
    };
    let prior = previous_state.unwrap_or(&empty);
    let predicted = matmul(
        &cumulative_key,
        prior,
        heads,
        (ORDER, WIDTH, WIDTH),
        (false, false),
    );
    let correction = new_values
        .into_iter()
        .zip(predicted)
        .map(|(v, p)| v - p)
        .collect::<Vec<_>>();
    let inter = matmul(
        &decayed_q,
        prior,
        heads,
        (ORDER, WIDTH, WIDTH),
        (false, false),
    );
    observe(&mut observer, ChunkStage::ValueCorrection, &correction)?;
    observe(&mut observer, ChunkStage::InterChunkAttention, &inter)?;
    let mut core = matmul(
        &intra,
        &correction,
        heads,
        (ORDER, WIDTH, ORDER),
        (false, false),
    );
    for (value, old) in core.iter_mut().zip(inter) {
        *value += old;
    }
    observe(&mut observer, ChunkStage::CoreF32, &core)?;
    let mut state = matmul(
        &decayed_k,
        &correction,
        heads,
        (WIDTH, WIDTH, ORDER),
        (true, false),
    );
    for (h, &decay) in chunk_decay.iter().enumerate() {
        for i in 0..WIDTH * WIDTH {
            let at = h * WIDTH * WIDTH + i;
            state[at] += prior[at] * decay;
        }
    }
    observe(&mut observer, ChunkStage::FinalState, &state)?;
    let mut output = vec![0.0; rows * heads * WIDTH];
    for t in 0..rows {
        for h in 0..heads {
            for d in 0..WIDTH {
                output[(t * heads + h) * WIDTH + d] =
                    super::super::round(core[(h * ORDER + t) * WIDTH + d])?;
            }
        }
    }
    observe(&mut observer, ChunkStage::Output, &output)?;
    Ok(ChunkCompleted {
        output_bf16: output,
        state,
    })
}

// Original SGEMM: one FP32 FMA chain, then ADD into the separately initialized C
// plane (Zen kernel0, ELF0xcbda4c0). Do not drop that beta=0 C epilogue.
fn matmul(
    a: &[f32],
    b: &[f32],
    heads: usize,
    shape: (usize, usize, usize),
    transpose: (bool, bool),
) -> Vec<f32> {
    let (m, n, k) = shape;
    let mut output = vec![0.0; heads * m * n];
    for h in 0..heads {
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0f32;
                for z in 0..k {
                    let ai = h * m * k
                        + if transpose.0 {
                            z * m + row
                        } else {
                            row * k + z
                        };
                    let bi = h * k * n
                        + if transpose.1 {
                            column * k + z
                        } else {
                            z * n + column
                        };
                    sum = a[ai].mul_add(b[bi], sum);
                }
                output[(h * m + row) * n + column] += sum;
            }
        }
    }
    output
}
