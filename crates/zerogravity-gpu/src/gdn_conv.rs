//! P07-H4 original Qwen GDN causal convolution, physically rounded.
//!
//! The source scalar contract evaluates four independent FP32 products and
//! then left-associated FP32 additions (not FMA). On real second-token GDN
//! history, allowing one GPU compiler kernel to inline the cast boundaries
//! caused 1,840 source-nonidentical raw values. Ordered GPU kernels place
//! four rounded products and each left-associated sum in actual FP32 memory.
//! H5 four-token qualification also required a physical boundary per addition.
//! The per-session three-frame history is ONLY PROPOSED until full physical
//! readback and source-bit comparison; one Rust QwenState alone may publish it.
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn gdn_conv_products(
    activation: &[f32],
    weights: &[f32],
    history: &[f32],
    products: &mut [f32],
    proposed_history: &mut [f32],
    #[comptime] channels: usize,
) {
    let channel = ABSOLUTE_POS;
    if channel < channels {
        let h = channel * 3;
        let w = channel * 4;
        let c0 = history[h];
        let c1 = history[h + 1];
        let c2 = history[h + 2];
        let x = activation[channel];

        // Each rounded f32 product is physically stored. The next kernel
        // cannot legally use any unrounded intermediate value.
        products[w] = f32::cast_from(f64::cast_from(c0) * f64::cast_from(weights[w]));
        products[w + 1] = f32::cast_from(f64::cast_from(c1) * f64::cast_from(weights[w + 1]));
        products[w + 2] = f32::cast_from(f64::cast_from(c2) * f64::cast_from(weights[w + 2]));
        products[w + 3] = f32::cast_from(f64::cast_from(x) * f64::cast_from(weights[w + 3]));

        proposed_history[h] = c1;
        proposed_history[h + 1] = c2;
        proposed_history[h + 2] = x;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn gdn_conv_reduce(
    products: &[f32],
    raw: &mut [f32],
    #[comptime] channels: usize,
    #[comptime] level: usize,
) {
    let channel = ABSOLUTE_POS;
    if channel < channels {
        let w = channel * 4;
        let mut a = products[w];
        if level > 0usize {
            a = raw[channel];
        }
        let b = products[w + level + 1];
        raw[channel] = f32::cast_from(f64::cast_from(a) + f64::cast_from(b));
    }
}
