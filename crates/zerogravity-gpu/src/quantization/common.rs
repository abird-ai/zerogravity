//! Shared source reduction and signed-byte decoding.
use cubecl::prelude::*;

#[cube]
pub(crate) fn signed_byte(value: u32) -> i32 {
    let mut result = i32::cast_from(value);
    if value >= 128u32 {
        result -= 256i32;
    }
    result
}

/// Kernel lanes preserve original GGML eight-way float reduction order.
#[cube]
pub(crate) fn eight_lane_reduce(lanes: &Array<f32>) -> f32 {
    let r0 = lanes[0] + lanes[4];
    let r1 = lanes[1] + lanes[5];
    let r2 = lanes[2] + lanes[6];
    let r3 = lanes[3] + lanes[7];
    (r0 + r2) + (r1 + r3)
}
