//! Additive finite47 QKV union; old eight exact MLP records remain unchanged.
use super::{CubinKernelManifest, VerifiedCubin, failure};
use cubecl_core::server::ServerError;

// ROOT materializes the two complete ABI digests after actual packet/source
// comparison and Rust digest guards. None is an explicit cold refusal.
const QKV_ABIS: Option<[[u8; 32]; 2]> = Some([
    [
        0x4b, 0x67, 0x75, 0x86, 0xff, 0xed, 0xb0, 0xce, 0x2c, 0x90, 0xf2, 0x8e, 0x96, 0x94, 0x57,
        0x9d, 0x91, 0x57, 0x82, 0x79, 0x7d, 0x89, 0x87, 0xe3, 0x4c, 0xff, 0xa4, 0x42, 0xc5, 0x11,
        0x14, 0xcd,
    ],
    [
        0x18, 0xc8, 0xb2, 0x71, 0x6a, 0x67, 0xf9, 0x06, 0x41, 0x3a, 0x7f, 0x69, 0x99, 0xed, 0x70,
        0x4d, 0x77, 0xb4, 0x71, 0xdc, 0x72, 0x87, 0xbc, 0xa7, 0x74, 0xa6, 0x31, 0x19, 0xd9, 0x4c,
        0x47, 0xbd,
    ],
]);
const IMAGES: [([u8; 32], usize, u8); 2] = [
    (
        [
            0x6a, 0xb9, 0x46, 0x8c, 0x2d, 0x21, 0xb9, 0xb2, 0x4c, 0x66, 0xfb, 0xb4, 0x38, 0x0e,
            0xfc, 0xa4, 0x7a, 0x90, 0x57, 0xdd, 0xcc, 0x6a, 0xdb, 0x4e, 0xb8, 0x39, 0x97, 0x46,
            0x09, 0x86, 0xa8, 0x3e,
        ],
        6_352_608,
        80,
    ),
    (
        [
            0x07, 0x48, 0x07, 0xd6, 0x71, 0xd9, 0x15, 0x6c, 0xcc, 0x56, 0xb5, 0xa8, 0xcb, 0xe8,
            0x2f, 0x1b, 0xa8, 0xfa, 0xe2, 0x1c, 0x81, 0x69, 0xe6, 0x74, 0xb3, 0x1b, 0x4b, 0x86,
            0x03, 0xba, 0x42, 0xec,
        ],
        4_252_960,
        86,
    ),
];

pub(super) fn require(
    image: &VerifiedCubin,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    if super::fixed_prompt_mlp_policy::require(image, manifest).is_ok() {
        return Ok(());
    }
    match (image.derived_from(), image.captured_from()) {
        (Some(pin), None) => {
            pin.require_image(image.byte_len(), image.sha256(), image.binary_sm())?
        }
        (None, Some(pin)) => {
            pin.require_image(image.byte_len(), image.sha256(), image.binary_sm())?
        }
        _ => {
            return Err(failure(
                "canonical47 QKV exactly one sealed image provenance",
            ));
        }
    }
    require_record(
        image.sha256(),
        image.byte_len(),
        image.binary_sm(),
        manifest,
        QKV_ABIS,
    )
}

fn require_record(
    sha: [u8; 32],
    bytes: usize,
    sm: u8,
    manifest: &CubinKernelManifest,
    abis: Option<[[u8; 32]; 2]>,
) -> Result<(), ServerError> {
    let abis =
        abis.ok_or_else(|| failure("canonical47 QKV complete ABI pins pending ROOT acceptance"))?;
    let digest = super::attention_policy::digest(manifest);
    if !IMAGES
        .iter()
        .zip(abis)
        .any(|(image, abi)| *image == (sha, bytes, sm) && digest == abi)
    {
        return Err(failure(
            "canonical47 QKV full image/geometry/scalar/access ABI differs",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
