#![cfg(all(feature = "acquisition", target_os = "linux"))]
//! Exact future native-reader source, currently unwired pure primitives only.
//! Passing these tests does not establish native Xet transport or acquisition.
pub use zerogravity_artifacts::{Error, Result};
#[path = "../src/acquisition/xet/codec.rs"]
mod codec;
#[path = "../src/acquisition/xet/hash.rs"]
mod hash;
