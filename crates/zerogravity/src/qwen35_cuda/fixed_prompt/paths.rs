//! Fixed-prompt native artifact path resolution. Held source directory and
//! digest helpers are shared with ordinary CUDA source authentication.

use super::*;
pub(crate) use crate::qwen35_cuda::{held_directory::Directory, source_inventory::digest};
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

pub(crate) fn native_root(
    explicit: Option<&Path>,
    xdg: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if !path.is_absolute() {
            return Err(Error::Malformed(
                "native artifacts root must be absolute".into(),
            ));
        }
        return Ok(path.to_owned());
    }
    let base = if let Some(path) = xdg.map(Path::new).filter(|p| p.is_absolute()) {
        path.to_owned()
    } else {
        home.map(Path::new)
            .filter(|p| p.is_absolute())
            .ok_or_else(|| Error::State("absolute HOME required for native artifact cache".into()))?
            .join(".cache")
    };
    Ok(base.join("abird/zerogravity/cuda/fixed-prompt"))
}
