//! Cold config inspection through compiled family adapters; never execution admission.
use crate::{Error, Result};
use std::path::Path;

const MAX_CONFIG: u64 = 1024 * 1024;

#[derive(Debug)]
pub enum DeclaredModel {
    #[cfg(feature = "model-qwen35")]
    Qwen(crate::qwen35::config::ModelGeometry),
    #[cfg(feature = "model-gemma4")]
    Gemma(crate::gemma4::config::DeclaredGeometry),
}

impl DeclaredModel {
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_json(&zerogravity_artifacts::config::read(path, MAX_CONFIG)?)
    }
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let config = zerogravity_artifacts::config::json(bytes, MAX_CONFIG)?;
        match config.get("model_type").and_then(serde_json::Value::as_str) {
            Some("qwen3_5") => {
                #[cfg(feature = "model-qwen35")]
                {
                    Ok(Self::Qwen(crate::qwen35::config::ModelGeometry::from_json(
                        bytes,
                    )?))
                }
                #[cfg(not(feature = "model-qwen35"))]
                {
                    Err(Error::Unsupported(
                        "Qwen config inspection adapter is not compiled".into(),
                    ))
                }
            }
            Some("gemma4") => {
                #[cfg(feature = "model-gemma4")]
                {
                    Ok(Self::Gemma(
                        crate::gemma4::config::DeclaredGeometry::from_json(bytes)?,
                    ))
                }
                #[cfg(not(feature = "model-gemma4"))]
                {
                    Err(Error::Unsupported(
                        "Gemma config inspection adapter is not compiled".into(),
                    ))
                }
            }
            _ => Err(Error::Unsupported(
                "no compiled family config inspection adapter".into(),
            )),
        }
    }
}

impl std::fmt::Display for DeclaredModel {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(any(feature = "model-qwen35", feature = "model-gemma4"))]
        {
            match self {
                #[cfg(feature = "model-qwen35")]
                Self::Qwen(geometry) => geometry.fmt(output),
                #[cfg(feature = "model-gemma4")]
                Self::Gemma(geometry) => geometry.fmt(output),
            }
        }
        #[cfg(not(any(feature = "model-qwen35", feature = "model-gemma4")))]
        {
            let _ = output;
            match *self {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_family_and_malformed_config_refuse() {
        assert!(DeclaredModel::from_json(br#"{"model_type":"llama"}"#).is_err());
        assert!(DeclaredModel::from_json(b"{").is_err());
        assert!(DeclaredModel::from_json(&vec![b' '; MAX_CONFIG as usize + 1]).is_err());
    }
    #[cfg(not(feature = "model-qwen35"))]
    #[test]
    fn missing_qwen_adapter_refuses_without_claiming_geometry() {
        let error = DeclaredModel::from_json(br#"{"model_type":"qwen3_5"}"#).unwrap_err();
        assert!(error.to_string().contains("not compiled"));
    }
    #[cfg(not(feature = "model-gemma4"))]
    #[test]
    fn missing_gemma_adapter_refuses_without_claiming_geometry() {
        let error = DeclaredModel::from_json(br#"{"model_type":"gemma4"}"#).unwrap_err();
        assert!(error.to_string().contains("not compiled"));
    }
}
