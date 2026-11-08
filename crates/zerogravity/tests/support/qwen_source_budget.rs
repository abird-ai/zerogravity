//! Explicit diagnostic input; production loaders never read environment budgets.
use zerogravity::{Error, Result, gpu::HipSourceBudget};

pub fn explicit() -> Result<Option<HipSourceBudget>> {
    match std::env::var("ZG_QWEN_HIP_TOTAL_DEVICE_BYTES") {
        Ok(value) => {
            let bytes = value
                .parse::<u64>()
                .map_err(|_| Error::Malformed("explicit HIP total device bytes".into()))?;
            HipSourceBudget::new(bytes).map(Some)
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(Error::Malformed(
            "explicit HIP device budget encoding".into(),
        )),
    }
}
