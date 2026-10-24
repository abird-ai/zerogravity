//! Bounded original-model reuse; existing five numerical geometries only.
pub use super::full_model_qualify::natural_cold::qualify_vllm08_natural_reuse_pressure;
pub(crate) use super::full_model_qualify::natural_cold::{
    PreparedReusableNatural, ReusableNaturalStorage, prepare_reusable_natural_cold,
};
