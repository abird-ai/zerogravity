//! One authenticated complete parameter reader for family/native consumers.
use super::{ParameterWindow, VerifiedTensorSource};
use crate::{Error, Result};
use std::sync::atomic::AtomicBool;
use zerogravity_core::accelerator::program::SourceParameter;
impl VerifiedTensorSource {
    /// Resolves the actual source/header issuer before reading. Conversion and
    /// native numerical eligibility remain with the explicit consumer contract.
    pub fn read_declared_parameter(
        &self,
        descriptor: &SourceParameter,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ParameterWindow> {
        let parameter = if descriptor.shape.len() == 3 {
            self.conv4_parameter(&descriptor.name)?
        } else {
            self.parameter(&descriptor.name)?
        };
        parameter.validate_source_parameter(descriptor)?;
        let range = 0..parameter.elements();
        let window = match cancelled {
            Some(flag) => self.read_parameter_cancellable(&parameter, range, flag)?,
            None => self.read_parameter(&parameter, range)?,
        };
        if !window.is_complete() {
            return Err(Error::State(
                "complete original declared parameter required".into(),
            ));
        }
        Ok(window)
    }
}
