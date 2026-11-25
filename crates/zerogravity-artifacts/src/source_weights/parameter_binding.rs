//! One authenticated complete parameter reader for family/native consumers.
use super::{ParameterWindow, VerifiedParameter, VerifiedTensorSource};
use crate::Result;
use std::sync::atomic::AtomicBool;
use zerogravity_core::accelerator::program::SourceParameter;
impl VerifiedTensorSource {
    /// Bind exact issuer/header metadata without IO, flattening or staging.
    /// Conversion and numerical eligibility remain with the consumer contract.
    pub fn bind_declared_parameter(
        &self,
        descriptor: &SourceParameter,
    ) -> Result<VerifiedParameter> {
        let parameter = if descriptor.shape.len() == 3 {
            self.conv4_parameter(&descriptor.name)?
        } else {
            self.parameter(&descriptor.name)?
        };
        parameter.validate_source_parameter(descriptor)?;
        Ok(parameter)
    }
    /// Complete raw bytes under the same validated binding and source owner.
    pub fn read_declared_parameter(
        &self,
        descriptor: &SourceParameter,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ParameterWindow> {
        let parameter = self.bind_declared_parameter(descriptor)?;
        let range = 0..parameter.elements();
        let window = match cancelled {
            Some(flag) => self.read_parameter_cancellable(&parameter, range, flag)?,
            None => self.read_parameter(&parameter, range)?,
        };
        window.require_complete(&parameter)?;
        Ok(window)
    }
}
