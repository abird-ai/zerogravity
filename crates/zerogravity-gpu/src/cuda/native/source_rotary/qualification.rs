//! Diagnostic cases over the shared source constructor.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotaryMode {
    Complete,
    Pressure,
    CancelBeforeFirstKernel,
    UnknownAfterPow,
    CancelAfterOuter,
    UnknownAfterCat,
    CancelBeforeCast,
    UnknownAfterCast,
}
impl RotaryMode {
    fn cancel_before(self) -> Option<usize> {
        match self {
            Self::CancelBeforeFirstKernel => Some(0),
            Self::CancelAfterOuter => Some(8),
            Self::CancelBeforeCast => Some(11),
            _ => None,
        }
    }
    fn unknown_after(self) -> Option<usize> {
        match self {
            Self::UnknownAfterPow => Some(4),
            Self::UnknownAfterCat => Some(11),
            Self::UnknownAfterCast => Some(12),
            _ => None,
        }
    }
}
impl RotaryMode {
    fn plan(self) -> FaultPlan {
        FaultPlan {
            pressure: self == Self::Pressure,
            cancel_before: self.cancel_before(),
            unknown_after: self.unknown_after(),
        }
    }
}

/// All mathematical inputs are original source factories/constants. Callback
/// may compare independent expected bytes only AFTER the actual native math.
/// No captured inv_freq/frequencies/cos/sin/cache can enter device preparation.
pub async fn qualify_source_rotary_cubins<F>(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: Option<CubinQualificationInputs>,
    mode: RotaryMode,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<RotaryOutcome>
where
    F: FnOnce(&[(RotaryOutput, &[u8])]) -> Result<()>,
{
    prepare_rotary_inner(
        device,
        source,
        device_cap,
        inputs,
        mode.plan(),
        cancelled,
        validate,
        Layout::new(SourceKind::FixedPrompt, 67)?,
    )
    .await
    .map(|mut outcome| {
        if outcome.prefix.is_some() {
            outcome.comparison.status = "SOURCE_ROTARY_FIRST67_EXACT_NO_MODEL_QUALIFICATION";
        }
        outcome
    })
}

/// Same original native source constructor; authentic .8B/2B and positions0..79.
pub async fn qualify_text_rotary_cubins<F>(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: Option<CubinQualificationInputs>,
    mode: RotaryMode,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<RotaryOutcome>
where
    F: FnOnce(&[(RotaryOutput, &[u8])]) -> Result<()>,
{
    prepare_rotary_inner(
        device,
        source,
        device_cap,
        inputs,
        mode.plan(),
        cancelled,
        validate,
        Layout::new(SourceKind::Text, 79)?,
    )
    .await
    .map(|mut outcome| {
        if outcome.prefix.is_some() {
            outcome.comparison.status = "SOURCE_ROTARY_FIRST79_EXACT_NO_MODEL_QUALIFICATION";
        }
        outcome
    })
}
