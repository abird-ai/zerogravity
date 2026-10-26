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

#[derive(Clone, Copy)]
struct FaultPlan {
    pressure: bool,
    cancel_before: Option<usize>,
    unknown_after: Option<usize>,
}
impl FaultPlan {
    fn new(mode: RotaryMode) -> Self {
        Self {
            pressure: mode == RotaryMode::Pressure,
            cancel_before: mode.cancel_before(),
            unknown_after: mode.unknown_after(),
        }
    }
    fn is_fault(self) -> bool {
        self.cancel_before.is_some() || self.unknown_after.is_some()
    }
}

struct QualificationControl<F> {
    plan: FaultPlan,
    validate: Option<F>,
    complete_status: &'static str,
}
impl<F> PreparationControl for QualificationControl<F>
where
    F: FnOnce(&[(RotaryOutput, &[u8])]) -> Result<()>,
{
    fn pressure(&self) -> bool {
        self.plan.pressure
    }
    fn cancel_before(&self, index: usize) -> bool {
        self.plan.cancel_before == Some(index)
    }
    fn unknown_after(&self, launches: usize) -> bool {
        self.plan.unknown_after == Some(launches)
    }
    fn is_fault(&self) -> bool {
        self.plan.is_fault()
    }
    fn validate(&mut self, records: &[(RotaryOutput, &[u8])]) -> Result<()> {
        self.validate
            .take()
            .ok_or_else(|| Error::State("rotary diagnostic callback already consumed".into()))?(
            records,
        )
    }
    fn pressure_status(&self) -> &'static str {
        "SOURCE_ROTARY_PRESSURE_REFUSED_BEFORE_BUFFER_NATIVE_H2D"
    }
    fn fault_status(&self, cancelled: bool) -> &'static str {
        if cancelled {
            "SOURCE_ROTARY_CANCELLED_UNKNOWN_RETAINED"
        } else {
            "SOURCE_ROTARY_UNKNOWN_RETAINED"
        }
    }
    fn complete_status(&self, _kind: SourceKind) -> &'static str {
        self.complete_status
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
        cancelled,
        QualificationControl {
            plan: FaultPlan::new(mode),
            validate: Some(validate),
            complete_status: "SOURCE_ROTARY_FIRST67_EXACT_NO_MODEL_QUALIFICATION",
        },
        Layout::new(SourceKind::FixedPrompt, 67)?,
    )
    .await
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
        cancelled,
        QualificationControl {
            plan: FaultPlan::new(mode),
            validate: Some(validate),
            complete_status: "SOURCE_ROTARY_FIRST79_EXACT_NO_MODEL_QUALIFICATION",
        },
        Layout::new(SourceKind::Text, 79)?,
    )
    .await
}
