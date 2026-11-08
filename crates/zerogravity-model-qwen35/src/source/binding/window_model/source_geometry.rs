//! Complete original role/type/shape validation shared by ordinary and finite binders.
use super::*;
use crate::plan::OrdinaryQwenGeometry;
use zerogravity_core::encoding::ScalarDType;

pub(super) fn require_roles(
    model: &OriginalWindowModelSource,
    profile: OrdinaryQwenGeometry,
    gdn_a_norm_kind: ScalarDType,
) -> Result<()> {
    if OrdinaryQwenGeometry::from_config(model.owner().config())? != profile {
        return Err(Error::Unsupported(
            "closed full-model source/profile geometry".into(),
        ));
    }
    let h = profile.hidden_size();
    let m = profile.intermediate_size();
    matrix(model, QwenMatrixRole::Embedding, 248320, h)?;
    let tied = model.matrix_descriptor(QwenMatrixRole::Embedding)?
        == model.matrix_descriptor(QwenMatrixRole::Output)?;
    if tied != profile.tied_embeddings() {
        return Err(Error::State(
            "original full-model configured output tying differs".into(),
        ));
    }
    matrix(model, QwenMatrixRole::Output, 248320, h)?;
    parameter(
        model,
        QwenParameterRole::FinalNorm,
        StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
        &[h as u64],
    )?;
    for i in 0..profile.layer_count() {
        for role in [
            QwenParameterRole::InputNorm(i),
            QwenParameterRole::FfnNorm(i),
        ] {
            parameter(
                model,
                role,
                StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
                &[h as u64],
            )?;
        }
        for (role, rows, columns) in [
            (QwenMatrixRole::FfnGate(i), m, h),
            (QwenMatrixRole::FfnUp(i), m, h),
            (QwenMatrixRole::FfnDown(i), h, m),
        ] {
            matrix(model, role, rows, columns)?;
        }
        if i % 4 == 3 {
            for (role, rows, columns) in [
                (QwenMatrixRole::Query(i), 2 * profile.query_width(), h),
                (QwenMatrixRole::Key(i), profile.kv_width(), h),
                (QwenMatrixRole::Value(i), profile.kv_width(), h),
                (QwenMatrixRole::AttentionOutput(i), h, profile.query_width()),
            ] {
                matrix(model, role, rows, columns)?;
            }
            for role in [
                QwenParameterRole::QueryNorm(i),
                QwenParameterRole::KeyNorm(i),
            ] {
                parameter(
                    model,
                    role,
                    StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
                    &[256],
                )?;
            }
        } else {
            for (role, rows, columns) in [
                (QwenMatrixRole::GdnQkv(i), profile.conv_channels(), h),
                (QwenMatrixRole::GdnGate(i), profile.recurrent_width(), h),
                (QwenMatrixRole::GdnBeta(i), profile.value_heads(), h),
                (QwenMatrixRole::GdnAlpha(i), profile.value_heads(), h),
                (QwenMatrixRole::GdnOutput(i), h, profile.recurrent_width()),
            ] {
                matrix(model, role, rows, columns)?;
            }
            parameter(
                model,
                QwenParameterRole::GdnConvolution(i),
                StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
                &[profile.conv_channels() as u64, 1, 4],
            )?;
            parameter(
                model,
                QwenParameterRole::GdnA(i),
                StoredTensorEncoding::Scalar(gdn_a_norm_kind),
                &[profile.value_heads() as u64],
            )?;
            parameter(
                model,
                QwenParameterRole::GdnDt(i),
                StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
                &[profile.value_heads() as u64],
            )?;
            parameter(
                model,
                QwenParameterRole::GdnNorm(i),
                StoredTensorEncoding::Scalar(gdn_a_norm_kind),
                &[128],
            )?;
        }
    }
    Ok(())
}

fn matrix(
    model: &OriginalWindowModelSource,
    role: QwenMatrixRole,
    rows: usize,
    columns: usize,
) -> Result<()> {
    let descriptor = model.matrix_descriptor(role)?;
    if descriptor.kind
        != StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16)
        || descriptor.axes != MatrixAxes::OutputInput
        || descriptor.rows != rows
        || descriptor.columns != columns
    {
        return Err(Error::State(
            "full-model exact original matrix shape/axes/type".into(),
        ));
    }
    model.validate_matrix(role, &descriptor)
}
fn parameter(
    model: &OriginalWindowModelSource,
    role: QwenParameterRole,
    kind: StoredTensorEncoding,
    shape: &[u64],
) -> Result<()> {
    let descriptor = model.parameter_descriptor(role)?;
    require_parameter_descriptor(&descriptor, role, kind, shape)?;
    model.validate_parameter(role, &descriptor)
}

fn require_parameter_descriptor(
    descriptor: &QwenParameterDescriptor,
    role: QwenParameterRole,
    kind: StoredTensorEncoding,
    shape: &[u64],
) -> Result<()> {
    let elements = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
        .ok_or_else(|| Error::Limit("full-model source parameter extent".into()))?;
    let bytes = elements
        .checked_mul(
            if kind == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16) {
                2
            } else {
                4
            },
        )
        .ok_or_else(|| Error::Limit("full-model source parameter bytes".into()))?;
    if descriptor.kind != kind || descriptor.shape != shape || descriptor.bytes != bytes {
        return Err(Error::State(format!(
            "full-model exact original parameter shape/type/extent: {role:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "source_geometry/tests.rs"]
mod tests;
