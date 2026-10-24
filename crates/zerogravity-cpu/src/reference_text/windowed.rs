//! Same immutable original-index row windows and shared CPU projection kernel.
use super::*;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::{quant::QuantKind, source::SourceTensorRef};

pub struct WindowReferenceTextSource {
    source: VerifiedTensorSource,
}
pub struct ReferenceTextParameter {
    pub descriptor: SourceParameter,
    pub values: Vec<f32>,
    _admission: TextAdmission,
}
impl WindowReferenceTextSource {
    pub fn new(source: VerifiedTensorSource) -> Self {
        Self { source }
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub fn matrix(&self, name: &str) -> Result<SourceTensor> {
        let matrix = self.source.matrix(name, MatrixAxes::OutputInput)?;
        if !matches!(matrix.tensor().kind, QuantKind::Bf16 | QuantKind::F32)
            || matrix.columns() > 21504
            || matrix.rows() > 1_048_576
        {
            return Err(Error::Unsupported(
                "analytic text bounded original BF16/F32 matrices only".into(),
            ));
        }
        Ok(SourceTensor {
            source: self.source.source_identity(),
            name: name.into(),
            kind: matrix.tensor().kind,
            axes: matrix.axes(),
            columns: matrix.columns(),
            rows: matrix.rows(),
        })
    }
    pub fn expert_tensor(&self, name: &str) -> Result<TextExpertTensor> {
        let raw = self.source.tensor_info(name)?;
        let [experts, rows, columns] = raw.shape.as_slice() else {
            return Err(Error::Malformed(
                "text experts require original rank3".into(),
            ));
        };
        let matrix = self.source.expert_matrix(name, 0)?;
        let experts =
            usize::try_from(*experts).map_err(|_| Error::Limit("text expert count".into()))?;
        if experts > 4096 || matrix.rows() > 32768 || matrix.columns() > 16384 {
            return Err(Error::Limit("bounded text expert tensor".into()));
        }
        Ok(TextExpertTensor {
            source: self.source_identity(),
            name: name.into(),
            kind: raw.kind,
            experts,
            rows: *rows as usize,
            columns: *columns as usize,
        })
    }
    pub fn parameter(
        &self,
        name: &str,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ReferenceTextParameter> {
        check(cancelled)?;
        let parameter = self.source.parameter(name)?;
        if parameter.elements() == 0 || parameter.elements() > 16384 {
            return Err(Error::Limit("bounded original text parameter".into()));
        }
        let admission: TextAdmission =
            Arc::new(self.source.reserve_workspace(parameter.elements() * 4)?);
        let raw = match cancelled {
            Some(flag) => {
                self.source
                    .read_parameter_cancellable(&parameter, 0..parameter.elements(), flag)?
            }
            None => self
                .source
                .read_parameter(&parameter, 0..parameter.elements())?,
        };
        if !raw.is_complete() {
            return Err(Error::State("whole original text parameter window".into()));
        }
        let values = decode(parameter.tensor().kind, raw.as_slice())?;
        if values.len() != parameter.elements() {
            return Err(Error::State(
                "complete original text parameter words".into(),
            ));
        }
        check(cancelled)?;
        Ok(ReferenceTextParameter {
            descriptor: parameter.source_parameter(),
            values,
            _admission: admission,
        })
    }
    fn project_matrix(
        &self,
        tensor: &SourceTensor,
        input: &[f32],
        matrix: &zerogravity_artifacts::source_weights::VerifiedMatrix,
        cancelled: Option<&AtomicBool>,
        profile: crate::weights::SourceProjectionProfile,
    ) -> Result<Vec<f32>> {
        let row_bytes = usize::try_from(tensor.kind.byte_len(tensor.columns as u64)?)
            .map_err(|_| Error::Limit("text source row bytes".into()))?;
        let per_window = (1024 * 1024 / row_bytes).clamp(1, 128);
        // Covers shared projection scratch, independent of the complete output's ticket.
        let _scratch = self.source.reserve_workspace(
            tensor
                .columns
                .checked_mul(8)
                .and_then(|x| x.checked_add(per_window * 4))
                .ok_or_else(|| Error::Limit("text projection scratch".into()))?,
        )?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(tensor.rows)
            .map_err(|_| Error::Limit("text source full projection output".into()))?;
        for start in (0..tensor.rows).step_by(per_window) {
            check(cancelled)?;
            let rows = start..(start + per_window).min(tensor.rows);
            let window = match cancelled {
                Some(flag) => self
                    .source
                    .read_rows_cancellable(matrix, rows.clone(), flag)?,
                None => self.source.read_rows(matrix, rows.clone())?,
            };
            if window.matrix().source_identity() != tensor.source
                || window.row_range() != rows
                || output.len() != start
            {
                return Err(Error::State(
                    "original text matrix window identity/coverage".into(),
                ));
            }
            let shape = [rows.len() as u64, tensor.columns as u64];
            let part = crate::original_hf::output_first_projection_with_profile(
                tensor.source,
                &tensor.name,
                SourceTensorRef {
                    kind: tensor.kind,
                    shape: &shape,
                    packed_bytes: window.as_slice(),
                },
                input,
                None,
                profile,
            )?;
            if part.len() != rows.len() {
                return Err(Error::State(
                    "text source projection complete row cover".into(),
                ));
            }
            output.extend(part);
        }
        check(cancelled)?;
        Ok(output)
    }
    fn validate(&self, tensor: &SourceTensor) -> Result<()> {
        if *tensor != self.matrix(&tensor.name)? {
            return Err(Error::State(
                "text matrix changed actual immutable source role/axes/dtype".into(),
            ));
        }
        Ok(())
    }
}
fn check(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Err(Error::State("reference text cancelled".into()))
    } else {
        Ok(())
    }
}
fn decode(kind: QuantKind, raw: &[u8]) -> Result<Vec<f32>> {
    let width = match kind {
        QuantKind::Bf16 => 2,
        QuantKind::F32 => 4,
        _ => return Err(Error::Unsupported("original text parameter dtype".into())),
    };
    if !raw.len().is_multiple_of(width) {
        return Err(Error::Malformed("original text packed words".into()));
    }
    let result = raw
        .chunks_exact(width)
        .map(|word| {
            if width == 2 {
                f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
            } else {
                f32::from_le_bytes([word[0], word[1], word[2], word[3]])
            }
        })
        .collect::<Vec<_>>();
    if result.iter().any(|x| !x.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite original text source parameter".into(),
        ));
    }
    Ok(result)
}
impl ReferenceTextSource for WindowReferenceTextSource {
    fn project_expert(
        &self,
        tensor: &TextExpertTensor,
        expert: usize,
        input: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        check(cancelled)?;
        if *tensor != self.expert_tensor(&tensor.name)?
            || expert >= tensor.experts
            || input.len() != tensor.columns
        {
            return Err(Error::State(
                "text expert actual source/plane/input differs".into(),
            ));
        }
        let matrix = self.source.expert_matrix(&tensor.name, expert)?;
        let descriptor = SourceTensor {
            source: tensor.source,
            name: tensor.name.clone(),
            kind: tensor.kind,
            axes: MatrixAxes::OutputInput,
            columns: tensor.columns,
            rows: tensor.rows,
        };
        self.project_matrix(
            &descriptor,
            input,
            &matrix,
            cancelled,
            crate::weights::SourceProjectionProfile::Source,
        )
    }

    fn source_identity(&self) -> SourceIdentity {
        self.source.source_identity()
    }
    fn reserve_output(&self, bytes: usize) -> Result<Option<TextAdmission>> {
        Ok(Some(Arc::new(self.source.reserve_workspace(bytes)?)))
    }
    fn embed(
        &self,
        tensor: &SourceTensor,
        token: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        check(cancelled)?;
        self.validate(tensor)?;
        if token >= tensor.rows {
            return Err(Error::Malformed("text embedding source token".into()));
        }
        let matrix = self.source.matrix(&tensor.name, tensor.axes)?;
        let row = match cancelled {
            Some(flag) => self
                .source
                .read_rows_cancellable(&matrix, token..token + 1, flag)?,
            None => self.source.read_rows(&matrix, token..token + 1)?,
        };
        let values = decode(tensor.kind, row.as_slice())?;
        if values.len() != tensor.columns {
            return Err(Error::State("complete original text embedding row".into()));
        }
        check(cancelled)?;
        Ok(values)
    }
    fn project_with_arithmetic(
        &self,
        tensor: &SourceTensor,
        input: &[f32],
        arithmetic: ReferenceTextArithmetic,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        if arithmetic == ReferenceTextArithmetic::AnalyticF32 {
            return self.project(tensor, input, cancelled);
        }
        check(cancelled)?;
        self.validate(tensor)?;
        if input.len() != tensor.columns || tensor.kind != QuantKind::Bf16 {
            return Err(Error::Unsupported(
                "candidate BF16 projection actual source/input".into(),
            ));
        }
        let matrix = self.source.matrix(&tensor.name, tensor.axes)?;
        self.project_matrix(
            tensor,
            input,
            &matrix,
            cancelled,
            crate::weights::SourceProjectionProfile::TorchCpuBf16Avx2,
        )
    }
    fn project(
        &self,
        tensor: &SourceTensor,
        input: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        check(cancelled)?;
        self.validate(tensor)?;
        if input.len() != tensor.columns {
            return Err(Error::Malformed(
                "complete original text projection input".into(),
            ));
        }
        let matrix = self.source.matrix(&tensor.name, tensor.axes)?;
        self.project_matrix(
            tensor,
            input,
            &matrix,
            cancelled,
            crate::weights::SourceProjectionProfile::Source,
        )
    }
}
