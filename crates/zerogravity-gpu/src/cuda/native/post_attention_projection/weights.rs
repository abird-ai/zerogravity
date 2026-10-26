//! Complete source-owned Out W; pending writes remain on the original init.
use super::*;
struct WeightSource {
    recipe: Arc<ComponentRecipe>,
    window: TensorWindow,
    bytes: Vec<u8>,
    _host: WindowWorkspace,
}
pub(crate) struct PendingOutWeights {
    source: Arc<WeightSource>,
    lease: ResidentLease,
}
pub(super) struct OutWeights {
    source: Arc<WeightSource>,
    pub(super) lease: ResidentLease,
}
impl PendingOutWeights {
    pub(crate) fn upload(
        values: &Arc<ComponentValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        require_coverage(values.recipe())?;
        Self::upload_checked(values, init, cancelled)
    }
    pub(crate) fn upload_mlp(
        values: &Arc<ComponentValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        require_mlp_coverage(values.recipe())?;
        Self::upload_checked(values, init, cancelled)
    }
    pub(crate) fn upload_qwen2b_mlp(
        values: &Arc<ComponentValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        require_qwen2b_mlp_coverage(values.recipe())?;
        Self::upload_checked(values, init, cancelled)
    }
    fn upload_checked(
        values: &Arc<ComponentValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        init.require_service(values.service())?;
        let recipe = values.recipe();
        let op = recipe.operator(6)?;
        let member = &op.matrices()[0];
        let hidden = recipe.kind().columns();
        let bytes = hidden * 2048 * 2;
        let source = recipe.source();
        // Shadow, write copy, and D2H validation capacity. Source chunks are separately admitted.
        let host = source.reserve_workspace(3 * bytes)?;
        let matrix = source.matrix(&member.name, member.axes)?;
        if matrix.tensor().kind != zerogravity_core::quant::QuantKind::Bf16
            || matrix.rows() != hidden
            || matrix.columns() != 2048
            || member.rows != hidden
            || member.columns != 2048
            || matrix.source_identity() != values.service().source_identity()
        {
            return Err(Error::State(
                "GdnOut exact original BF16 learned source extent".into(),
            ));
        }
        let window = match cancelled {
            Some(flag) => source.read_rows_cancellable(&matrix, 0..hidden, flag)?,
            None => source.read_rows(&matrix, 0..hidden)?,
        };
        if window.row_range() != (0..hidden)
            || window.as_slice().len() != bytes
            || !super::super::finite_bf16(window.as_slice())
        {
            return Err(Error::State("GdnOut complete finite source window".into()));
        }
        let packed = window.as_slice().to_vec();
        let source = Arc::new(WeightSource {
            recipe: recipe.clone(),
            window,
            bytes: packed,
            _host: host,
        });
        init.retain_completion_owner(source.clone())?;
        let lease = init.upload_bytes(&source.bytes, BufferRole::ImmutableWeights)?;
        Ok(Self { source, lease })
    }
    pub(super) fn bind(self, init: &ResidentInitialization<'_>) -> Result<Arc<OutWeights>> {
        init.require_fixed_prompt_projection_upload(&self.lease)?;
        Ok(Arc::new(OutWeights {
            source: self.source,
            lease: self.lease,
        }))
    }
}
impl OutWeights {
    pub(super) fn require_source(
        &self,
        values: &Arc<ComponentValues>,
        initialized: bool,
    ) -> Result<()> {
        let recipe = values.recipe();
        require_bound_recipe(recipe)?;
        values.service().ensure_usable()?;
        if !Arc::ptr_eq(recipe, &self.source.recipe)
            || !Arc::ptr_eq(self.lease.service(), values.service())
            || self.lease.source_identity() != recipe.bindings().call().source()
            || self.lease.service_generation() != values.service().generation()
            || self.lease.role() != BufferRole::ImmutableWeights
            || initialized && !self.lease.is_initialized()
            || self.lease.bytes() != recipe.kind().columns() * 4096
            || self.source.bytes.len() != self.lease.bytes()
            || self.source.window.row_range() != (0..recipe.kind().columns())
            || self.source.window.as_slice() != self.source.bytes.as_slice()
        {
            return Err(Error::State(
                "GdnOut original pending/initialized source W identity".into(),
            ));
        }
        require_matrix_source(recipe.source(), &recipe.operator(6)?.matrices()[0])
    }
    pub(super) fn immutable(&self) -> (ResidentLease, &[u8]) {
        (self.lease.clone(), &self.source.bytes)
    }
}
