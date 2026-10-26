//! Activation only under the existing fair/original workspace guard.
use super::*;

impl ColdModel {
    pub(in crate::resident_inference::native_fragment::text) fn activate_workspace(
        &self,
        rows: usize,
    ) -> Result<Option<Arc<crate::resident_leases::backing_pool::ResidentWorkspaceRun>>> {
        let index = [47, 57, 62, 67, 72, 1]
            .iter()
            .position(|&n| n == rows)
            .ok_or_else(|| Error::Unsupported("short-HF workspace geometry absent".into()))?;
        let cold = self
            .values
            .get(index)
            .ok_or_else(|| Error::State("short-HF cold workspace absent".into()))?;
        if cold.body.record().call().rows().len() != rows {
            return Err(Error::State(
                "short-HF cold workspace source geometry differs".into(),
            ));
        }
        cold.body
            .values()
            .workspace_geometry()
            .map(|g| g.try_activate())
            .transpose()
    }
}
