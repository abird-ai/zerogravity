//! Actual source records and exact allocator layouts; no physical-fit permit.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct GeometryInventory {
    pub(crate) rows: usize,
    pub(crate) values: StorageInventory,
    pub(crate) rms_private_leases: usize,
    pub(crate) rms_private_bytes: usize,
    pub(crate) rms_private_aligned_bytes: usize,
    pub(crate) physical_leases: usize,
    pub(crate) physical_aligned_bytes: usize,
}
pub(crate) struct ScopeGeometryInventory {
    pub(crate) geometries: Vec<GeometryInventory>,
    pub(crate) physical_leases: usize,
    pub(crate) physical_aligned_bytes: usize,
    pub(crate) isolated_physical_aligned_bytes: usize,
    pub(crate) workspace_views: usize,
    pub(crate) workspace_host_bytes: usize,
    _metadata: WindowWorkspace,
}
fn sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |n, v| {
        n.checked_add(v)
            .ok_or_else(|| Error::Limit("text captured geometry storage sum".into()))
    })
}
fn aligned(bytes: usize, alignment: usize) -> Result<usize> {
    if !alignment.is_power_of_two() {
        return Err(Error::State("text actual allocator alignment".into()));
    }
    bytes
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| Error::Limit("text actual private aligned extent".into()))
}
fn private_rms_layout(rows: usize, width: usize) -> Result<Vec<usize>> {
    // SAME private layout functions consumed by the actual original factories.
    // Public Embedding/normalized/residual are already in StorageRecipe slots.
    let mut layout = FullModelRmsScratch::allocation_layout(rows, width)?;
    if layout.len() != 9 {
        return Err(Error::State("text exact nine shared RMS owners".into()));
    }
    let ordinary = RmsPrepared::text_private_layout(rows, width)?;
    let dual = FusedRmsPrepared::text_private_layout(rows, width)?;
    if ordinary.len() != 1 || dual.len() != 2 {
        return Err(Error::State("text gamma/raw SumF32 private cohort".into()));
    }
    layout.extend(ordinary);
    for _ in 0..48 {
        layout.extend_from_slice(&dual);
    }
    Ok(layout)
}
impl StorageRecipe {
    /// Price the actual captured prompt records and cached geometry using the
    /// SAME native allocator recipe. Original source/profile/IDs remain sealed;
    /// no synthetic plan or logical-byte subtraction stands in for real slots.
    /// Caller separately adds learned W, roots, private GDN/native/other owners.
    pub(crate) fn scope_geometry_inventory(
        scopes: &[Arc<TextModelBinding>],
        limits: ValueStorageLimits,
    ) -> Result<ScopeGeometryInventory> {
        let metadata_bytes = workspace::WorkspaceLayout::metadata_bytes(scopes)?;
        let _metadata = scopes[0].source().reserve_workspace(metadata_bytes)?;
        let mut geometries = Vec::with_capacity(scopes.len());
        let mut recipes = Vec::with_capacity(scopes.len());
        for scope in scopes {
            let rows = scope.call().rows().len();
            let width = scope.hidden_size();
            let recipe = Self::new(scope.clone(), StorageMode::ReuseMlp, limits)?;
            let values = recipe.inventory();
            recipes.push(recipe);
            let rms = private_rms_layout(rows, width)?;
            let rms_private_bytes = sum(rms.iter().copied())?;
            let rms_private_aligned_bytes = sum(rms
                .iter()
                .map(|&b| aligned(b, limits.alignment))
                .collect::<Result<Vec<_>>>()?)?;
            let physical_leases = sum([values.physical_slots, rms.len()])?;
            let physical_aligned_bytes =
                sum([values.aligned_slot_bytes, rms_private_aligned_bytes])?;
            geometries.push(GeometryInventory {
                rows,
                values,
                rms_private_leases: rms.len(),
                rms_private_bytes,
                rms_private_aligned_bytes,
                physical_leases,
                physical_aligned_bytes,
            });
        }
        let isolated_physical_aligned_bytes =
            sum(geometries.iter().map(|g| g.physical_aligned_bytes))?;
        let workspace = workspace::WorkspaceLayout::derive(&recipes)?;
        let workspace_views = workspace.view_count()?;
        let physical_leases = sum([workspace.physical.len(), workspace.gamma_leases])?;
        let physical_aligned_bytes = sum([
            workspace.physical_aligned_bytes,
            workspace.gamma_aligned_bytes,
        ])?;
        let workspace_host_bytes = sum([
            metadata_bytes,
            crate::resident_leases::backing_pool::ResidentWorkspace::metadata_bytes(
                workspace.physical.len(),
                workspace_views,
            )?,
        ])?;
        Ok(ScopeGeometryInventory {
            geometries,
            physical_leases,
            physical_aligned_bytes,
            isolated_physical_aligned_bytes,
            workspace_views,
            workspace_host_bytes,
            _metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_overlay_counts_late_histories_and_does_not_invent_2b_head_fit() {
        // Independent source census:279 retained values plus the two actual
        // MLP cells; BA is the only sub512B-per-row value. Keep294 raw gammas.
        let expected = [
            (1024usize, 3584usize, 428_599_296usize, 101_092_864usize),
            (2048usize, 6144usize, 590_622_720usize, 139_673_088usize),
        ];
        for (h, intermediate, isolated, shared) in expected {
            let value = |rows: usize| {
                (146 * h + 427584 + 3 * intermediate) * rows * 2
                    + 496640
                    + 18 * (aligned(64 * rows, 512).unwrap() - 64 * rows)
            };
            let private =
                |rows: usize| 106 * h + 208 * rows * h + 3 * aligned(4 * rows, 512).unwrap();
            assert_eq!(
                [47, 57, 62, 67, 72, 1]
                    .into_iter()
                    .map(|m| value(m) + private(m))
                    .sum::<usize>(),
                isolated
            );
            assert_eq!(value(72) + private(72) + 5 * 98 * h, shared);
        }
        let upper = 4_033_269_760usize - (590_622_720 - 139_673_088) + 1_017_118_720 - 1024 * 1024;
        assert_eq!(upper, 4_598_390_272);
        assert_eq!(upper - 4 * 1024 * 1024 * 1024, 303_422_976);
    }
    #[test]
    fn exact_private_allocator_layout_keeps_all_source_gammas_and_48_sums() {
        for hidden in [1024, 2048] {
            for rows in [1, 47, 57, 62, 67, 72] {
                let layout = private_rms_layout(rows, hidden).unwrap();
                assert_eq!(layout.len(), 106);
                let raw = hidden * 2;
                let sum_f32 = rows * hidden * 4;
                assert_eq!(
                    RmsPrepared::text_private_layout(rows, hidden).unwrap(),
                    [raw]
                );
                assert_eq!(
                    FusedRmsPrepared::text_private_layout(rows, hidden).unwrap(),
                    [raw, sum_f32]
                );
                assert_eq!(
                    sum(layout).unwrap(),
                    106 * hidden + 208 * rows * hidden + 12 * rows
                );
            }
        }
    }
    #[test]
    fn invalid_private_alignment_and_overflow_refuse() {
        assert!(aligned(1, 0).is_err());
        assert!(aligned(1, 3).is_err());
        assert!(aligned(usize::MAX, 512).is_err());
        assert!(sum([usize::MAX, 1]).is_err());
    }
}
