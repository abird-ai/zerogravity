//! Captured source recipes, semantic cells and exact original geometry views.
use super::*;
use crate::resident_leases::backing_pool::{
    ResidentPlaneLayout, ResidentViewLayout, ResidentWorkspace, ResidentWorkspaceGeometry,
};

type ValueMembership = Box<[(usize, usize)]>;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CellKey {
    // Exact actual ValueId/overwrite membership, never a packed slot ordinal.
    Value(ValueMembership),
    RmsPrivate(usize),
    DualSum(usize),
}
#[derive(Clone)]
struct GeometryLayout {
    views: Box<[ResidentViewLayout]>,
    value_slots: usize,
    sums: Box<[usize]>,
}
pub(crate) struct WorkspaceLayout {
    pub(crate) physical: Box<[ResidentPlaneLayout]>,
    geometries: Box<[GeometryLayout]>,
    pub(crate) physical_aligned_bytes: usize,
    pub(crate) gamma_leases: usize,
    pub(crate) gamma_aligned_bytes: usize,
}
fn aligned(bytes: usize, alignment: usize) -> Result<usize> {
    if bytes == 0 || !alignment.is_power_of_two() {
        return Err(Error::State(
            "shared geometry actual extent/alignment".into(),
        ));
    }
    bytes
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| Error::Limit("shared geometry extent overflow".into()))
}
fn signatures(bindings: &[ValueStorageBinding], slots: usize) -> Result<Vec<ValueMembership>> {
    let mut groups = vec![Vec::new(); slots];
    for (value, b) in bindings.iter().enumerate() {
        groups
            .get_mut(b.slot)
            .ok_or_else(|| Error::State("shared geometry slot index".into()))?
            .push((value, b.epoch));
    }
    if groups.iter().any(Vec::is_empty) {
        return Err(Error::State("shared geometry unused physical slot".into()));
    }
    Ok(groups.into_iter().map(Vec::into_boxed_slice).collect())
}
fn cell(
    keys: &mut BTreeMap<CellKey, usize>,
    physical: &mut Vec<ResidentPlaneLayout>,
    key: CellKey,
    bytes: usize,
    role: BufferRole,
    alignment: usize,
) -> Result<ResidentViewLayout> {
    let reserved = aligned(bytes, alignment)?;
    let index = if let Some(&index) = keys.get(&key) {
        let p = &mut physical[index];
        if p.role != role || p.alignment != alignment {
            return Err(Error::State(
                "shared semantic cell role/alignment differs".into(),
            ));
        }
        p.bytes = p.bytes.max(reserved);
        index
    } else {
        let index = physical.len();
        keys.insert(key, index);
        physical.push(ResidentPlaneLayout {
            bytes: reserved,
            role,
            alignment,
        });
        index
    };
    Ok(ResidentViewLayout {
        plane: index,
        offset: 0,
        bytes,
        role,
        alignment,
    })
}
impl WorkspaceLayout {
    pub(crate) fn metadata_bytes(scopes: &[Arc<TextModelBinding>]) -> Result<usize> {
        let first = scopes
            .first()
            .ok_or_else(|| Error::State("shared geometry source scopes absent".into()))?;
        first
            .native_capture()
            .require_cold_scopes(scopes.iter().map(Arc::as_ref))?;
        let bytes = scopes.iter().try_fold(0usize, |total, scope| {
            let values = scope.record().plan().values().len();
            let sums = scope
                .rms_nodes()
                .len()
                .checked_sub(1)
                .ok_or_else(|| Error::State("shared geometry RMS roles absent".into()))?;
            let views = values
                .checked_add(9)
                .and_then(|n| n.checked_add(sums))
                .ok_or_else(|| Error::Limit("shared geometry metadata view count".into()))?;
            // Cell membership/maps and temporary vectors: <=256B per source
            // value. Include both geometry view/sum copies during preparation.
            let geometry = values
                .checked_mul(256)
                .and_then(|n| {
                    views
                        .checked_mul(2 * std::mem::size_of::<ResidentViewLayout>())
                        .and_then(|v| n.checked_add(v))
                })
                .and_then(|n| {
                    sums.checked_mul(2 * std::mem::size_of::<usize>())
                        .and_then(|v| n.checked_add(v))
                })
                .and_then(|n| n.checked_add(2 * std::mem::size_of::<GeometryLayout>()))
                .ok_or_else(|| Error::Limit("shared geometry metadata extent".into()))?;
            total
                .checked_add(geometry)
                .ok_or_else(|| Error::Limit("shared geometry metadata sum".into()))
        })?;
        Ok(bytes.max(METADATA_BYTES))
    }
    pub(crate) fn view_count(&self) -> Result<usize> {
        self.geometries.iter().try_fold(0usize, |n, g| {
            n.checked_add(g.views.len())
                .ok_or_else(|| Error::Limit("shared view count overflow".into()))
        })
    }
    pub(crate) fn derive(recipes: &[Arc<StorageRecipe>]) -> Result<Self> {
        let first = recipes
            .first()
            .ok_or_else(|| Error::State("shared geometry cohort absent".into()))?;
        let source = first.scope.source();
        let width = first.scope.hidden_size();
        first
            .scope
            .native_capture()
            .require_cold_scopes(recipes.iter().map(|r| r.scope.as_ref()))?;
        let mut keys = BTreeMap::new();
        let mut physical = Vec::new();
        let mut geometries = Vec::with_capacity(recipes.len());
        let mut gamma_leases = 0usize;
        let mut gamma_aligned_bytes = 0usize;
        for recipe in recipes {
            let scope = &recipe.scope;
            if !scope.source().same_issuer(source)
                || scope.hidden_size() != width
                || recipe.mode != StorageMode::ReuseMlp
                || recipe.alignment != first.alignment
                || scope.rms_nodes() != first.scope.rms_nodes()
                || scope.record().plan().precisions() != first.scope.record().plan().precisions()
                || scope.record().plan().numerical_profile()
                    != first.scope.record().plan().numerical_profile()
                || scope.record().plan().required_domains()
                    != first.scope.record().plan().required_domains()
                || scope.record().plan().proposed_domains()
                    != first.scope.record().plan().proposed_domains()
            {
                return Err(Error::State(
                    "shared geometry authentic source/profile/producer classes".into(),
                ));
            }
            let groups = signatures(&recipe.bindings, recipe.slot_bytes.len())?;
            let mut views = Vec::with_capacity(groups.len() + 9 + 48);
            for (slot, members) in groups.into_iter().enumerate() {
                let external = recipe.external[members[0].0];
                if members.iter().any(|&(v, _)| recipe.external[v] != external)
                    || external && (!recipe.dedicated[slot] || members.len() != 1)
                {
                    return Err(Error::State(
                        "shared geometry actor/dedicated slot class".into(),
                    ));
                }
                let role = if external && members[0].0 != 0 {
                    BufferRole::Output
                } else {
                    BufferRole::Activation
                };
                views.push(cell(
                    &mut keys,
                    &mut physical,
                    CellKey::Value(members),
                    recipe.slot_bytes[slot],
                    role,
                    recipe.alignment,
                )?);
            }
            let rows = scope.call().rows().len();
            let private = FullModelRmsScratch::allocation_layout(rows, width)?;
            if private.len() != 9 {
                return Err(Error::State("shared geometry exact nine RMS roles".into()));
            }
            for (role, bytes) in private.into_iter().enumerate() {
                views.push(cell(
                    &mut keys,
                    &mut physical,
                    CellKey::RmsPrivate(role),
                    bytes,
                    BufferRole::Scratch,
                    recipe.alignment,
                )?);
            }
            let mut sums = Vec::with_capacity(48);
            let sum_bytes = rows
                .checked_mul(width)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| Error::Limit("shared geometry raw sum extent".into()))?;
            for &node in scope.rms_nodes().iter().skip(1) {
                sums.push(node);
                views.push(cell(
                    &mut keys,
                    &mut physical,
                    CellKey::DualSum(node),
                    sum_bytes,
                    BufferRole::Scratch,
                    recipe.alignment,
                )?);
            }
            if sums.len() != 48 {
                return Err(Error::State(
                    "shared geometry all48 raw sum histories".into(),
                ));
            }
            gamma_leases = gamma_leases
                .checked_add(49)
                .ok_or_else(|| Error::Limit("shared gamma count".into()))?;
            gamma_aligned_bytes = gamma_aligned_bytes
                .checked_add(49 * aligned(width * 2, recipe.alignment)?)
                .ok_or_else(|| Error::Limit("shared retained gamma bytes".into()))?;
            geometries.push(GeometryLayout {
                views: views.into_boxed_slice(),
                value_slots: recipe.slot_bytes.len(),
                sums: sums.into_boxed_slice(),
            });
        }
        let physical_aligned_bytes = physical.iter().try_fold(0usize, |n, p| {
            n.checked_add(p.bytes)
                .ok_or_else(|| Error::Limit("shared physical cell sum".into()))
        })?;
        Ok(Self {
            physical: physical.into_boxed_slice(),
            geometries: geometries.into_boxed_slice(),
            physical_aligned_bytes,
            gamma_leases,
            gamma_aligned_bytes,
        })
    }
}

/// Prepared owners only. Current record/cursor and original workspace guard
/// remain necessary; no cold prototype receives dispatch or Ready authority.
pub(crate) struct SharedValueWorkspace {
    recipes: Box<[Arc<StorageRecipe>]>,
    geometries: Box<[Arc<ValueWorkspaceGeometry>]>,
    _backing: Arc<ResidentWorkspace>,
    _metadata: Arc<WindowWorkspace>,
}
pub(crate) struct ValueWorkspaceGeometry {
    pub(crate) owner: Arc<ResidentWorkspaceGeometry>,
    pub(crate) recipe: Arc<StorageRecipe>,
    layout: GeometryLayout,
    _metadata: Arc<WindowWorkspace>,
}
impl SharedValueWorkspace {
    pub(crate) fn prepare(
        scopes: &[Arc<TextModelBinding>],
        service: Arc<ResidentService>,
        limits: ValueStorageLimits,
    ) -> Result<Arc<Self>> {
        let first = scopes
            .first()
            .ok_or_else(|| Error::State("shared value scopes absent".into()))?;
        let metadata = Arc::new(
            first
                .source()
                .reserve_workspace(WorkspaceLayout::metadata_bytes(scopes)?)?,
        );
        let recipes = scopes
            .iter()
            .map(|s| StorageRecipe::new(s.clone(), StorageMode::ReuseMlp, limits))
            .collect::<Result<Vec<_>>>()?;
        let layout = WorkspaceLayout::derive(&recipes)?;
        let max_views = layout.view_count()?;
        let backing_metadata = first
            .source()
            .reserve_workspace(ResidentWorkspace::metadata_bytes(
                layout.physical.len(),
                max_views,
            )?)?
            .into_host_workspace()?;
        let backing =
            ResidentWorkspace::prepare(service, backing_metadata, &layout.physical, max_views)?;
        let geometries = recipes
            .iter()
            .zip(layout.geometries.iter())
            .map(|(recipe, layout)| {
                Ok(Arc::new(ValueWorkspaceGeometry {
                    owner: backing.geometry(&layout.views)?,
                    recipe: recipe.clone(),
                    layout: layout.clone(),
                    _metadata: metadata.clone(),
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(Self {
            recipes: recipes.into_boxed_slice(),
            geometries: geometries.into_boxed_slice(),
            _backing: backing,
            _metadata: metadata,
        }))
    }
    pub(crate) fn geometry(&self, index: usize) -> Result<Arc<ValueWorkspaceGeometry>> {
        let geometry = self
            .geometries
            .get(index)
            .ok_or_else(|| Error::State("shared geometry index".into()))?;
        if !Arc::ptr_eq(&geometry.recipe, &self.recipes[index]) {
            return Err(Error::State("shared sealed recipe differs".into()));
        }
        Ok(geometry.clone())
    }
}
impl ValueWorkspaceGeometry {
    pub(crate) fn require_scope(&self, scope: &Arc<TextModelBinding>) -> Result<()> {
        if !Arc::ptr_eq(scope, &self.recipe.scope) {
            return Err(Error::State(
                "shared geometry original scope differs".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn slots(&self) -> &[ResidentLease] {
        &self.owner.leases()[..self.layout.value_slots]
    }
    pub(crate) fn rms_views(&self) -> Result<Arc<crate::cuda::rms::RmsWorkspaceViews>> {
        let n = self.layout.value_slots;
        let scratch = self.owner.leases()[n..n + 9].to_vec();
        let mut outputs = BTreeMap::new();
        use crate::cuda::rms::RmsWorkspaceOutput as O;
        outputs.insert(
            (1, O::Embedding),
            self.slots()[self.recipe.bindings[0].slot].clone(),
        );
        outputs.insert(
            (1, O::Normalized),
            self.slots()[self.recipe.bindings[1].slot].clone(),
        );
        for (index, &node) in self.layout.sums.iter().enumerate() {
            let actual = &self.recipe.scope.record().plan().nodes()[node];
            let [normalized, residual] = actual.outputs.as_slice() else {
                return Err(Error::State("shared dual actual two outputs".into()));
            };
            outputs.insert((node, O::Sum), self.owner.leases()[n + 9 + index].clone());
            outputs.insert(
                (node, O::Normalized),
                self.slots()[self.recipe.bindings[normalized.0 as usize].slot].clone(),
            );
            outputs.insert(
                (node, O::Residual),
                self.slots()[self.recipe.bindings[residual.0 as usize].slot].clone(),
            );
        }
        crate::cuda::rms::RmsWorkspaceViews::bind(
            self.recipe.scope.record(),
            self.owner.clone(),
            scratch,
            outputs,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn permuted_packing_matches_semantic_members_not_slot_numbers() {
        let a = [
            ValueStorageBinding {
                slot: 0,
                epoch: 0,
                bytes: 8,
            },
            ValueStorageBinding {
                slot: 1,
                epoch: 0,
                bytes: 16,
            },
            ValueStorageBinding {
                slot: 0,
                epoch: 1,
                bytes: 8,
            },
        ];
        let b = a.map(|mut x| {
            x.slot = 1 - x.slot;
            x
        });
        let a = signatures(&a, 2).unwrap();
        let b = signatures(&b, 2).unwrap();
        assert_eq!(a[0], b[1]);
        assert_eq!(a[1], b[0]);
        assert_ne!(a[0], b[0]);
    }
    #[test]
    fn matching_cells_take_actual_maxima_keep_roles_and_exact_views() {
        let mut keys = BTreeMap::new();
        let mut planes = Vec::new();
        let key = CellKey::Value(vec![(7, 0)].into_boxed_slice());
        let a = cell(
            &mut keys,
            &mut planes,
            key.clone(),
            47 * 2048 * 2,
            BufferRole::Output,
            512,
        )
        .unwrap();
        let b = cell(
            &mut keys,
            &mut planes,
            key.clone(),
            72 * 2048 * 2,
            BufferRole::Output,
            512,
        )
        .unwrap();
        assert_eq!(planes.len(), 1);
        assert_eq!(a.plane, b.plane);
        assert_eq!(a.bytes, 192512);
        assert_eq!(b.bytes, 294912);
        assert_eq!(planes[0].bytes, 294912);
        assert!(cell(&mut keys, &mut planes, key, 4, BufferRole::Scratch, 512).is_err());
        cell(
            &mut keys,
            &mut planes,
            CellKey::DualSum(7),
            4,
            BufferRole::Scratch,
            512,
        )
        .unwrap();
        assert_eq!(planes.len(), 2); // Same integer is a different actual producer class.
        assert!(aligned(usize::MAX, 512).is_err());
        assert!(aligned(1, 3).is_err());
        assert!(signatures(&[], 1).is_err());
    }
}
