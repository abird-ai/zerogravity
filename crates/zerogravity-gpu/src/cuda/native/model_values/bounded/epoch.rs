//! Ordered submission epochs only; no physical initialized/Ready transitions.
use super::*;

pub(super) struct Cursor {
    started: bool,
    finished: bool,
    failed: bool,
    next: usize,
    nodes: usize,
    active: Option<usize>,
    owners: Vec<Option<(ValueId, usize)>>,
    produced: Vec<bool>,
}
impl Cursor {
    pub(super) fn new(recipe: &StorageRecipe) -> Self {
        Self::empty(
            recipe.bindings.len(),
            recipe.slot_bytes.len(),
            recipe.scope.record().plan().nodes().len(),
        )
    }
    pub(super) fn empty(values: usize, slots: usize, nodes: usize) -> Self {
        Self {
            started: false,
            finished: false,
            failed: false,
            next: 0,
            nodes,
            active: None,
            owners: vec![None; slots],
            produced: vec![false; values],
        }
    }
    pub(super) fn begin(&mut self) -> Result<()> {
        if self.started || self.finished || self.failed {
            self.failed = true;
            return Err(Error::State("bounded actual record/run consumed".into()));
        }
        self.started = true;
        Ok(())
    }
    pub(super) fn before(
        &mut self,
        recipe: &StorageRecipe,
        index: usize,
        node: &PlanNode,
    ) -> Result<()> {
        self.before_parts(
            &recipe.bindings,
            &recipe.lives,
            index,
            &node.inputs,
            &node.outputs,
        )
    }
    pub(super) fn before_parts(
        &mut self,
        bindings: &[ValueStorageBinding],
        lives: &[ValueLifetime],
        index: usize,
        inputs: &[ValueId],
        outputs: &[ValueId],
    ) -> Result<()> {
        let result = self.check_before(bindings, lives, index, inputs, outputs);
        if result.is_err() {
            self.failed = true;
        }
        result?;
        self.active = Some(index);
        Ok(())
    }
    fn check_before(
        &self,
        bindings: &[ValueStorageBinding],
        lives: &[ValueLifetime],
        index: usize,
        inputs: &[ValueId],
        outputs: &[ValueId],
    ) -> Result<()> {
        if !self.started
            || self.finished
            || self.failed
            || self.active.is_some()
            || self.next != index
            || index >= self.nodes
            || bindings.len() != self.produced.len()
            || lives.len() != bindings.len()
        {
            return Err(Error::State(
                "bounded actual original node/certificate order".into(),
            ));
        }
        for id in inputs {
            let i = id.0 as usize;
            let b = bindings
                .get(i)
                .ok_or_else(|| Error::State("bounded input index".into()))?;
            if !self.produced[i]
                || self.owners.get(b.slot) != Some(&Some((*id, b.epoch)))
                || lives[i].producer >= index
                || lives[i].last_use < index
            {
                return Err(Error::State(
                    "bounded input logical overwrite epoch not produced/live".into(),
                ));
            }
        }
        for (output_index, id) in outputs.iter().enumerate() {
            let i = id.0 as usize;
            let b = bindings
                .get(i)
                .ok_or_else(|| Error::State("bounded output index".into()))?;
            let owner = self
                .owners
                .get(b.slot)
                .ok_or_else(|| Error::State("bounded output slot".into()))?;
            if self.produced[i]
                || lives[i].producer != index
                || lives[i].last_use < index
                || outputs[..output_index]
                    .iter()
                    .any(|prior| bindings[prior.0 as usize].slot == b.slot)
                || inputs
                    .iter()
                    .any(|input| bindings[input.0 as usize].slot == b.slot)
            {
                return Err(Error::State(
                    "bounded actual output/input alias or duplicate producer".into(),
                ));
            }
            match owner {
                None if b.epoch == 0 => {}
                Some((prior, epoch))
                    if epoch.checked_add(1) == Some(b.epoch)
                        && lives[prior.0 as usize].last_use < index => {}
                _ => {
                    return Err(Error::State(
                        "bounded overwrite precedes final ordered consumer".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    pub(super) fn after(
        &mut self,
        recipe: &StorageRecipe,
        index: usize,
        node: &PlanNode,
    ) -> Result<()> {
        self.after_parts(&recipe.bindings, index, &node.outputs)
    }
    pub(super) fn after_parts(
        &mut self,
        bindings: &[ValueStorageBinding],
        index: usize,
        outputs: &[ValueId],
    ) -> Result<()> {
        if self.failed || self.finished || self.active != Some(index) || self.next != index {
            self.failed = true;
            return Err(Error::State(
                "bounded certificate boundary without whole actual producer".into(),
            ));
        }
        // Called by the SAME original loop only after every output certificate.
        for id in outputs {
            let i = id.0 as usize;
            let b = bindings
                .get(i)
                .ok_or_else(|| Error::State("bounded certified output index".into()))?;
            let owner = self
                .owners
                .get_mut(b.slot)
                .ok_or_else(|| Error::State("bounded certified output slot".into()))?;
            *owner = Some((*id, b.epoch));
            self.produced[i] = true;
        }
        self.active = None;
        self.next += 1;
        Ok(())
    }
    pub(super) fn finish(&mut self) -> Result<()> {
        if !self.started
            || self.finished
            || self.failed
            || self.next != self.nodes
            || self.active.is_some()
        {
            self.failed = true;
            return Err(Error::State(
                "bounded whole graph incomplete before greedy consumer".into(),
            ));
        }
        self.finished = true;
        Ok(())
    }
    pub(super) fn require_finished(&self) -> Result<()> {
        if self.started
            && self.finished
            && !self.failed
            && self.next == self.nodes
            && self.active.is_none()
        {
            Ok(())
        } else {
            Err(Error::State(
                "bounded complete traversal/greedy boundary absent".into(),
            ))
        }
    }
    pub(super) fn require_active(&self, node: usize) -> Result<()> {
        if self.started
            && !self.finished
            && !self.failed
            && self.next == node
            && self.active == Some(node)
        {
            Ok(())
        } else {
            Err(Error::State(
                "bounded actual original node is not active".into(),
            ))
        }
    }
}

#[cfg(test)]
mod output_alias_tests {
    use super::*;
    #[test]
    fn distinct_output_ids_cannot_write_the_same_physical_slot() {
        let bindings = [
            ValueStorageBinding {
                slot: 0,
                epoch: 0,
                bytes: 512,
            },
            ValueStorageBinding {
                slot: 0,
                epoch: 0,
                bytes: 512,
            },
        ];
        let lives = [ValueLifetime {
            producer: 0,
            last_use: 1,
        }; 2];
        for outputs in [[ValueId(0), ValueId(1)], [ValueId(0), ValueId(0)]] {
            let mut cursor = Cursor::empty(2, 1, 1);
            cursor.begin().unwrap();
            assert!(
                cursor
                    .before_parts(&bindings, &lives, 0, &[], &outputs)
                    .is_err()
            );
            assert!(cursor.after_parts(&bindings, 0, &outputs).is_err());
            assert!(cursor.finish().is_err());
        }
        // A distinct physical slot is a genuine second output, not a duplicate.
        let bindings = [
            bindings[0],
            ValueStorageBinding {
                slot: 1,
                ..bindings[1]
            },
        ];
        let mut cursor = Cursor::empty(2, 2, 1);
        cursor.begin().unwrap();
        cursor
            .before_parts(&bindings, &lives, 0, &[], &[ValueId(0), ValueId(1)])
            .unwrap();
        cursor
            .after_parts(&bindings, 0, &[ValueId(0), ValueId(1)])
            .unwrap();
        cursor.finish().unwrap();
    }
}
