//! One alias map, shared unchanged between erased metadata and typed execution.
use super::{ServiceError, worker::SchedulerSnapshot};
use crate::model_instance::{
    ModelInstance, PreparedModel,
    execution::{Executor, ModelHandle},
};
use std::{collections::BTreeMap, sync::Arc};

pub(crate) struct Registry<A: Executor> {
    pub models: BTreeMap<String, A::Model>,
}
pub(crate) trait Models: Send + Sync {
    fn model(&self, alias: &str) -> Option<Arc<dyn PreparedModel>>;
    fn count(&self) -> usize;
    fn alias(&self, index: usize) -> Option<&str>;
    fn admission(&self, snapshot: &SchedulerSnapshot) -> Result<serde_json::Value, ServiceError>;
    fn retirement_pending(&self) -> bool;
}
impl<A: Executor> Models for Registry<A> {
    fn model(&self, alias: &str) -> Option<Arc<dyn PreparedModel>> {
        self.models.get(alias).map(ModelHandle::erased)
    }
    fn count(&self) -> usize {
        self.models.len()
    }
    fn alias(&self, index: usize) -> Option<&str> {
        self.models.keys().nth(index).map(String::as_str)
    }
    fn admission(&self, snapshot: &SchedulerSnapshot) -> Result<serde_json::Value, ServiceError> {
        A::diagnostics(&self.models, snapshot)
    }
    fn retirement_pending(&self) -> bool {
        A::retirement_pending(&self.models)
    }
}
/// Allocation-free cold enumeration over the bounded alias registry.
pub struct ModelIds<'a> {
    pub(crate) models: &'a dyn Models,
    pub(crate) index: usize,
}
impl<'a> Iterator for ModelIds<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        let alias = self.models.alias(self.index)?;
        self.index += 1;
        Some(alias)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.models.count() - self.index;
        (left, Some(left))
    }
    fn count(self) -> usize {
        self.models.count() - self.index
    }
}
impl ExactSizeIterator for ModelIds<'_> {}
impl std::iter::FusedIterator for ModelIds<'_> {}

pub(crate) fn instance(models: &dyn Models, alias: &str) -> Option<ModelInstance> {
    models.model(alias).map(ModelInstance::new)
}
