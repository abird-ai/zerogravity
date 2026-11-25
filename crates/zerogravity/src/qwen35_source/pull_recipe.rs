//! Exact catalog acquisition recipes are derived from the original compiled
//! source inventories, not a second model registry or heuristic quant choice.
use super::qwen_source_inventory;
use crate::{
    Error, Result,
    acquisition::{Provider, PullRequest, ResolvedPullPlan, Selection},
};

struct Member {
    path: String,
    bytes: u64,
    sha256: String,
}

pub struct QwenPullRecipe {
    target: &'static str,
    snapshot: &'static str,
    repository: &'static str,
    revision: &'static str,
    members: Vec<Member>,
    bytes: u64,
}
impl QwenPullRecipe {
    pub fn for_target(target: &str) -> Result<Self> {
        let spec = qwen_source_inventory::for_target(target)
            .ok_or_else(|| Error::Unsupported("no approved exact acquisition recipe for this model; use model pull with explicit source selection".into()))?;
        // The original .8B snapshot uses a legacy manifest without digest
        // provenance. A modern transfer cannot recreate that exact identity.
        if spec.family == "0.8b" {
            return Err(Error::Unsupported("the approved Qwen 0.8B snapshot uses a legacy manifest; automatic recreation is not qualified".into()));
        }
        let raw: serde_json::Value = serde_json::from_str(spec.inventory)
            .map_err(|_| Error::State("invalid compiled Qwen source inventory".into()))?;
        let rows = raw["files"]
            .as_array()
            .filter(|rows| rows.len() == spec.member_count)
            .ok_or_else(|| Error::State("compiled Qwen source inventory extent".into()))?;
        let mut members = Vec::with_capacity(rows.len());
        let mut bytes = 0u64;
        for row in rows {
            let text = |key| {
                row[key]
                    .as_str()
                    .ok_or_else(|| Error::State("compiled Qwen source member".into()))
            };
            let path = text(if spec.named_members { "name" } else { "path" })?;
            crate::cache::paths::validate_relative_path(path)?;
            let sha256 = text(if spec.named_members {
                "sha256"
            } else {
                "digest"
            })?;
            crate::cache::paths::validate_digest(sha256)?;
            let length = row["bytes"]
                .as_u64()
                .filter(|n| *n != 0)
                .ok_or_else(|| Error::State("compiled Qwen source member length".into()))?;
            bytes = bytes
                .checked_add(length)
                .filter(|bytes| *bytes <= spec.max_source)
                .ok_or_else(|| Error::Limit("compiled Qwen source size".into()))?;
            members.push(Member {
                path: path.into(),
                bytes: length,
                sha256: sha256.into(),
            });
        }
        members.sort_unstable_by(|a, b| a.path.cmp(&b.path));
        if members.windows(2).any(|w| w[0].path == w[1].path) {
            return Err(Error::State("duplicate compiled Qwen source member".into()));
        }
        Ok(Self {
            target: spec.target,
            snapshot: spec.snapshot_sha,
            repository: spec.repository,
            revision: spec.revision,
            members,
            bytes,
        })
    }
    pub fn target(&self) -> &'static str {
        self.target
    }
    pub fn snapshot(&self) -> &'static str {
        self.snapshot
    }
    pub fn repository(&self) -> &'static str {
        self.repository
    }
    pub fn revision(&self) -> &'static str {
        self.revision
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn request(&self) -> PullRequest {
        PullRequest {
            provider: Provider::HuggingFace,
            repository: self.repository.into(),
            revision: self.revision.into(),
            selection: Selection::Files {
                paths: self
                    .members
                    .iter()
                    .map(|member| member.path.clone())
                    .collect(),
                subfolder: None,
            },
        }
    }
    /// Compare the resolved immutable plan before any cache/payload effect.
    /// The actual final manifest must ALSO match the original exact digest.
    pub fn validate_plan(&self, plan: &ResolvedPullPlan) -> Result<()> {
        let source = plan.source();
        if source.provider() != "huggingface"
            || source.repository() != self.repository
            || source.revision() != self.revision
            || plan.total_bytes() != Some(self.bytes)
            || plan.files().len() != self.members.len()
            || plan
                .files()
                .iter()
                .zip(&self.members)
                .any(|(file, member)| {
                    file.path() != member.path
                        || file.bytes() != Some(member.bytes)
                        || file.sha256().is_some_and(|hash| hash != member.sha256)
                })
        {
            return Err(Error::Malformed(
                "resolved acquisition plan differs from approved catalog source".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition::{PullLimits, ResolvedFile};
    use crate::cache::{FileRole, ManifestSource};

    #[test]
    fn exact_catalog_recipes_use_original_member_inventory() {
        assert!(QwenPullRecipe::for_target("qwen3.5-0.8b").is_err());
        assert!(QwenPullRecipe::for_target("unregistered").is_err());
        for target in ["qwen3.5-2b", "qwen3.8-27b"] {
            let recipe = QwenPullRecipe::for_target(target).unwrap();
            assert_eq!(
                Some(recipe.snapshot()),
                crate::models::qwen_source_snapshot(target)
            );
            let source =
                ManifestSource::new("huggingface", recipe.repository(), recipe.revision()).unwrap();
            let build = |wrong: bool| {
                ResolvedPullPlan::new(
                    source.clone(),
                    recipe
                        .members
                        .iter()
                        .enumerate()
                        .map(|(i, m)| {
                            ResolvedFile::new(
                                &m.path,
                                "https://fixture.invalid/object",
                                FileRole::Auxiliary,
                                m.bytes,
                                Some(if wrong && i == 0 {
                                    "0".repeat(64)
                                } else {
                                    m.sha256.clone()
                                }),
                                &m.path,
                            )
                            .unwrap()
                        })
                        .collect(),
                    &PullLimits::default(),
                )
                .unwrap()
            };
            recipe.validate_plan(&build(false)).unwrap();
            assert!(recipe.validate_plan(&build(true)).is_err());
        }
    }
}
