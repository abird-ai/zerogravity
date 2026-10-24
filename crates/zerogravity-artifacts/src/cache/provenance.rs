//! Pure provider labels shared by acquisition and already acquired cache
//! selection. No URL/transport/client, credentials, filesystem or source grant.
use super::ManifestSource;
use crate::{Error, Result};

pub fn validate_repository(provider: &str, repository: &str) -> Result<()> {
    let parts = repository.split('/').collect::<Vec<_>>();
    match provider {
        "huggingface" => {
            if parts.is_empty()
                || parts.len() > 2
                || parts.iter().any(|p| {
                    p.is_empty()
                        || *p == "."
                        || *p == ".."
                        || !p
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"_-.".contains(&b))
                })
            {
                return Err(Error::Malformed("invalid HF repository identifier".into()));
            }
        }
        "github" => {
            if parts.len() != 2
                || parts.iter().any(|p| {
                    p.is_empty()
                        || p.len() > 100
                        || p.starts_with('.')
                        || !p
                            .bytes()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
                })
            {
                return Err(Error::Malformed(
                    "invalid GitHub metadata or selection".into(),
                ));
            }
        }
        _ => {
            return Err(Error::Unsupported(
                "unregistered cached provider locator".into(),
            ));
        }
    }
    Ok(())
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderLocator {
    pub provider: String,
    pub repository: String,
    pub revision: Option<String>,
}
impl ProviderLocator {
    pub fn parse(provider: &str, value: &str) -> Result<Self> {
        let (repository, revision) = value
            .rsplit_once('@')
            .map_or((value, None), |(r, v)| (r, Some(v)));
        ManifestSource::new(provider, repository, revision.unwrap_or("unspecified"))?;
        validate_repository(provider, repository)?;
        Ok(Self {
            provider: provider.into(),
            repository: repository.into(),
            revision: revision.map(str::to_owned),
        })
    }
    pub fn matches(&self, source: &ManifestSource) -> bool {
        self.provider == source.provider()
            && self.repository == source.repository()
            && self
                .revision
                .as_deref()
                .is_none_or(|r| r == source.revision())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_cached_provenance_and_optional_revision() {
        let source = ManifestSource::new("huggingface", "org/model", "pinned").unwrap();
        assert!(
            ProviderLocator::parse("huggingface", "org/model@pinned")
                .unwrap()
                .matches(&source)
        );
        assert!(
            ProviderLocator::parse("huggingface", "org/model")
                .unwrap()
                .matches(&source)
        );
        assert!(
            !ProviderLocator::parse("huggingface", "org/model@other")
                .unwrap()
                .matches(&source)
        );
        assert!(
            !ProviderLocator::parse("github", "org/model@pinned")
                .unwrap()
                .matches(&source)
        );
        for (provider, value) in [
            ("huggingface", "../model"),
            ("github", "org"),
            ("github", ".org/model"),
            ("huggingface", "org/model@"),
            ("huggingface", "https://secret@host/model"),
        ] {
            assert!(ProviderLocator::parse(provider, value).is_err());
        }
    }
}
