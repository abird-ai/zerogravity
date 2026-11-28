//! Metadata-only discovery over the existing verified HTTP and tree resolver.
use super::*;
use crate::acquisition::{
    self, CandidateRequest, Provider, PullLimits,
    http::{HttpAccept, HttpOptions, HttpRequest, HttpTransport, HttpsClient},
};
use std::time::Duration;

fn provider(kind: Kind) -> Result<(Provider, &'static str, &'static str)> {
    match kind {
        Kind::HuggingFace => Ok((Provider::HuggingFace, "https://huggingface.co", "HF_TOKEN")),
        Kind::Github => Ok((Provider::GitHub, "https://api.github.com", "GITHUB_TOKEN")),
        Kind::Local => Err(bad("remote provider required")),
    }
}
fn client(kind: Kind) -> Result<HttpsClient> {
    let (p, origin, token) = provider(kind)?;
    HttpsClient::from_env(
        origin,
        p.label(),
        Some(token),
        HttpOptions {
            total_timeout: Duration::from_secs(20),
            ..Default::default()
        },
    )
}
fn encode(s: &str) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            write!(out, "%{b:02X}").expect("String write");
        }
    }
    out
}
/// One bounded page; HF uses its server-issued cursor, GitHub page numbers.
pub fn browse(
    source: &Source,
    query: &str,
    page: usize,
    cursor: Option<&str>,
    offline: bool,
    cancel: &AtomicBool,
) -> Result<Page> {
    check(source, cancel)?;
    if source.kind == Kind::Local {
        return local(source, query, page, cancel);
    }
    if offline {
        return Err(Error::Unsupported(
            "offline mode prohibits remote discovery".into(),
        ));
    }
    if query.len() > 256 || query.chars().any(char::is_control) {
        return Err(bad("invalid repository query"));
    }
    if let Some(scope) = &source.scope
        && repo(scope)
    {
        return Ok(Page {
            entries: if page == 0 && scope.to_lowercase().contains(&query.to_lowercase()) {
                vec![entry(source.kind, scope.clone())]
            } else {
                vec![]
            },
            more: false,
            next_cursor: None,
        });
    }
    let (p, origin, _) = provider(source.kind)?;
    let url = match p {
        Provider::HuggingFace => {
            if (page != 0) != cursor.is_some()
                || cursor.is_some_and(|v| {
                    v.is_empty() || v.len() > 8192 || v.chars().any(char::is_control)
                })
            {
                return Err(bad("Hugging Face next page requires its returned cursor"));
            }
            let next = cursor
                .map(|v| format!("&cursor={}", encode(v)))
                .unwrap_or_default();
            let author = source
                .scope
                .as_ref()
                .map(|s| format!("&author={}", encode(s)))
                .unwrap_or_default();
            format!(
                "{origin}/api/models?search={}&limit={PAGE}&sort=downloads&direction=-1{author}{next}",
                encode(query)
            )
        }
        Provider::GitHub => {
            if cursor.is_some() {
                return Err(bad("GitHub uses numeric pages"));
            }
            let mut q = if query.trim().is_empty() {
                "model".to_owned()
            } else {
                query.to_owned()
            };
            if let Some(scope) = &source.scope {
                q.push_str(&format!(" user:{scope}"));
            }
            let page = page
                .checked_add(1)
                .filter(|p| *p <= 34)
                .ok_or_else(|| bad("GitHub search page out of range"))?;
            format!(
                "{origin}/search/repositories?q={}&per_page={PAGE}&page={page}",
                encode(&q)
            )
        }
    };
    let mut request = HttpRequest::new(url, 1024 * 1024);
    request.accept = if p == Provider::GitHub {
        HttpAccept::GitHubJson
    } else {
        HttpAccept::Json
    };
    let admission = crate::artifacts::transfer_admission(128 * 1024 * 1024)?;
    let _charge = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[crate::resource_arbiter::AllocationRequest {
            kind: crate::resource_arbiter::AllocationClass::TransferStaging,
            bytes: 16 * 1024 * 1024,
        }],
    )?;
    let response = client(source.kind)?.get_without_redirects(&request, cancel)?;
    let next_cursor = if p == Provider::HuggingFace {
        response
            .header("link")
            .map(acquisition::model_search_cursor)
            .transpose()?
            .flatten()
    } else {
        None
    };
    let bytes = response.read_bounded(1024 * 1024, cancel)?;
    let json: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| bad("invalid provider discovery metadata"))?;
    let items = if p == Provider::GitHub {
        json.get("items")
    } else {
        Some(&json)
    }
    .and_then(|v| v.as_array())
    .ok_or_else(|| bad("invalid provider discovery response"))?;
    if items.len() > PAGE {
        return Err(Error::Limit("provider exceeded discovery page".into()));
    }
    let mut seen = BTreeSet::new();
    let entries = items
        .iter()
        .filter_map(|row| {
            let name = row
                .get(if p == Provider::GitHub {
                    "full_name"
                } else {
                    "id"
                })?
                .as_str()?;
            (repo(name) && in_scope(source, name) && seen.insert(name))
                .then(|| entry(source.kind, name.to_owned()))
        })
        .collect();
    check(source, cancel)?;
    Ok(Page {
        entries,
        more: next_cursor.is_some() || (p == Provider::GitHub && items.len() == PAGE && page < 33),
        next_cursor,
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct File {
    pub path: String,
    pub bytes: Option<u64>,
    pub role: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct Repository {
    pub locator: String,
    pub files: Vec<File>,
    pub payload_hold: bool,
    pub curated: Option<String>,
}
/// Resolve current branch/default or explicitly selected revision, then use the
/// original pinned tree resolver. No weight data or cache publication.
pub fn files(locator: &str, offline: bool, cancel: &AtomicBool) -> Result<Repository> {
    if offline {
        return Err(Error::Unsupported(
            "offline mode prohibits remote discovery".into(),
        ));
    }
    let (kind, value) = match crate::run_policy::SourceSelector::parse(&normalize_locator(locator))?
    {
        crate::run_policy::SourceSelector::HuggingFace(value) => (Kind::HuggingFace, value),
        crate::run_policy::SourceSelector::GitHub(value) => (Kind::Github, value),
        _ => {
            return Err(bad(
                "repository discovery requires a Hugging Face or GitHub locator",
            ));
        }
    };
    let (p, origin, _) = provider(kind)?;
    let loc = crate::cache::provenance::ProviderLocator::parse(p.label(), &value)?;
    let client = client(kind)?;
    let revision = match loc.revision {
        Some(revision) => revision,
        None if p == Provider::HuggingFace => "main".into(),
        None => {
            let mut request =
                HttpRequest::new(format!("{origin}/repos/{}", loc.repository), 1024 * 1024);
            request.accept = HttpAccept::GitHubJson;
            let bytes = client
                .get(&request, cancel)?
                .read_bounded(1024 * 1024, cancel)?;
            let json: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|_| bad("invalid GitHub repository metadata"))?;
            json.get("default_branch")
                .and_then(|s| s.as_str())
                .ok_or_else(|| bad("GitHub default branch unavailable"))?
                .to_owned()
        }
    };
    let request = CandidateRequest::new(p, &loc.repository, &revision, None)?;
    let limits = PullLimits::default();
    let admission = crate::artifacts::transfer_admission(128 * 1024 * 1024)?;
    let observed =
        acquisition::discover_candidates(&client, &request, &limits, &admission, cancel)?;
    let e = entry(kind, loc.repository);
    let files = observed
        .files()
        .iter()
        .map(|f| File {
            path: f.path().into(),
            bytes: f.bytes(),
            role: format!("{:?}", f.role()).to_lowercase(),
        })
        .collect();
    Ok(Repository {
        locator: format!("{}@{}", e.locator, observed.source().revision()),
        files,
        payload_hold: e.payload_hold,
        curated: e.curated,
    })
}
