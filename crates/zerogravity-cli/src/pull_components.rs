//! Bounded explicit component inputs. This never infers authority from a model
//! card/config URL or creates a composite cache/publisher.
use super::{Options, ProgressDisplay, admission, limits, open, signals};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::Read;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::acquisition::http::{
    HttpOptions, HttpRequest, HttpResponse, HttpTransport, HttpsClient,
};
use zerogravity::acquisition::{
    self, ComponentKind, ComponentPullRequest, ComponentRequest, ComponentTransport, Provider,
    PullRequest, Selection,
};
use zerogravity::{Error, Result};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanFile {
    schema: String,
    components: Vec<Input>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    label: String,
    role: String,
    provider: String,
    repository: String,
    revision: Option<String>,
    files: Option<Vec<String>>,
    subfolder: Option<String>,
    release: Option<String>,
    asset: Option<String>,
    snapshot: Option<String>,
}
fn invalid() -> Error {
    Error::Malformed(
        "invalid component plan; use the fixed pull components schema (no credentials or URLs)"
            .into(),
    )
}
fn read_plan(
    path: &str,
    bounds: &acquisition::PullLimits,
) -> Result<(ComponentPullRequest, Vec<(String, String)>)> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err(Error::Limit("component plan exceeds1MiB".into()));
    }
    let raw: PlanFile = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
    if raw.schema != "zerogravity.pull.components.v1"
        || raw.components.is_empty()
        || raw.components.len() > 16
    {
        return Err(invalid());
    }
    let mut requests = Vec::new();
    let mut bindings = Vec::new();
    for input in raw.components {
        let role = match input.role.as_str() {
            "weights" => ComponentKind::Weights,
            "config" => ComponentKind::Config,
            "tokenizer" => ComponentKind::Tokenizer,
            "auxiliary" => ComponentKind::Auxiliary,
            _ => return Err(invalid()),
        };
        let provider = match input.provider.as_str() {
            "huggingface" => Provider::HuggingFace,
            "github" => Provider::GitHub,
            _ => return Err(invalid()),
        };
        let (revision, selection) = match (input.revision, input.files, input.release, input.asset)
        {
            (Some(revision), Some(paths), None, None) => (
                revision,
                Selection::Files {
                    paths,
                    subfolder: input.subfolder,
                },
            ),
            (None, None, Some(tag), Some(name)) if input.subfolder.is_none() => {
                (tag, Selection::ReleaseAsset { name })
            }
            _ => return Err(invalid()),
        };
        if let Some(snapshot) = input.snapshot {
            if snapshot.is_empty()
                || snapshot.len() > 4096
                || snapshot.chars().any(char::is_control)
            {
                return Err(invalid());
            }
            bindings.push((input.label.clone(), snapshot));
        }
        requests.push(ComponentRequest::new(
            input.label,
            role,
            PullRequest {
                provider,
                repository: input.repository,
                revision,
                selection,
            },
        )?);
    }
    Ok((ComponentPullRequest::new(requests, bounds)?, bindings))
}
struct Clients(BTreeMap<String, HttpsClient>);
impl ComponentTransport for Clients {
    fn get(&self, label: &str, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        self.0
            .get(label)
            .ok_or_else(|| Error::State("component client binding missing".into()))?
            .get(request, cancel)
    }
}
pub(super) fn run(path: &str, options: &Options) -> Result<()> {
    let bounds = limits(options)?;
    let (request, bindings) = read_plan(path, &bounds)?;
    if options.offline {
        let store = open(options)?;
        let verified = acquisition::open_components(&store, &request, &bindings, &bounds)?;
        for component in verified.components() {
            println!(
                "Component: {}\nSnapshot: {}",
                component.label(),
                component.lease().manifest().digest()?
            );
        }
        eprintln!(
            "Verified independent cached data; no remote freshness, token revocation or executable-model eligibility inferred."
        );
        return Ok(());
    }
    // An offline selector must not be mistaken for an online publication target.
    if !bindings.is_empty() {
        return Err(Error::Malformed(
            "component snapshot selectors are offline-only".into(),
        ));
    }
    let mut clients = BTreeMap::new();
    for component in request.components() {
        let (origin, token) = match component.request().provider {
            Provider::HuggingFace => ("https://huggingface.co", "HF_TOKEN"),
            Provider::GitHub => ("https://api.github.com", "GITHUB_TOKEN"),
        };
        let mut http = HttpOptions::default();
        if let Some(seconds) = options.timeout {
            http.total_timeout = std::time::Duration::from_secs(seconds);
        }
        let mut client = HttpsClient::from_env(
            origin,
            component.request().provider.label(),
            Some(token),
            http,
        )?;
        if component.request().provider == Provider::GitHub {
            client = client.with_credential_origin("https://raw.githubusercontent.com")?;
        }
        clients.insert(component.label().to_owned(), client);
    }
    let clients = Clients(clients);
    let admission = admission(options.memory)?;
    let cancel = Arc::new(AtomicBool::new(false));
    let _signals = signals::CancellationSignals::register(cancel.clone())?;
    let plan = acquisition::resolve_components(&request, &clients, &bounds, &admission, &cancel)?;
    if options.dry_run {
        println!("Selected component bytes: {}", plan.total_bytes());
        for component in plan.components() {
            println!(
                "Component: {}\nProvider: {}\nRepository: {}\nPinned revision: {}",
                component.label(),
                component.plan().source().provider(),
                component.plan().source().repository(),
                component.plan().source().revision()
            );
            for file in component.plan().files() {
                println!(
                    "{} {} {:?}",
                    file.path(),
                    file.bytes(),
                    file.digest_provenance()
                );
            }
        }
        println!("DRY RUN: metadata only; no model-weight transfer or cache publication.");
        return Ok(());
    }
    let mut display = ProgressDisplay::default();
    let mut current = String::new();
    let receipt = acquisition::execute_components_with_progress(
        &open(options)?,
        &plan,
        &clients,
        &admission,
        &cancel,
        &mut |label, event| {
            if current != label {
                eprintln!("Component: {label}");
                current = label.to_string();
                display = ProgressDisplay::default();
            }
            let _ = display.report(event, &mut std::io::stderr());
        },
    )?;
    for component in receipt.verified_set().components() {
        println!(
            "Component: {}\nSnapshot: {}",
            component.label(),
            component.lease().manifest().digest()?
        );
    }
    eprintln!(
        "Independent component snapshots verified; no cross-source atomic ref or model-readiness claim. Earlier snapshots may remain after a later component failure."
    );
    Ok(())
}
