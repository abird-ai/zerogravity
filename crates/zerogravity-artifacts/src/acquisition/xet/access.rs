//! Ephemeral repository/read-scope grants and reconstruction requests. All HTTP
//! execution remains in HttpTransport's typed capability and original owner.
use super::budget::Budget;
use super::multipart::maximum_body;
use super::reader::RangeSource;
use super::reconstruction::{Group, Reconstruction};
use super::{XetOptions, XetSpec, check_cancel};
use crate::acquisition::http::{
    HttpAccept, HttpRequest, HttpResponse, HttpTarget, HttpTransport, XetGrant, XetHubMode,
};
use crate::{Error, Result};
use serde::Deserialize;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub(super) struct Access<'a> {
    transport: &'a dyn HttpTransport,
    spec: XetSpec,
    bytes: u64,
    options: XetOptions,
    budget: Arc<Budget>,
    grant: Option<Arc<XetGrant>>,
    metadata_limit: usize,
    cancel: &'a AtomicBool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Token {
    access_token: String,
    exp: u64,
    cas_url: String,
}
fn now() -> Result<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|time| time.as_secs())
        .map_err(|_| Error::State("invalid Xet authorization clock".into()))
}
fn metadata_body(response: HttpResponse, maximum: usize, cancel: &AtomicBool) -> Result<Vec<u8>> {
    let declared = response
        .header("content-length")
        .map(crate::acquisition::http::parse_u64)
        .transpose()?;
    if response.status != 200
        || response.header("content-range").is_some()
        || response
            .header("content-encoding")
            .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
        || response
            .header("transfer-encoding")
            .is_some_and(|value| declared.is_some() || !value.eq_ignore_ascii_case("chunked"))
    {
        return Err(Error::Malformed(
            "invalid Xet metadata response framing".into(),
        ));
    }
    if declared.is_some_and(|value| value > maximum as u64) {
        return Err(Error::Limit("Xet metadata response bound".into()));
    }
    let body = response.read_bounded(maximum, cancel)?;
    if declared.is_some_and(|value| value != body.len() as u64) {
        return Err(Error::Malformed("Xet metadata body length changed".into()));
    }
    Ok(body)
}
impl<'a> Access<'a> {
    pub(super) fn new(
        transport: &'a dyn HttpTransport,
        spec: &XetSpec,
        bytes: u64,
        options: &XetOptions,
        budget: Arc<Budget>,
        metadata_limit: usize,
        cancel: &'a AtomicBool,
    ) -> Result<Self> {
        check_cancel(cancel)?;
        Ok(Self {
            transport,
            spec: spec.clone(),
            bytes,
            options: options.clone(),
            budget,
            grant: None,
            metadata_limit,
            cancel,
        })
    }
    fn token(&mut self) -> Result<()> {
        check_cancel(self.cancel)?;
        let mut url = url::Url::parse(&self.options.endpoints.hub)
            .map_err(|_| Error::Malformed("invalid Xet Hub endpoint".into()))?;
        url.path_segments_mut()
            .map_err(|_| Error::Malformed("invalid Xet token authority".into()))?
            .clear()
            .extend(["api", "models"])
            .extend(self.spec.repository.split('/'))
            .extend(["xet-read-token", &self.spec.commit]);
        let mut request = HttpRequest::new(url.as_str(), self.metadata_limit as u64);
        request.accept = HttpAccept::Json;
        let target = HttpTarget::xet_hub(&request.url, self.budget.clone(), XetHubMode::Metadata)?;
        let response = self.transport.get_target(&request, &target, self.cancel)?;
        if response.status != 200 {
            return Err(Error::State(
                "Xet repository read authorization failed".into(),
            ));
        }
        let body = metadata_body(response, self.metadata_limit, self.cancel)?;
        let token: Token = serde_json::from_slice(&body)
            .map_err(|_| Error::Malformed("invalid Xet read authorization metadata".into()))?;
        if token.cas_url != self.options.endpoints.cas || token.exp <= now()?.saturating_add(30) {
            return Err(Error::State(
                "Xet read grant scope or expiry is unsupported".into(),
            ));
        }
        self.grant = Some(XetGrant::new(
            &token.cas_url,
            token.access_token,
            token.exp,
            &self.spec.repository,
            &self.spec.commit,
            self.options.endpoints.diagnostic,
        )?);
        Ok(())
    }
    fn ensure_grant(&mut self) -> Result<()> {
        let now = now()?;
        if self
            .grant
            .as_ref()
            .is_some_and(|grant| grant.expires() > now.saturating_add(30))
        {
            return Ok(());
        }
        if self.grant.is_some() {
            self.budget.refresh(self.cancel)?;
        }
        self.token()
    }
    fn metadata(&mut self, v1: bool) -> Result<HttpResponse> {
        self.ensure_grant()?;
        let version = if v1 { "v1" } else { "v2" };
        let url = format!(
            "{}/{version}/reconstructions/{}",
            self.options.endpoints.cas,
            self.spec.file_id.text()
        );
        let mut request = HttpRequest::new(url, self.metadata_limit as u64);
        request.accept = HttpAccept::Json;
        let target = HttpTarget::xet_cas(
            &request.url,
            self.grant
                .as_ref()
                .ok_or_else(|| Error::State("Xet grant is absent".into()))?
                .clone(),
            self.budget.clone(),
        )?;
        self.transport.get_target(&request, &target, self.cancel)
    }
    pub(super) fn reconstruction(&mut self) -> Result<Reconstruction> {
        let mut v1 = false;
        loop {
            check_cancel(self.cancel)?;
            let response = self.metadata(v1)?;
            match response.status {
                401 => {
                    self.budget.refresh(self.cancel)?;
                    self.grant = None;
                    self.token()?;
                }
                404 | 501 if !v1 => {
                    v1 = true;
                }
                200 => {
                    let body = metadata_body(response, self.metadata_limit, self.cancel)?;
                    let reconstruction = Reconstruction::parse(
                        &body,
                        v1,
                        self.spec.file_id,
                        self.bytes,
                        &self.options,
                        self.cancel,
                    )?;
                    let groups =
                        reconstruction
                            .groups
                            .values()
                            .try_fold(0usize, |sum, group| {
                                sum.checked_add(group.len()).ok_or_else(|| {
                                    Error::Limit("Xet accepted group count overflow".into())
                                })
                            })?;
                    let ranges = reconstruction.groups.values().flatten().try_fold(
                        0usize,
                        |sum, group| {
                            sum.checked_add(group.ranges.len()).ok_or_else(|| {
                                Error::Limit("Xet accepted range count overflow".into())
                            })
                        },
                    )?;
                    self.budget.reconstruction(
                        reconstruction.terms.len(),
                        groups,
                        ranges,
                        self.cancel,
                    )?;
                    return Ok(reconstruction);
                }
                _ => return Err(Error::State("Xet reconstruction request refused".into())),
            }
        }
    }
}
impl RangeSource for Access<'_> {
    fn fetch(&mut self, group: &Group) -> Result<HttpResponse> {
        check_cancel(self.cancel)?;
        let target = HttpTarget::xet_ranges(
            &group.url,
            group
                .ranges
                .iter()
                .map(|descriptor| descriptor.bytes)
                .collect(),
            self.budget.clone(),
        )?;
        let request = HttpRequest::new(target.url(), maximum_body(&group.ranges)?);
        self.transport.get_target(&request, &target, self.cancel)
    }
    fn refresh(&mut self) -> Result<Reconstruction> {
        self.budget.refresh(self.cancel)?;
        self.reconstruction()
    }
}
