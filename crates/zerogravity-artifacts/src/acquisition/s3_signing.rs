//! Private SigV4 preparation for the original synchronous HTTP owner.
use super::{HttpRequest, HttpTarget};
use crate::{Error, Result};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

const EMPTY_SHA: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Explicit in-memory static or session credentials. No environment, profile,
/// identity-service or metadata-server lookup occurs in this type.
#[derive(Clone)]
pub struct S3Credentials {
    access: String,
    secret: String,
    session: Option<String>,
}
impl S3Credentials {
    pub fn new(access: &str, secret: &str, session: Option<&str>) -> Result<Self> {
        if access.is_empty()
            || access.len() > 128
            || !access
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
            || secret.is_empty()
            || secret.len() > 1024
            || !secret.bytes().all(|b| b.is_ascii_graphic())
            || session.is_some_and(|v| {
                v.is_empty() || v.len() > 4096 || !v.bytes().all(|b| b.is_ascii_graphic())
            })
        {
            return Err(Error::Malformed("invalid explicit S3 credential".into()));
        }
        Ok(Self {
            access: access.to_owned(),
            secret: secret.to_owned(),
            session: session.map(str::to_owned),
        })
    }
}
impl fmt::Debug for S3Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3Credentials").finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) struct Signing {
    region: String,
    credentials: S3Credentials,
}
impl Signing {
    pub(super) fn new(region: &str, credentials: S3Credentials) -> Result<Self> {
        if region.is_empty()
            || region.len() > 64
            || !region
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return Err(Error::Malformed("invalid S3 signing region".into()));
        }
        Ok(Self {
            region: region.to_owned(),
            credentials,
        })
    }
    pub(super) fn headers(
        &self,
        target: &HttpTarget,
        request: &HttpRequest,
        head: bool,
        now: SystemTime,
    ) -> Result<Vec<(&'static str, String)>> {
        if target.s3_region()? != self.region {
            return Err(Error::State("S3 signing region differs from target".into()));
        }
        let timestamp = utc_timestamp(now)?;
        let mut headers = BTreeMap::from([
            (
                "host",
                target
                    .uri
                    .authority()
                    .ok_or_else(invalid)?
                    .as_str()
                    .to_owned(),
            ),
            ("x-amz-content-sha256", EMPTY_SHA.to_owned()),
            ("x-amz-date", timestamp.clone()),
        ]);
        if let Some(token) = &self.credentials.session {
            headers.insert("x-amz-security-token", token.clone());
        }
        if let Some(range) = &request.range {
            headers.insert("range", format!("bytes={}-", range.start));
            headers.insert("if-range", range.validator.clone());
        }
        let authorization = self.authorization(
            if head { "HEAD" } else { "GET" },
            &target.uri,
            &headers,
            &timestamp,
        )?;
        let mut result = vec![
            ("Authorization", authorization),
            ("x-amz-content-sha256", EMPTY_SHA.to_owned()),
            ("x-amz-date", timestamp),
        ];
        if let Some(token) = &self.credentials.session {
            result.push(("x-amz-security-token", token.clone()));
        }
        Ok(result)
    }
    pub(super) fn authorization(
        &self,
        method: &str,
        uri: &ureq::http::Uri,
        headers: &BTreeMap<&str, String>,
        timestamp: &str,
    ) -> Result<String> {
        let mut canonical_headers = String::new();
        for (name, value) in headers {
            canonical_headers.push_str(name);
            canonical_headers.push(':');
            canonical_headers
                .push_str(&value.split_ascii_whitespace().collect::<Vec<_>>().join(" "));
            canonical_headers.push('\n');
        }
        let signed = headers.keys().copied().collect::<Vec<_>>().join(";");
        let canonical = format!(
            "{method}\n{}\n{}\n{canonical_headers}\n{signed}\n{EMPTY_SHA}",
            uri.path(),
            uri.query().unwrap_or("")
        );
        let date = timestamp.get(..8).ok_or_else(invalid)?;
        let scope = format!("{date}/{}/s3/aws4_request", self.region);
        let to_sign = format!(
            "AWS4-HMAC-SHA256\n{timestamp}\n{scope}\n{:x}",
            Sha256::digest(canonical.as_bytes())
        );
        let date_key = mac(
            format!("AWS4{}", self.credentials.secret).as_bytes(),
            date.as_bytes(),
        )?;
        let region_key = mac(&date_key, self.region.as_bytes())?;
        let service_key = mac(&region_key, b"s3")?;
        let signing_key = mac(&service_key, b"aws4_request")?;
        let signature = mac(&signing_key, to_sign.as_bytes())?;
        let signature = signature
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        Ok(format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope},SignedHeaders={signed},Signature={signature}",
            self.credentials.access
        ))
    }
}
pub(super) fn mac(key: &[u8], value: &[u8]) -> Result<[u8; 32]> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).map_err(|_| invalid())?;
    mac.update(value);
    Ok(mac.finalize().into_bytes().into())
}
fn invalid() -> Error {
    Error::Malformed("invalid S3 signature inputs".into())
}
fn utc_timestamp(now: SystemTime) -> Result<String> {
    let datetime = checked_utc(now)?;
    Ok(format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        datetime.year(),
        u8::from(datetime.month()),
        datetime.day(),
        datetime.hour(),
        datetime.minute(),
        datetime.second()
    ))
}
pub(super) fn checked_utc(now: SystemTime) -> Result<time::OffsetDateTime> {
    let seconds = now
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid())?
        .as_secs();
    let datetime =
        time::OffsetDateTime::from_unix_timestamp(i64::try_from(seconds).map_err(|_| invalid())?)
            .map_err(|_| invalid())?;
    if !(1000..=9999).contains(&datetime.year()) {
        return Err(invalid());
    }
    Ok(datetime)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_s3_get_range_and_listing_signature_vectors() {
        let signer = Signing::new(
            "us-east-1",
            S3Credentials::new(
                "AKIAIOSFODNN7EXAMPLE",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let mut headers = BTreeMap::from([
            ("host", "examplebucket.s3.amazonaws.com".into()),
            ("range", "bytes=0-9".into()),
            ("x-amz-content-sha256", EMPTY_SHA.into()),
            ("x-amz-date", "20130524T000000Z".into()),
        ]);
        let uri: ureq::http::Uri = "https://examplebucket.s3.amazonaws.com/test.txt"
            .parse()
            .unwrap();
        assert!(
            signer
                .authorization("GET", &uri, &headers, "20130524T000000Z")
                .unwrap()
                .ends_with(
                    "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
                )
        );
        headers.remove("range");
        for (query, expected) in [
            (
                "lifecycle=",
                "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543",
            ),
            (
                "max-keys=2&prefix=J",
                "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7",
            ),
        ] {
            let uri = format!("https://examplebucket.s3.amazonaws.com/?{query}")
                .parse()
                .unwrap();
            assert!(
                signer
                    .authorization("GET", &uri, &headers, "20130524T000000Z")
                    .unwrap()
                    .ends_with(expected)
            );
        }
    }
    #[test]
    fn utc_rollover_and_credentials_are_bounded_without_local_timezone() {
        assert_eq!(
            utc_timestamp(UNIX_EPOCH + std::time::Duration::from_secs(1_704_067_199)).unwrap(),
            "20231231T235959Z"
        );
        assert_eq!(
            utc_timestamp(UNIX_EPOCH + std::time::Duration::from_secs(1_704_067_200)).unwrap(),
            "20240101T000000Z"
        );
        for (access, secret, token) in [
            ("", "secret", None),
            ("slash/id", "secret", None),
            ("id", "bad secret", None),
            ("id", "secret", Some("bad\ntoken")),
        ] {
            assert!(S3Credentials::new(access, secret, token).is_err());
        }
        assert!(
            !format!(
                "{:?}",
                S3Credentials::new("id", "SECRET", Some("TOKEN")).unwrap()
            )
            .contains("SECRET")
        );
    }
}
