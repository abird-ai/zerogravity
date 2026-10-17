//! Fixed native Azure read profiles for the original HTTP request owner.
use super::{AzureRequestKind, HttpRequest, HttpTarget};
use crate::{Error, Result};
use base64::{Engine, engine::general_purpose::STANDARD};
use std::collections::BTreeMap;
use std::fmt;
use std::time::SystemTime;

pub(super) const API_VERSION: &str = "2023-11-03";
const MIN_SAS_VERSION: &str = "2019-12-12";
#[derive(Clone)]
pub struct AzureSharedKey {
    account: String,
    key: [u8; 64],
}
impl AzureSharedKey {
    pub fn new(account: &str, key: &str) -> Result<Self> {
        account_checked(account)?;
        if key.len() != 88 {
            return Err(invalid());
        }
        let decoded = STANDARD.decode(key).map_err(|_| invalid())?;
        if STANDARD.encode(&decoded) != key {
            return Err(invalid());
        }
        let key = decoded.try_into().map_err(|_| invalid())?;
        Ok(Self {
            account: account.to_owned(),
            key,
        })
    }
    pub(super) fn account(&self) -> &str {
        &self.account
    }
}
impl fmt::Debug for AzureSharedKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureSharedKey").finish_non_exhaustive()
    }
}
#[derive(Clone)]
pub(super) enum Auth {
    SharedKey(AzureSharedKey),
    Sas(Sas),
}
#[derive(Clone)]
pub(super) struct Sas {
    raw: String,
    resource: String,
    permissions: String,
}
impl Auth {
    pub(super) fn sas(raw: &str) -> Result<Self> {
        if raw.is_empty()
            || raw.len() > 8192
            || !raw
                .bytes()
                .all(|byte| byte.is_ascii_graphic() && !b"?#;\\+".contains(&byte))
        {
            return Err(invalid());
        }
        let mut fields = BTreeMap::new();
        for (count, pair) in raw.split('&').enumerate() {
            if count == 32 {
                return Err(Error::Limit("Azure SAS field bound".into()));
            }
            let (key, value) = pair.split_once('=').ok_or_else(invalid)?;
            if !matches!(
                key,
                "sv" | "sr" | "sp" | "se" | "sig" | "st" | "spr" | "sip"
            ) {
                return Err(Error::Unsupported(
                    "Azure SAS requires the bounded ad-hoc service profile".into(),
                ));
            }
            let value = decode(value)?;
            if value.is_empty()
                || value.len() > 2048
                || !value
                    .bytes()
                    .all(|byte| byte.is_ascii_graphic() && !b";?#&\\\"".contains(&byte))
                || fields.insert(key, value).is_some()
            {
                return Err(invalid());
            }
        }
        let required = |key| fields.get(key).map(String::as_str).ok_or_else(invalid);
        let sv = required("sv")?;
        date_version(sv)?;
        if !(MIN_SAS_VERSION..=API_VERSION).contains(&sv) {
            return Err(Error::Unsupported(
                "Azure SAS version is outside the qualified profile".into(),
            ));
        }
        if required("spr")? != "https"
            || !matches!(required("sr")?, "b" | "bv" | "bs" | "c")
            || !matches!(required("sp")?, "r" | "rl")
            || (required("sr")? != "c" && required("sp")? != "r")
        {
            return Err(Error::Unsupported(
                "Azure SAS requires HTTPS and read/list service scope".into(),
            ));
        }
        if required("se")?.len() > 128
            || fields.get("st").is_some_and(|value| value.len() > 128)
            || fields.get("sip").is_some_and(|value| value.len() > 128)
        {
            return Err(invalid());
        }
        let signature = STANDARD.decode(required("sig")?).map_err(|_| invalid())?;
        if signature.len() != 32 || STANDARD.encode(signature) != required("sig")? {
            return Err(invalid());
        }
        Ok(Self::Sas(Sas {
            raw: raw.to_owned(),
            resource: required("sr")?.to_owned(),
            permissions: required("sp")?.to_owned(),
        }))
    }
}
pub(super) struct Prepared {
    pub uri: String,
    pub headers: Vec<(&'static str, String)>,
}
pub(super) fn prepare(
    auth: Option<&Auth>,
    target: &HttpTarget,
    request: &HttpRequest,
    head: bool,
    now: SystemTime,
) -> Result<Prepared> {
    let kind = target.azure_kind().ok_or_else(invalid)?;
    if head && kind == AzureRequestKind::List {
        return Err(invalid());
    }
    let mut uri = target.url();
    let mut headers = vec![("x-ms-version", API_VERSION.to_owned())];
    match auth {
        None => {}
        Some(Auth::Sas(sas)) => {
            let allowed = match kind {
                AzureRequestKind::Content => sas.resource == "b" || sas.resource == "c",
                AzureRequestKind::List => sas.resource == "c" && sas.permissions == "rl",
                AzureRequestKind::Version => sas.resource == "bv" || sas.resource == "c",
                AzureRequestKind::Snapshot => sas.resource == "bs" || sas.resource == "c",
            };
            if !allowed {
                return Err(Error::Unsupported(
                    "Azure SAS resource or permission differs from target".into(),
                ));
            }
            // api-version is not part of a service-SAS signature. Never parse or
            // reserialize the retained raw escaped credential query here.
            uri.push(if target.uri.query().is_some() {
                '&'
            } else {
                '?'
            });
            uri.push_str("api-version=");
            uri.push_str(API_VERSION);
            uri.push('&');
            uri.push_str(&sas.raw);
            if uri.len() > 16 * 1024 {
                return Err(Error::Limit("Azure authenticated URI bound".into()));
            }
        }
        Some(Auth::SharedKey(credentials)) => {
            let date = rfc1123(now)?;
            let authorization = authorization(
                credentials,
                if head { "HEAD" } else { "GET" },
                &target.uri,
                request
                    .range
                    .as_ref()
                    .map(|range| format!("bytes={}-", range.start))
                    .as_deref(),
                &date,
                target.if_match(),
            )?;
            headers.push(("x-ms-date", date));
            headers.push(("Authorization", authorization));
        }
    }
    Ok(Prepared { uri, headers })
}
fn authorization(
    credentials: &AzureSharedKey,
    method: &str,
    uri: &ureq::http::Uri,
    range: Option<&str>,
    date: &str,
    if_match: Option<&str>,
) -> Result<String> {
    let mut canonical = format!("/{}{}", credentials.account, uri.path());
    let mut queries = BTreeMap::<String, Vec<String>>::new();
    if let Some(query) = uri.query() {
        for pair in query.split('&') {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            queries
                .entry(decode(key)?.to_ascii_lowercase())
                .or_default()
                .push(decode(value)?);
        }
    }
    for (key, mut values) in queries {
        values.sort_unstable();
        canonical.push('\n');
        canonical.push_str(&key);
        canonical.push(':');
        canonical.push_str(&values.join(","));
    }
    let preimage = format!(
        "{method}\n\n\n\n\n\n\n\n{}\n\n\n{}\nx-ms-date:{date}\nx-ms-version:{API_VERSION}\n{canonical}",
        if_match.unwrap_or(""),
        range.unwrap_or("")
    );
    let digest =
        super::s3_signing::mac(&credentials.key, preimage.as_bytes()).map_err(|_| invalid())?;
    Ok(format!(
        "SharedKey {}:{}",
        credentials.account,
        STANDARD.encode(digest)
    ))
}
pub(super) fn check_origin(origin: &str, account: Option<&str>, diagnostic: bool) -> Result<()> {
    let parsed = super::parse_url(origin, diagnostic)?;
    super::validate_origin_url(&parsed)?;
    let authority = origin
        .split_once("://")
        .map(|(_, value)| value)
        .ok_or_else(invalid)?;
    if origin.chars().any(char::is_control)
        || origin.contains(['@', '?', '#', '\\'])
        || !matches!(authority.split_once('/'), None | Some((_, "")))
        || origin.trim() != origin
    {
        return Err(invalid());
    }
    if diagnostic {
        return Ok(());
    }
    let actual = parsed
        .host_str()
        .and_then(|host| host.strip_suffix(".blob.core.windows.net"))
        .ok_or_else(|| {
            Error::Unsupported("Azure credentials require native account origin".into())
        })?;
    account_checked(actual)?;
    if parsed.port().is_some() || account.is_some_and(|expected| expected != actual) {
        return Err(Error::State(
            "Azure credential account and origin differ".into(),
        ));
    }
    Ok(())
}
fn account_checked(account: &str) -> Result<()> {
    if !(3..=24).contains(&account.len())
        || !account
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    {
        return Err(invalid());
    }
    Ok(())
}
fn invalid() -> Error {
    Error::Malformed("invalid explicit Azure credentials or signature inputs".into())
}
fn decode(raw: &str) -> Result<String> {
    let mut bytes = Vec::with_capacity(raw.len());
    let raw = raw.as_bytes();
    let mut offset = 0;
    while offset < raw.len() {
        if raw[offset] == b'%' {
            let a = *raw.get(offset + 1).ok_or_else(invalid)?;
            let b = *raw.get(offset + 2).ok_or_else(invalid)?;
            let digit = |byte: u8| {
                char::from(byte)
                    .to_digit(16)
                    .map(|value| value as u8)
                    .ok_or_else(invalid)
            };
            bytes.push(digit(a)? * 16 + digit(b)?);
            offset += 3;
        } else {
            bytes.push(raw[offset]);
            offset += 1;
        }
    }
    String::from_utf8(bytes).map_err(|_| invalid())
}
fn date_version(value: &str) -> Result<()> {
    if value.len() != 10
        || value.as_bytes()[4] != b'-'
        || value.as_bytes()[7] != b'-'
        || !value
            .bytes()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
    {
        return Err(invalid());
    }
    let year = value[..4].parse().map_err(|_| invalid())?;
    let month = value[5..7].parse::<u8>().map_err(|_| invalid())?;
    let day = value[8..].parse().map_err(|_| invalid())?;
    time::Date::from_calendar_date(
        year,
        time::Month::try_from(month).map_err(|_| invalid())?,
        day,
    )
    .map_err(|_| invalid())?;
    Ok(())
}
fn rfc1123(now: SystemTime) -> Result<String> {
    let datetime = super::s3_signing::checked_utc(now).map_err(|_| invalid())?;
    let weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    Ok(format!(
        "{}, {:02} {} {:04} {:02}:{:02}:{:02} GMT",
        weekdays[datetime.weekday().number_days_from_monday() as usize],
        datetime.day(),
        months[u8::from(datetime.month()) as usize - 1],
        datetime.year(),
        datetime.hour(),
        datetime.minute(),
        datetime.second()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    const KEY: &str =
        "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+Pw==";
    const DATE: &str = "Fri, 17 Oct 2025 01:02:03 GMT";
    const SAS: &str = "sv=2023-11-03&sr=bv&sp=r&se=2030-01-01T00%3a00%3A00Z&spr=https&sig=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA%3d";

    #[test]
    fn shared_key_published_preimage_independent_literal_hmac() {
        // SharedKey REST's twelve standard fields, fixed x-ms headers and the
        // encoded path / decoded sorted query. Literal expected HMACs were made
        // independently with Python stdlib hmac/hashlib/base64 and key bytes0..63;
        // no expected signature calls this implementation's HMAC/preimage code.
        // Canonical resource (after the x-ms-version line):
        // /fixtureaccount/container/dir/a%20b.bin
        // a:1,2
        // versionid:2025-10-16T12:00:00.0000000Z
        // z:raw+plus
        let credentials = AzureSharedKey::new("fixtureaccount", KEY).unwrap();
        let uri = "https://fixtureaccount.blob.core.windows.net/container/dir/a%20b.bin?z=raw%2Bplus&a=2&versionid=2025-10-16T12%3A00%3A00.0000000Z&a=1".parse().unwrap();
        for (method, range, expected) in [
            ("GET", None, "sdGfD5ed+KL5kvzLoGbxtR7SfPfd+S8V6BG+6wSvMpM="),
            ("HEAD", None, "/DsgwMUjPZLo07kwcaaUCE98vExiA+xfrnlv9fV96bk="),
            (
                "GET",
                Some("bytes=3-"),
                "kT57faY27Mt8fxEGkuewXzYIRdGSRlHd+/0Yr+UdE1M=",
            ),
        ] {
            assert_eq!(
                authorization(&credentials, method, &uri, range, DATE, None).unwrap(),
                format!("SharedKey fixtureaccount:{expected}")
            );
        }
        assert_eq!(
            rfc1123(UNIX_EPOCH + Duration::from_secs(1760662923)).unwrap(),
            DATE
        );
        assert!(rfc1123(UNIX_EPOCH - Duration::from_secs(40_000_000_000)).is_err());
        assert!(!format!("{credentials:?}").contains(KEY));
    }

    #[test]
    fn conditional_shared_key_literal_preimages_include_standard_if_match_slot() {
        // Independent Python stdlib, twelve-field REST preimages: field8 is
        // If-Match, field11 is Range. Existing None vectors above remain exact.
        let credentials = AzureSharedKey::new("fixtureaccount", KEY).unwrap();
        let uri = "https://fixtureaccount.blob.core.windows.net/container/dir/a%20b.bin"
            .parse()
            .unwrap();
        for (range, expected) in [
            (None, "1y8XKbNvJqlUlX5Y320ewRAMq0F1scHSslUgI9b6rG4="),
            (
                Some("bytes=3-"),
                "MES9TgHku9q6TUBtP9TaakOkKlWKiOtDd5p/13Cag9M=",
            ),
        ] {
            assert_eq!(
                authorization(
                    &credentials,
                    "GET",
                    &uri,
                    range,
                    DATE,
                    Some("\"observed-2\"")
                )
                .unwrap(),
                format!("SharedKey fixtureaccount:{expected}")
            );
        }
    }

    #[test]
    fn sas_preserves_raw_escapes_and_adds_unsigned_fixed_protocol() {
        let auth = Auth::sas(SAS).unwrap();
        let target = HttpTarget::azure(
            "https://fixtureaccount.blob.core.windows.net",
            "/container/a?versionid=opaque%2Bversion",
            AzureRequestKind::Version,
        )
        .unwrap();
        let result = prepare(
            Some(&auth),
            &target,
            &HttpRequest::new(target.url(), 1),
            false,
            UNIX_EPOCH,
        )
        .unwrap();
        assert_eq!(
            result.uri,
            format!("{}&api-version={API_VERSION}&{SAS}", target.url())
        );
        assert_eq!(result.headers, [("x-ms-version", API_VERSION.to_owned())]);
        let snapshot = HttpTarget::azure(
            "https://fixtureaccount.blob.core.windows.net",
            "/container/a?snapshot=opaque",
            AzureRequestKind::Snapshot,
        )
        .unwrap();
        assert!(
            prepare(
                Some(&auth),
                &snapshot,
                &HttpRequest::new(snapshot.url(), 1),
                false,
                UNIX_EPOCH
            )
            .is_err()
        );
        let list = HttpTarget::azure(
            "https://fixtureaccount.blob.core.windows.net",
            "/container?restype=container&comp=list",
            AzureRequestKind::List,
        )
        .unwrap();
        assert!(
            prepare(
                Some(&auth),
                &list,
                &HttpRequest::new(list.url(), 1),
                false,
                UNIX_EPOCH
            )
            .is_err()
        );
        let container = Auth::sas(&SAS.replace("sr=bv&sp=r", "sr=c&sp=rl")).unwrap();
        assert!(
            prepare(
                Some(&container),
                &list,
                &HttpRequest::new(list.url(), 1),
                false,
                UNIX_EPOCH
            )
            .is_ok()
        );
        assert!(
            prepare(
                Some(&container),
                &snapshot,
                &HttpRequest::new(snapshot.url(), 1),
                false,
                UNIX_EPOCH
            )
            .is_ok()
        );
    }

    #[test]
    fn content_sas_resource_is_distinct_from_immutable_revision_targets() {
        let origin = "https://fixtureaccount.blob.core.windows.net";
        let content = HttpTarget::azure(origin, "/container/a", AzureRequestKind::Content).unwrap();
        let version = HttpTarget::azure(
            origin,
            "/container/a?versionid=opaque",
            AzureRequestKind::Version,
        )
        .unwrap();
        let snapshot = HttpTarget::azure(
            origin,
            "/container/a?snapshot=opaque",
            AzureRequestKind::Snapshot,
        )
        .unwrap();
        let list = HttpTarget::azure(
            origin,
            "/container?comp=list&restype=container",
            AzureRequestKind::List,
        )
        .unwrap();
        for (resource, expected) in [
            ("b", [true, false, false, false]),
            ("bv", [false, true, false, false]),
            ("bs", [false, false, true, false]),
            ("c", [true, true, true, false]),
        ] {
            let raw = SAS.replace("sr=bv", &format!("sr={resource}"));
            let auth = Auth::sas(&raw).unwrap();
            for (target, allowed) in [&content, &version, &snapshot, &list]
                .into_iter()
                .zip(expected)
            {
                let actual = prepare(
                    Some(&auth),
                    target,
                    &HttpRequest::new(target.url(), 1),
                    false,
                    UNIX_EPOCH,
                );
                assert_eq!(actual.is_ok(), allowed);
                if allowed {
                    assert!(actual.unwrap().uri.ends_with(&raw));
                }
            }
        }
    }

    #[test]
    fn credentials_refuse_ambiguous_profiles_and_secret_errors_are_static() {
        for raw in [
            SAS.replace("sv=2023-11-03", "sv=2024-01-01"),
            SAS.replace("sv=2023-11-03", "sv=2023-02-30"),
            SAS.replace("spr=https", "spr=http"),
            SAS.replace("sr=bv", "sr=d"),
            SAS.replace("sp=r", "sp=rw"),
            SAS.replace("sp=r", "sp=rl"),
            format!("{SAS}&sv=2023-11-03"),
            format!("{SAS}&api-version=2023-11-03"),
            format!("{SAS}&rsce=identity"),
            format!("{SAS}&si=stored-policy"),
            format!("{SAS}&ss=b"),
            SAS.replace("se=2030", "se=%0A2030"),
            SAS.replace("se=2030", "se=%3b2030"),
            SAS.replace("se=2030", "se=%2030"),
            SAS.replace("%3d", "%G0"),
            SAS.replace("%3d", "+"),
        ] {
            let error = Auth::sas(&raw).err().unwrap();
            assert!(!format!("{error:?} {error}").contains("2030-01-01"));
        }
        for account in ["ab", "UPPERCASE", "a.b", "has-dash"] {
            assert!(AzureSharedKey::new(account, KEY).is_err());
        }
        for key in ["x".to_owned(), STANDARD.encode([0u8; 32]), "x".repeat(8192)] {
            assert!(AzureSharedKey::new("fixtureaccount", &key).is_err());
        }
        assert!(
            check_origin(
                "https://other.blob.core.windows.net",
                Some("fixtureaccount"),
                false
            )
            .is_err()
        );
        for origin in [
            "https://fixtureaccount.blob.core.windows.net/a/..",
            "https://@fixtureaccount.blob.core.windows.net",
            "https://fixtureaccount.blob.core.windows.net?secret",
        ] {
            assert!(check_origin(origin, Some("fixtureaccount"), false).is_err());
        }
    }
}
