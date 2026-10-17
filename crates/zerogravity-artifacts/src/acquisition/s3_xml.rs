//! Bounded S3 listing parser. No DTD, external/custom entity or URL following.
use crate::{Error, Result};
use quick_xml::{Reader, events::Event};
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

pub(super) struct Page {
    pub fields: BTreeMap<String, String>,
    pub rows: Vec<Row>,
}
pub(super) struct Row {
    pub key: String,
    pub version: Option<String>,
    pub latest: Option<bool>,
    pub size: Option<u64>,
    pub etag: Option<String>,
    pub deleted: bool,
}
fn invalid() -> Error {
    Error::Malformed("invalid bounded S3 listing XML".into())
}
fn insert(fields: &mut BTreeMap<String, String>, key: String, value: String) -> Result<()> {
    if fields.insert(key, value).is_some() {
        return Err(invalid());
    }
    Ok(())
}
pub(super) fn parse(
    raw: &[u8],
    versions: bool,
    maximum_rows: usize,
    cancel: &AtomicBool,
) -> Result<Page> {
    let input = std::str::from_utf8(raw).map_err(|_| invalid())?;
    for (index, character) in input.chars().enumerate() {
        if index % 4096 == 0 {
            super::http::check_cancel(cancel)?;
        }
        if !xml_character(character) {
            return Err(invalid());
        }
    }
    let mut reader = Reader::from_str(input);
    reader.config_mut().expand_empty_elements = true;
    reader.config_mut().check_end_names = true;
    let root = if versions {
        "ListVersionsResult"
    } else {
        "ListBucketResult"
    };
    let mut stack = Vec::<String>::new();
    let mut fields = BTreeMap::new();
    let mut rows = Vec::new();
    let mut row = None::<(bool, BTreeMap<String, String>)>;
    let mut capture = None::<(usize, String, String)>;
    let mut closed = false;
    let mut seen_decl = false;
    let mut events = 0usize;
    loop {
        super::http::check_cancel(cancel)?;
        events += 1;
        if events > 131072 {
            return Err(Error::Limit("S3 XML event limit".into()));
        }
        match reader.read_event().map_err(|_| invalid())? {
            Event::Start(event) => {
                if closed || stack.len() == 16 || capture.is_some() {
                    return Err(invalid());
                }
                let name = std::str::from_utf8(event.name().as_ref())
                    .map_err(|_| invalid())?
                    .to_owned();
                if name.len() > 64 || name.contains(':') {
                    return Err(invalid());
                }
                if stack.is_empty() {
                    if name != root {
                        return Err(invalid());
                    }
                    for attribute in event.attributes() {
                        let attribute = attribute.map_err(|_| invalid())?;
                        if attribute.key.as_ref() != b"xmlns"
                            || attribute.value.as_ref()
                                != b"http://s3.amazonaws.com/doc/2006-03-01/"
                        {
                            return Err(invalid());
                        }
                    }
                } else if event.attributes().next().is_some() {
                    return Err(invalid());
                }
                if stack.len() == 1 && name == "CommonPrefixes" {
                    return Err(invalid());
                }
                stack.push(name.clone());
                if stack.len() == 2
                    && matches!(name.as_str(), "Contents" | "Version" | "DeleteMarker")
                {
                    if row.is_some()
                        || (!versions && name != "Contents")
                        || (versions && name == "Contents")
                    {
                        return Err(invalid());
                    }
                    row = Some((name == "DeleteMarker", BTreeMap::new()));
                } else if (stack.len() == 2
                    && matches!(
                        name.as_str(),
                        "Name"
                            | "Prefix"
                            | "EncodingType"
                            | "IsTruncated"
                            | "KeyCount"
                            | "MaxKeys"
                            | "ContinuationToken"
                            | "NextContinuationToken"
                            | "KeyMarker"
                            | "VersionIdMarker"
                            | "NextKeyMarker"
                            | "NextVersionIdMarker"
                    ))
                    || (stack.len() == 3
                        && row.is_some()
                        && matches!(
                            name.as_str(),
                            "Key" | "VersionId" | "IsLatest" | "Size" | "ETag"
                        ))
                {
                    capture = Some((stack.len(), name, String::new()));
                }
            }
            Event::End(event) => {
                let name = std::str::from_utf8(event.name().as_ref())
                    .map_err(|_| invalid())?
                    .to_owned();
                if stack.last() != Some(&name) {
                    return Err(invalid());
                }
                if capture
                    .as_ref()
                    .is_some_and(|(depth, _, _)| *depth == stack.len())
                {
                    let (_, name, text) = capture.take().ok_or_else(invalid)?;
                    if stack.len() == 2 {
                        insert(&mut fields, name, text)?;
                    } else {
                        insert(&mut row.as_mut().ok_or_else(invalid)?.1, name, text)?;
                    }
                }
                if stack.len() == 2
                    && matches!(name.as_str(), "Contents" | "Version" | "DeleteMarker")
                {
                    if rows.len() == maximum_rows {
                        return Err(Error::Limit("S3 listing row limit".into()));
                    }
                    let (deleted, mut values) = row.take().ok_or_else(invalid)?;
                    let key = super::decode_key(&values.remove("Key").ok_or_else(invalid)?)?;
                    let version = values.remove("VersionId");
                    let latest = values.remove("IsLatest").map(|v| boolean(&v)).transpose()?;
                    let size = values
                        .remove("Size")
                        .map(|v| super::decimal(&v))
                        .transpose()?;
                    let etag = values.remove("ETag");
                    if (versions && (version.is_none() || latest.is_none()))
                        || (!deleted && (size.is_none() || etag.is_none()))
                    {
                        return Err(invalid());
                    }
                    rows.push(Row {
                        key,
                        version,
                        latest,
                        size,
                        etag,
                        deleted,
                    });
                }
                stack.pop();
                if stack.is_empty() {
                    closed = true;
                }
            }
            Event::Text(event) => {
                let text = event.decode().map_err(|_| invalid())?;
                if let Some((_, _, value)) = &mut capture {
                    append(value, &text)?;
                } else if stack.is_empty() && !text.trim().is_empty() {
                    return Err(invalid());
                }
            }
            Event::GeneralRef(reference) => {
                if stack.is_empty() || closed {
                    return Err(invalid());
                }
                let character =
                    if let Some(character) = reference.resolve_char_ref().map_err(|_| invalid())? {
                        character
                    } else {
                        match reference.decode().map_err(|_| invalid())?.as_ref() {
                            "lt" => '<',
                            "gt" => '>',
                            "amp" => '&',
                            "apos" => '\'',
                            "quot" => '"',
                            _ => return Err(invalid()),
                        }
                    };
                if !xml_character(character) {
                    return Err(invalid());
                }
                if let Some((_, _, value)) = &mut capture {
                    append(value, &character.to_string())?;
                }
            }
            Event::CData(event) => {
                if let Some((_, _, value)) = &mut capture {
                    append(value, &event.decode().map_err(|_| invalid())?)?;
                } else {
                    return Err(invalid());
                }
            }
            Event::Decl(event) => {
                if seen_decl
                    || !stack.is_empty()
                    || closed
                    || event.version().map_err(|_| invalid())?.as_ref() != b"1.0"
                    || event
                        .encoding()
                        .is_some_and(|v| v.map_or(true, |v| !v.eq_ignore_ascii_case(b"UTF-8")))
                {
                    return Err(invalid());
                }
                seen_decl = true;
            }
            Event::Comment(_) => {}
            Event::DocType(_) | Event::PI(_) => {
                return Err(Error::Unsupported(
                    "S3 listing DTD or processing instruction refused".into(),
                ));
            }
            Event::Eof => {
                if !closed
                    || !stack.is_empty()
                    || row.is_some()
                    || capture.is_some()
                    || fields.get("EncodingType").map(String::as_str) != Some("url")
                {
                    return Err(invalid());
                }
                return Ok(Page { fields, rows });
            }
            _ => return Err(invalid()),
        }
    }
}
fn xml_character(character: char) -> bool {
    matches!(character, '\t' | '\n' | '\r' | ' '..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..='\u{10FFFF}')
}
fn append(target: &mut String, value: &str) -> Result<()> {
    if target
        .len()
        .checked_add(value.len())
        .is_none_or(|n| n > 4096)
    {
        return Err(Error::Limit("S3 XML scalar bound".into()));
    }
    target.push_str(value);
    Ok(())
}
pub(super) fn boolean(value: &str) -> Result<bool> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(invalid()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_xml_decodes_only_builtin_references_and_rejects_ambiguous_documents() {
        let raw = b"<ListBucketResult><Name>bucket</Name><Prefix>a%2F</Prefix><EncodingType>url</EncodingType><IsTruncated>false</IsTruncated><Contents><Key>a%2Fx%26y</Key><Size>0</Size><ETag>&quot;e&quot;</ETag></Contents></ListBucketResult>";
        let page = parse(raw, false, 1, &AtomicBool::new(false)).unwrap();
        assert_eq!(page.rows[0].key, "a/x&y");
        assert_eq!(page.rows[0].etag.as_deref(), Some("\"e\""));
        for raw in [b"<!DOCTYPE x [<!ENTITY a 'boom'>]><ListBucketResult/>".as_slice(), b"<ListBucketResult><EncodingType>url</EncodingType><EncodingType>url</EncodingType></ListBucketResult>", b"<ListBucketResult><EncodingType>url</EncodingType><Prefix>&untrusted;</Prefix></ListBucketResult>", b"<ListBucketResult><EncodingType>url</EncodingType>"] { assert!(parse(raw, false, 10, &AtomicBool::new(false)).is_err()); }
        assert!(parse(raw, false, 0, &AtomicBool::new(false)).is_err());
    }
    #[test]
    fn entity_text_outside_one_root_and_repeated_xml_declarations_refuse() {
        let root = "<ListBucketResult><EncodingType>url</EncodingType><IsTruncated>false</IsTruncated></ListBucketResult>";
        for raw in [
            format!("&amp;{root}"),
            format!("{root}&amp;"),
            format!("<?xml version=\"1.0\"?><?xml version=\"1.0\"?>{root}"),
            format!("{root}<?xml version=\"1.0\"?>"),
        ] {
            assert!(parse(raw.as_bytes(), false, 1, &AtomicBool::new(false)).is_err());
        }
        assert!(
            parse(
                format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>{root}").as_bytes(),
                false,
                1,
                &AtomicBool::new(false)
            )
            .is_ok()
        );
    }
    #[test]
    fn xml10_forbidden_raw_and_referenced_characters_refuse_even_in_ignored_metadata() {
        for text in ["\u{1}", "\u{FFFE}", "&#1;", "&#xFFFF;"] {
            let raw = format!(
                "<ListBucketResult><EncodingType>url</EncodingType><Owner><DisplayName>{text}</DisplayName></Owner></ListBucketResult>"
            );
            assert!(parse(raw.as_bytes(), false, 1, &AtomicBool::new(false)).is_err());
        }
        let raw=b"<ListBucketResult><EncodingType>url</EncodingType><Owner><DisplayName>&#9;</DisplayName></Owner></ListBucketResult>";
        assert!(parse(raw, false, 1, &AtomicBool::new(false)).is_ok());
    }
}
