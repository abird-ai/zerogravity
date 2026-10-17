//! Provider-specific bounded listing grammar over the existing XML tokenizer.
//! No document type, declared/external entity or endpoint following is allowed.
use crate::{Error, Result};
use quick_xml::{Reader, events::Event};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;

pub(super) struct Page {
    pub origin: String,
    pub container: String,
    pub prefix: String,
    pub marker: String,
    pub next: String,
    pub rows: Vec<Row>,
}
pub(super) struct Row {
    pub name: String,
    pub version: String,
    pub etag: String,
    pub bytes: u64,
}
fn invalid() -> Error {
    Error::Malformed("invalid bounded Azure listing XML".into())
}
fn insert(map: &mut BTreeMap<String, String>, key: String, value: String) -> Result<()> {
    if map.insert(key, value).is_some() {
        return Err(invalid());
    }
    Ok(())
}
fn append(value: &mut String, text: &str) -> Result<()> {
    if value
        .len()
        .checked_add(text.len())
        .is_none_or(|size| size > 8192)
    {
        return Err(Error::Limit("Azure XML scalar bound".into()));
    }
    value.push_str(text);
    Ok(())
}
pub(super) fn parse(raw: &[u8], maximum_rows: usize, cancel: &AtomicBool) -> Result<Page> {
    let input = std::str::from_utf8(raw).map_err(|_| invalid())?;
    if !input.chars().all(xml_character) {
        return Err(invalid());
    }
    let mut reader = Reader::from_str(input);
    reader.config_mut().expand_empty_elements = true;
    reader.config_mut().check_end_names = true;
    let mut stack = Vec::<String>::new();
    let mut fields = BTreeMap::new();
    let mut attributes = BTreeMap::new();
    let mut root_children = BTreeSet::new();
    let mut row = None::<BTreeMap<String, String>>;
    let mut capture = None::<(usize, String, String)>;
    let mut rows = Vec::new();
    let mut closed = false;
    let mut started = false;
    let mut events = 0usize;
    loop {
        super::check_cancel(cancel)?;
        events += 1;
        if events > 131072 {
            return Err(Error::Limit("Azure XML event bound".into()));
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
                    if started || name != "EnumerationResults" {
                        return Err(invalid());
                    }
                    started = true;
                    for attribute in event.attributes() {
                        let attribute = attribute.map_err(|_| invalid())?;
                        let key =
                            std::str::from_utf8(attribute.key.as_ref()).map_err(|_| invalid())?;
                        if !matches!(key, "ServiceEndpoint" | "ContainerName")
                            || attribute.value.len() > 2048
                        {
                            return Err(invalid());
                        }
                        let value = attribute
                            .decoded_and_normalized_value(
                                quick_xml::XmlVersion::Implicit1_0,
                                reader.decoder(),
                            )
                            .map_err(|_| invalid())?;
                        insert(&mut attributes, key.to_owned(), value.into_owned())?;
                    }
                } else if event.attributes().next().is_some() {
                    if name == "Name" {
                        return Err(Error::Unsupported("Azure XML encoded-name profile".into()));
                    }
                    return Err(invalid());
                }
                if stack.len() == 1 && !root_children.insert(name.clone()) {
                    return Err(invalid());
                }
                if name == "BlobPrefix" {
                    return Err(Error::Unsupported(
                        "Azure listing directories are not selected data".into(),
                    ));
                }
                if name == "Blob" {
                    if stack.as_slice() != ["EnumerationResults", "Blobs"] || row.is_some() {
                        return Err(invalid());
                    }
                    row = Some(BTreeMap::new());
                }
                stack.push(name.clone());
                if stack.len() == 4 && name == "Properties" && row.is_some() {
                    insert(
                        row.as_mut().ok_or_else(invalid)?,
                        "properties".into(),
                        String::new(),
                    )?;
                }
                let root_scalar = stack.len() == 2
                    && matches!(
                        name.as_str(),
                        "Prefix" | "Marker" | "NextMarker" | "MaxResults"
                    );
                let blob_scalar = stack.len() == 4
                    && row.is_some()
                    && matches!(
                        name.as_str(),
                        "Name" | "VersionId" | "Snapshot" | "Deleted" | "IsCurrentVersion"
                    );
                let property = stack.len() == 5
                    && stack[3] == "Properties"
                    && row.is_some()
                    && matches!(
                        name.as_str(),
                        "Etag"
                            | "Content-Length"
                            | "Content-Encoding"
                            | "BlobType"
                            | "CopyStatus"
                            | "CustomerProvidedKeySha256"
                    );
                if root_scalar || blob_scalar || property {
                    let key = if property {
                        format!("Properties:{name}")
                    } else {
                        name
                    };
                    capture = Some((stack.len(), key, String::new()));
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
                    let (_, key, value) = capture.take().ok_or_else(invalid)?;
                    if stack.len() == 2 {
                        insert(&mut fields, key, value)?;
                    } else {
                        insert(row.as_mut().ok_or_else(invalid)?, key, value)?;
                    }
                }
                if stack.len() == 3 && name == "Blob" {
                    if rows.len() == maximum_rows || rows.len() == 4096 {
                        return Err(Error::Limit("Azure listing row bound".into()));
                    }
                    rows.push(finish(row.take().ok_or_else(invalid)?)?);
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
                if stack.is_empty() {
                    return Err(invalid());
                }
                let character =
                    if let Some(value) = reference.resolve_char_ref().map_err(|_| invalid())? {
                        value
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
                if events != 1
                    || started
                    || closed
                    || event.version().map_err(|_| invalid())?.as_ref() != b"1.0"
                    || event.encoding().is_some_and(|encoding| {
                        encoding.map_or(true, |encoding| !encoding.eq_ignore_ascii_case(b"UTF-8"))
                    })
                {
                    return Err(invalid());
                }
            }
            Event::Comment(_) => {}
            Event::DocType(_) | Event::PI(_) => return Err(invalid()),
            Event::Eof => break,
            Event::Empty(_) => return Err(invalid()),
        }
    }
    if !closed
        || !stack.is_empty()
        || row.is_some()
        || capture.is_some()
        || !root_children.contains("Blobs")
    {
        return Err(invalid());
    }
    Ok(Page {
        origin: attributes.remove("ServiceEndpoint").ok_or_else(invalid)?,
        container: attributes.remove("ContainerName").ok_or_else(invalid)?,
        prefix: fields.remove("Prefix").ok_or_else(invalid)?,
        marker: fields.remove("Marker").unwrap_or_default(),
        next: fields.remove("NextMarker").ok_or_else(invalid)?,
        rows,
    })
}
fn xml_character(value: char) -> bool {
    matches!(value, '\t' | '\n' | '\r' | '\u{20}'..='\u{d7ff}' | '\u{e000}'..='\u{fffd}' | '\u{10000}'..='\u{10ffff}')
}
fn finish(mut values: BTreeMap<String, String>) -> Result<Row> {
    if values.contains_key("Snapshot")
        || values.get("Deleted").is_some_and(|value| value != "false")
        || values
            .get("IsCurrentVersion")
            .is_some_and(|value| value != "true")
        || values
            .get("Properties:BlobType")
            .is_none_or(|value| value != "BlockBlob")
        || values
            .get("Properties:Content-Encoding")
            .is_some_and(|value| !value.is_empty() && !value.eq_ignore_ascii_case("identity"))
        || values.contains_key("Properties:CustomerProvidedKeySha256")
        || values
            .get("Properties:CopyStatus")
            .is_some_and(|value| value != "success")
    {
        return Err(Error::Unsupported(
            "Azure listing requires complete identity-encoded BlockBlob versions".into(),
        ));
    }
    let version = values
        .remove("VersionId")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            Error::Unsupported("Azure prefix requires versioned current blobs".into())
        })?;
    Ok(Row {
        name: values.remove("Name").ok_or_else(invalid)?,
        version,
        etag: values.remove("Properties:Etag").ok_or_else(invalid)?,
        bytes: super::decimal(
            &values
                .remove("Properties:Content-Length")
                .ok_or_else(invalid)?,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(name: &str) -> String {
        format!(
            "<Blob><Name>{name}</Name><VersionId>2023-10-01T01:02:03.0000000Z</VersionId><Properties><Etag>&quot;opaque&quot;</Etag><Content-Length>3</Content-Length><Content-Encoding/><BlobType>BlockBlob</BlobType></Properties></Blob>"
        )
    }
    fn document(rows: &str) -> String {
        format!(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?><EnumerationResults ServiceEndpoint=\"https://fixture.blob.core.windows.net/\" ContainerName=\"container\"><Prefix>folder/</Prefix><Marker/><Blobs>{rows}</Blobs><NextMarker/></EnumerationResults>"
        )
    }
    fn checked(raw: &str) -> Result<Page> {
        parse(raw.as_bytes(), 4096, &AtomicBool::new(false))
    }
    #[test]
    fn complete_document_decodes_only_bounded_xml_text() {
        let page = checked(&document(&row("folder/a&amp;b-&#955;.bin"))).unwrap();
        assert_eq!(page.rows.len(), 1);
        assert_eq!(page.rows[0].name, "folder/a&b-λ.bin");
        assert_eq!(page.rows[0].etag, "\"opaque\"");
        assert_eq!(page.rows[0].bytes, 3);
        assert!(page.next.is_empty());
    }
    #[test]
    fn document_framing_declared_entities_and_invalid_character_refs_refuse() {
        let base = document(&row("folder/a.bin"));
        for raw in [
            format!("{base}&amp;"),
            format!("&#32;{base}"),
            format!("{base}<extra/>"),
            format!("<?xml version=\"1.0\"?>{base}"),
            base.replace("<Blobs>", "<!DOCTYPE x [<!ENTITY boom 'expanded'>]><Blobs>"),
            base.replace("folder/a.bin", "folder/&boom;.bin"),
            base.replace("folder/a.bin", "folder/&#0;.bin"),
            base.replace("folder/a.bin", "folder/&#1;.bin"),
            base.replace("folder/a.bin", "folder/\u{1}.bin"),
            base.replace("encoding=\"utf-8\"", "encoding=\"utf-16\""),
            base[..base.len() - 1].to_owned(),
        ] {
            assert!(checked(&raw).is_err());
        }
    }
    #[test]
    fn mandatory_fields_duplicates_and_encoded_name_profile_refuse() {
        let row = row("folder/a.bin");
        for changed in [
            row.replace("<VersionId>", "<Unexpected>")
                .replace("</VersionId>", "</Unexpected>"),
            row.replace("</Name>", "</Name><Name>duplicate</Name>"),
            row.replace("</Properties>", "</Properties><Properties/>"),
            row.replace("<Name>", "<Name Encoded=\"true\">"),
            row.replace(
                "<Content-Length>3</Content-Length>",
                "<Content-Length>3.0</Content-Length>",
            ),
            row.replace(
                "<Content-Length>3</Content-Length>",
                "<Content-Length>18446744073709551616</Content-Length>",
            ),
            row.replace("<Name>", "<Name><nested>")
                .replace("</Name>", "</nested></Name>"),
        ] {
            assert!(checked(&document(&changed)).is_err());
        }
    }
    #[test]
    fn incomplete_copy_nonblock_encoded_and_encrypted_profiles_refuse() {
        let row = row("folder/a.bin");
        for changed in [row.replace("BlockBlob","PageBlob"), row.replace("BlockBlob","AppendBlob"),row.replace("<Content-Encoding/>","<Content-Encoding>gzip</Content-Encoding>"),row.replace("</Properties>","<CopyStatus>pending</CopyStatus></Properties>"),row.replace("</Properties>","<CopyStatus>failed</CopyStatus></Properties>"),row.replace("</Properties>","<CustomerProvidedKeySha256>not-content-sha</CustomerProvidedKeySha256></Properties>"),row.replace("</Blob>","<Deleted>true</Deleted></Blob>"),row.replace("</Blob>","<IsCurrentVersion>false</IsCurrentVersion></Blob>"),row.replace("</Blob>","<Snapshot>opaque</Snapshot></Blob>")] { assert!(checked(&document(&changed)).is_err()); }
    }
    #[test]
    fn row_scalar_depth_and_cancel_bounds_are_actual() {
        let rows = format!("{}{}", row("folder/a"), row("folder/b"));
        assert!(parse(document(&rows).as_bytes(), 1, &AtomicBool::new(false)).is_err());
        assert!(checked(&document(&row(&"x".repeat(8193)))).is_err());
        assert!(
            checked(&document(&format!(
                "{}{}{}",
                "<nested>".repeat(17),
                row("folder/a"),
                "</nested>".repeat(17)
            )))
            .is_err()
        );
        assert!(
            parse(
                document(&row("folder/a")).as_bytes(),
                1,
                &AtomicBool::new(true)
            )
            .is_err()
        );
    }
}
