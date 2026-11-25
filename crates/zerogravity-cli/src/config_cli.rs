//! Cold configuration/profile presentation over the shared document editor.
use crate::args::{Format, Topic};
use std::{
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
    process::Command,
};
use zerogravity::{
    Error, Result,
    configuration::{self as cfg, Change, ConfigFile, INITIAL},
    run::config::{DocumentSelection, select_document},
};

fn bad(message: &'static str) -> Error {
    Error::Malformed(message.into())
}

fn path(selection: DocumentSelection<'_>) -> Result<PathBuf> {
    select_document(
        selection,
        &std::env::current_dir()?,
        std::env::var_os("XDG_CONFIG_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?
    .ok_or_else(|| bad("configuration selection required"))
}

pub(super) fn selected_path(document: Option<(&str, &str)>) -> Result<PathBuf> {
    let selection = match document {
        None => DocumentSelection::Default,
        Some(("--config", file)) => DocumentSelection::File(Path::new(file)),
        Some(("--profile", name)) => DocumentSelection::Profile(name),
        _ => return Err(bad("invalid document selector")),
    };
    path(selection)
}
fn selected(document: Option<(&str, &str)>) -> Result<ConfigFile> {
    ConfigFile::open(selected_path(document)?)
}

fn named(name: &str) -> Result<ConfigFile> {
    ConfigFile::open(path(DocumentSelection::Profile(name))?)
}

pub(super) fn help(topic: Topic, _: Option<&str>) {
    match topic {
        Topic::Config => println!(
            "USAGE: zerogravity [--config FILE | --profile NAME] config [COMMAND]\n\
            show | path | get KEY | set KEY YAML | unset KEY | validate [FILE] | edit | explain | init\n\
            Keys follow the existing schema, e.g. ui.color, run.backend, run.task.context, serve.listen.\n\
            Values are one quoted YAML scalar, sequence or mapping. set/unset normalize YAML formatting.\n\
            edit invokes VISUAL or EDITOR (executable, no shell), validates a temporary draft, then replaces.\n\
            --format json for read commands (place global flags before operands). No model/device/network IO."
        ),
        Topic::Profile => println!(
            "USAGE: zerogravity profile [list | show NAME | create NAME | copy FROM TO | edit NAME | remove NAME --yes]\n\
            default is config.yaml; named profiles use profiles/NAME.yaml with identical schema.\n\
            --profile NAME selects one document, not inheritance. Create/copy never overwrite.\n\
            Use config --profile NAME to inspect or edit a named profile."
        ),
        _ => unreachable!(),
    }
}

/// Non-execution product commands consume only the selected UI preferences.
pub(super) fn apply_presentation(document: Option<(&str, &str)>) -> Result<()> {
    let selection = match document {
        None => DocumentSelection::Default,
        Some(("--config", file)) => DocumentSelection::File(Path::new(file)),
        Some(("--profile", name)) => DocumentSelection::Profile(name),
        _ => return Err(bad("invalid document selector")),
    };
    let selected = cfg::load_selected(
        selection,
        &std::env::current_dir()?,
        std::env::var_os("XDG_CONFIG_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?;
    crate::output::install_document(selected.and_then(|(_, doc)| doc.ui).unwrap_or_default());
    Ok(())
}

fn preferences(file: &ConfigFile) {
    if let Ok(doc) = zerogravity::pipelines::parse_document(file.bytes()) {
        crate::output::install_document(doc.ui.unwrap_or_default());
    }
}

pub(super) fn config(
    args: &[String],
    document: Option<(&'static str, String)>,
    format: Format,
) -> Result<()> {
    let selection = document
        .as_ref()
        .map(|(flag, value)| (*flag, value.as_str()));
    if args.is_empty() {
        if let Ok(file) = selected(selection) {
            preferences(&file);
        }
        if format != Format::Json && crate::output::reporter().run_interaction()? {
            return manager(
                "Config",
                "show · get KEY · set KEY YAML · unset KEY · explain · edit · validate · init · exit",
                |args| config_action(args, selection, Format::Text),
            );
        }
        return config_action(&["show".into()], selection, format);
    }
    config_action(args, selection, format)
}

fn config_action(args: &[String], selection: Option<(&str, &str)>, format: Format) -> Result<()> {
    let words = args.iter().map(String::as_str).collect::<Vec<_>>();
    if let ["validate", file] = words.as_slice() {
        if selection.is_some() {
            return Err(bad("validate FILE conflicts with --config/--profile"));
        }
        let doc = selected(Some(("--config", file)))?;
        doc.require_existing()?;
        return validate(&doc, format);
    }
    if words == ["path"] {
        let path = selected_path(selection)?;
        return if format == Format::Json {
            crate::output::json(&serde_json::json!({"path":path}))
        } else {
            println!("{}", path.display());
            Ok(())
        };
    }
    let file = selected(selection)?;
    preferences(&file);
    let implicit = selection.is_none();
    let default_selected = implicit || matches!(selection, Some(("--profile", "default")));
    // An explicitly selected missing named/portable document is not default.
    // Only config init explicitly creates arbitrary documents.
    if !default_selected && !matches!(words.as_slice(), ["init"]) {
        file.require_existing()?;
    }
    match words.as_slice() {
        ["show"] => show(&file, format),
        ["get", key] => {
            let setting = cfg::get(file.path(), file.bytes(), key)?;
            if format == Format::Json {
                crate::output::json(&setting)
            } else {
                print!("{}", yaml(&setting.value)?);
                Ok(())
            }
        }
        ["explain"] => explain(&file, format),
        ["validate"] => validate(&file, format),
        ["set", key, value] => {
            text_only(format)?;
            file.change(Change::Set {
                path: key,
                yaml: value,
            })?;
            saved(&file);
            Ok(())
        }
        ["unset", key] => {
            text_only(format)?;
            file.change(Change::Unset { path: key })?;
            saved(&file);
            Ok(())
        }
        ["init"] => {
            text_only(format)?;
            file.create(INITIAL.as_bytes())?;
            saved(&file);
            Ok(())
        }
        ["edit"] => {
            text_only(format)?;
            edit(&file)
        }
        _ => Err(bad("invalid config command; see config --help")),
    }
}

pub(super) fn profile(args: &[String], format: Format, yes: bool) -> Result<()> {
    if let Ok(file) = selected(None) {
        preferences(&file);
    }
    if args.is_empty() {
        if format != Format::Json && crate::output::reporter().run_interaction()? {
            list(format)?;
            return manager(
                "Profiles",
                "list · show NAME · create NAME · copy FROM TO · edit NAME · remove NAME · exit",
                |args| profile_action(args, Format::Text, false),
            );
        }
        return list(format);
    }
    profile_action(args, format, yes)
}

fn profile_action(args: &[String], format: Format, yes: bool) -> Result<()> {
    let words = args.iter().map(String::as_str).collect::<Vec<_>>();
    match words.as_slice() {
        ["list"] => list(format),
        ["show", name] => {
            let file = named(name)?;
            if *name != "default" {
                file.require_existing()?;
            }
            show(&file, format)
        }
        ["create", name] => {
            text_only(format)?;
            let file = new_profile(name)?;
            file.create(INITIAL.as_bytes())?;
            saved(&file);
            Ok(())
        }
        ["copy", from, to] => {
            text_only(format)?;
            let original = named(from)?;
            if *from != "default" {
                original.require_existing()?;
            }
            let file = new_profile(to)?;
            let bytes = cfg::copy_document(original.path(), file.path(), original.bytes())?;
            file.create(&bytes)?;
            saved(&file);
            Ok(())
        }
        ["edit", name] => {
            text_only(format)?;
            let file = named(name)?;
            if *name != "default" {
                file.require_existing()?;
            }
            edit(&file)
        }
        ["remove", name] => {
            text_only(format)?;
            if *name == "default" {
                return Err(bad("default is config.yaml; edit or unset its settings"));
            }
            let file = named(name)?;
            file.require_existing()?;
            if !yes {
                if !crate::output::reporter().run_interaction()? {
                    return Err(bad(
                        "profile remove requires --yes without an interactive terminal",
                    ));
                }
                eprint!("Remove profile {name}? [y/N] ");
                io::stderr().flush()?;
                if !line()?.is_some_and(|line| matches!(line.trim(), "y" | "yes" | "Y" | "YES")) {
                    return Err(bad("profile removal declined"));
                }
            }
            file.remove()?;
            crate::output::emit!(Status, "Removed profile {name}");
            Ok(())
        }
        _ => Err(bad("invalid profile command; see profile --help")),
    }
}

fn new_profile(name: &str) -> Result<ConfigFile> {
    if name == "default" {
        return Err(bad("default is config.yaml; use config init"));
    }
    named(name)
}

fn list(format: Format) -> Result<()> {
    let default = path(DocumentSelection::Default)?;
    let root = default.parent().expect("config root");
    let names = cfg::profiles(root)?;
    let mut rows = Vec::new();
    for name in names {
        let path = if name == "default" {
            default.clone()
        } else {
            root.join("profiles").join(format!("{name}.yaml"))
        };
        let exists = path.is_file();
        rows.push(serde_json::json!({"name":name,"path":path,"exists":exists}));
    }
    if format == Format::Json {
        crate::output::json(&rows)
    } else {
        for row in rows {
            println!(
                "{:20} {}{}",
                row["name"].as_str().unwrap(),
                row["path"].as_str().unwrap_or(""),
                if row["exists"] == false {
                    " (built-in defaults)"
                } else {
                    ""
                }
            );
        }
        Ok(())
    }
}
fn show(file: &ConfigFile, format: Format) -> Result<()> {
    if format == Format::Json {
        let document = cfg::document_value(file.path(), file.bytes())?;
        let effective = cfg::explain(file.path(), file.bytes())?;
        return crate::output::json(&serde_json::json!({
            "path": file.path(), "exists": file.exists(), "document": document, "effective": effective
        }));
    }
    if !file.exists() {
        println!(
            "# {} does not exist; effective built-in defaults",
            file.path().display()
        );
        return explain(file, format);
    }
    // Raw view is useful even when the document needs repair.
    io::stdout().lock().write_all(file.bytes())?;
    Ok(())
}
fn explain(file: &ConfigFile, format: Format) -> Result<()> {
    let values = cfg::explain(file.path(), file.bytes())?;
    if format == Format::Json {
        return crate::output::json(&values);
    }
    for (key, setting) in values {
        let value = serde_json::to_string(&setting.value)
            .map_err(|_| bad("configuration value encoding"))?;
        println!("{key:32} {value:24} {}", setting.origin);
    }
    Ok(())
}
fn validate(file: &ConfigFile, format: Format) -> Result<()> {
    cfg::validate(file.path(), file.bytes())?;
    if format == Format::Json {
        crate::output::json(
            &serde_json::json!({"valid":true,"path":file.path(),"exists":file.exists(),"execution_qualified":false}),
        )
    } else {
        println!(
            "Valid configuration (cold policy only): {}",
            file.path().display()
        );
        Ok(())
    }
}
fn yaml(value: &serde_yaml::Value) -> Result<String> {
    serde_yaml::to_string(value).map_err(|_| bad("configuration value encoding"))
}
fn text_only(format: Format) -> Result<()> {
    if format == Format::Json {
        Err(bad("configuration mutations use text mode"))
    } else {
        Ok(())
    }
}
fn saved(file: &ConfigFile) {
    crate::output::emit!(Status, "Saved {}", file.path().display());
}
fn edit(file: &ConfigFile) -> Result<()> {
    let editor = std::env::var_os("VISUAL")
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var_os("EDITOR").filter(|v| !v.is_empty()))
        .unwrap_or_else(|| "vi".into());
    file.edit(|draft| {
        let status = Command::new(editor).arg(draft).status()?;
        if status.success() {
            Ok(())
        } else {
            Err(Error::State(
                "editor exited unsuccessfully; configuration unchanged".into(),
            ))
        }
    })?;
    saved(file);
    Ok(())
}

fn line() -> Result<Option<String>> {
    let mut line = String::new();
    let count = io::stdin()
        .lock()
        .take(zerogravity::pipelines::MAX_CONFIG_BYTES as u64 + 1)
        .read_line(&mut line)?;
    if count == 0 {
        return Ok(None);
    }
    if count > zerogravity::pipelines::MAX_CONFIG_BYTES {
        return Err(Error::Limit("configuration input extent".into()));
    }
    Ok(Some(line.trim().into()))
}
fn manager(name: &str, help: &str, mut action: impl FnMut(&[String]) -> Result<()>) -> Result<()> {
    eprintln!("{name}\n{help}");
    loop {
        eprint!("{name} › ");
        io::stderr().flush()?;
        let Some(line) = line()? else {
            return Ok(());
        };
        if matches!(line.as_str(), "exit" | "quit" | "/exit") {
            return Ok(());
        }
        if line.is_empty() {
            continue;
        }
        if line == "help" {
            eprintln!("{help}");
            continue;
        }
        // Values following set KEY are one YAML expression, not shell argv.
        let args: Vec<String> = if line.starts_with("set ") {
            line.splitn(3, ' ').map(str::to_owned).collect()
        } else {
            line.split_whitespace().map(str::to_owned).collect()
        };
        if let Err(error) = action(&args) {
            crate::output::fail(crate::output::Failure::core("config/profile", &error));
        }
    }
}
