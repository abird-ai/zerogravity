//! CLI IO for the explicitly selected finite diagnostic SDK.
use super::*;
pub(super) use zerogravity::diagnostics::qwen_route::{require_route_capability, selected};
pub(super) fn run_route(request: Request, inventory: devices::Inventory) -> Result<()> {
    use zerogravity::diagnostics::qwen_route::{self, Report};
    qwen_route::run_route(
        request,
        inventory,
        || {
            use std::io::Read;
            let mut text = String::new();
            std::io::stdin()
                .take(1024 * 1024 + 1)
                .read_to_string(&mut text)?;
            if text.len() > 1024 * 1024 {
                return Err(Error::Limit("prompt stdin1MiB".into()));
            }
            Ok(text)
        },
        |event| {
            match event {
                Report::Decision {
                    value,
                    pretty: true,
                } => println!(
                    "{}",
                    serde_json::to_string_pretty(value)
                        .map_err(|error| Error::State(error.to_string()))?
                ),
                Report::Decision {
                    value,
                    pretty: false,
                } => eprintln!(
                    "{}",
                    serde_json::to_string(value)
                        .map_err(|error| Error::State(error.to_string()))?
                ),
                Report::Text(text) => println!("{text}"),
            }
            Ok(())
        },
    )
}
