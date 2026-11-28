//! Cold UI choices over the existing run-policy override and device observation.
use zerogravity::{devices::Observation, run::config::RunOverrides, run_policy::Backend};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Execution {
    pub backend: Option<Backend>,
    pub device: Option<String>,
}

impl Execution {
    pub fn on_device(device: String) -> Self {
        Self {
            backend: None,
            device: Some(device),
        }
    }

    pub fn overrides(self) -> RunOverrides {
        let devices = match (self.backend.as_ref(), self.device) {
            (_, Some(device)) => Some(vec![device]),
            (Some(_), None) => Some(vec!["auto".into()]),
            (None, None) => None,
        };
        RunOverrides {
            backend: self.backend,
            devices,
            ..Default::default()
        }
    }

    pub fn summary(&self, profile: &str) -> String {
        let Some(backend) = &self.backend else {
            return self
                .device
                .as_ref()
                .map_or_else(|| profile.into(), |id| format!("Profile · {id}"));
        };
        let backend = name(backend);
        self.device
            .as_ref()
            .map_or_else(|| backend.into(), |id| format!("{backend} · {id}"))
    }
}

pub(crate) fn name(backend: &Backend) -> &'static str {
    match backend {
        Backend::Auto => "Auto",
        Backend::Cpu => "CPU",
        Backend::Cuda => "CUDA",
        Backend::Rocm => "ROCm",
        Backend::Vulkan => "Vulkan",
    }
}

pub(crate) fn profile_summary(run: Option<&zerogravity::pipelines::RunDoc>) -> String {
    let Some(run) = run else { return "Auto".into() };
    let Ok(backend) = Backend::parse(run.backend.as_deref().unwrap_or("auto")) else {
        return "Profile · check configuration".into();
    };
    let pinned = run
        .device
        .as_ref()
        .is_some_and(|devices| devices.as_slice() != ["auto"]);
    if backend == Backend::Auto && !pinned {
        "Auto".into()
    } else if pinned {
        format!("Profile · {} · pinned devices", name(&backend))
    } else {
        format!("Profile · {}", name(&backend))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Choice {
    pub label: String,
    pub execution: Execution,
    pub refusal: Option<&'static str>,
}

/// Discovery supplies hints, never source-qualified execution guarantees.
pub(crate) fn choices(devices: &[Observation]) -> Vec<Choice> {
    let mut result = vec![Choice {
        label: "Automatic".into(),
        execution: Execution::default(),
        refusal: None,
    }];
    let mut push = |backend: Backend, device: Option<&Observation>, label: String, refusal| {
        result.push(Choice {
            label,
            execution: Execution {
                backend: Some(backend),
                device: device.map(|row| row.id.clone()),
            },
            refusal,
        });
    };
    let backends = [
        (Backend::Auto, true),
        (Backend::Cpu, cfg!(feature = "cpu")),
        (Backend::Cuda, cfg!(feature = "cuda")),
        (Backend::Rocm, cfg!(feature = "hip")),
        (Backend::Vulkan, cfg!(feature = "vulkan")),
    ];
    for (backend, compiled) in &backends {
        if *compiled || *backend == Backend::Cpu {
            let label = match backend {
                Backend::Auto => "Automatic".into(),
                Backend::Cpu => "CPU".into(),
                _ => format!("{} · any device", name(backend)),
            };
            push(
                backend.clone(),
                None,
                label,
                (!*compiled).then_some("Not compiled"),
            );
        }
    }
    for row in devices.iter().filter(|row| row.class == "gpu") {
        // Exact local physical identities only. Do not expose ordinal selectors.
        if !row.id.strip_prefix("pci:").is_some_and(|id| {
            zerogravity::devices::normalize_pci(id).is_ok_and(|canonical| canonical == id)
        }) {
            continue;
        }
        let vendor = match row.vendor.as_str() {
            "nvidia" => "NVIDIA",
            "amd" => "AMD",
            _ => "GPU",
        };
        let observed = |backend: &Backend| {
            row.backends
                .iter()
                .any(|hint| hint.split(':').next() == Some(backend.label()))
        };
        for (backend, compiled) in &backends {
            if *compiled && !matches!(backend, Backend::Auto | Backend::Cpu) && observed(backend) {
                push(
                    backend.clone(),
                    Some(row),
                    format!("{} · {vendor} · {}", name(backend), row.id),
                    None,
                );
            }
        }
    }
    result
}

/// Inheritance and explicit Auto are identical for the unconstrained default
/// profile. Keep both only when the profile actually changes execution policy.
pub(crate) fn for_profile(mut choices: Vec<Choice>, profile: &str) -> Vec<Choice> {
    if profile == "Auto" {
        choices.retain(|choice| {
            choice.execution
                != Execution {
                    backend: Some(Backend::Auto),
                    device: None,
                }
        });
    } else if let Some(first) = choices.first_mut() {
        first.label = profile.to_owned();
    }
    choices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_choice_is_unique_and_device_choices_are_exact() {
        let device = Observation {
            id: "pci:0000:01:00.0".into(),
            class: "gpu".into(),
            vendor: "nvidia".into(),
            ceiling: None,
            available: None,
            backends: vec![
                "cuda:binding-unverified".into(),
                "vulkan:binding-unverified".into(),
            ],
            exclusion: None,
            provenance: String::new(),
            cuda_ordinal: None,
            cuda_cc: None,
        };
        let rows = for_profile(choices(std::slice::from_ref(&device)), "Auto");
        assert_eq!(rows[0].label, "Automatic");
        assert_eq!(rows[0].execution, Execution::default());
        assert!(
            rows.iter()
                .all(|row| row.execution.backend != Some(Backend::Auto))
        );
        assert!(
            rows.iter()
                .any(|row| row.execution.backend == Some(Backend::Cpu))
        );
        let explicit = for_profile(choices(&[device]), "Profile · CPU");
        assert_eq!(explicit[0].label, "Profile · CPU");
        assert_eq!(
            explicit
                .iter()
                .filter(|row| row.execution.backend == Some(Backend::Auto))
                .count(),
            1
        );
        assert!(
            explicit
                .iter()
                .filter(|row| row.execution.device.is_some())
                .all(|row| matches!(row.execution.backend, Some(Backend::Cuda | Backend::Vulkan)))
        );
    }

    #[test]
    fn profile_inheritance_and_explicit_pair_overrides() {
        let inherited = Execution::default().overrides();
        assert!(inherited.backend.is_none() && inherited.devices.is_none());
        let automatic = Execution {
            backend: Some(Backend::Auto),
            device: None,
        }
        .overrides();
        assert_eq!(automatic.backend, Some(Backend::Auto));
        assert_eq!(automatic.devices, Some(vec!["auto".into()]));
        let device = Execution::on_device("cpu:0".into()).overrides();
        assert!(device.backend.is_none());
        assert_eq!(device.devices, Some(vec!["cpu:0".into()]));
        let forced = Execution {
            backend: Some(Backend::Cuda),
            device: Some("pci:0000:01:00.0".into()),
        }
        .overrides();
        assert_eq!(forced.backend, Some(Backend::Cuda));
        assert_eq!(forced.devices, Some(vec!["pci:0000:01:00.0".into()]));
    }
}
