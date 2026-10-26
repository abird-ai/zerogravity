use super::*;
use crate::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
#[test]
fn eight_owner_loader_checks_actual_host_arbiter_not_only_cli_numbers() {
    for bytes in [3072 * MIB, 4096 * MIB - 1, 4096 * MIB, 4096 * MIB + 1] {
        let arbiter = ResourceArbiter::new();
        let resource =
            PhysicalDeviceKey::new("host", &format!("canonical-active8-{bytes}")).unwrap();
        arbiter.register(resource.clone(), bytes as u64).unwrap();
        let host = TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("fixed-prompt", "cold-host-fixture").unwrap(),
        };
        assert_eq!(
            require_host_profile(&host, FixedPromptLifecycleProfile::BenchmarkEight).is_ok(),
            bytes == 4096 * MIB
        );
        require_host_profile(&host, FixedPromptLifecycleProfile::FiniteTwo).unwrap();
        assert_eq!(host.arbiter.snapshot(&host.resource).unwrap().reserved, 0);
    }
}
