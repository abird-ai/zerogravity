//! Same iterator through genuine layer9; completed diagnostics never publish.
use super::*;
pub(super) fn run(directory: &Path, report: &Path) -> Proof<()> {
    let start = std::time::Instant::now();
    let accepted = reference::initial_core::accepted()?;
    let old = reference::layer9::accepted()?;
    let derived = budget::derived_upper_with_layer9(true)? + budget::MIB;
    need(derived <= budget::HOST, "inner tape within unchanged768MiB")?;
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "original27b-initial-core")?;
    arbiter.register(resource.clone(), budget::HOST)?;
    let host = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("original27b", "initial-core-prefix")?,
    };
    let control = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 32 * budget::MIB,
        }],
    )?;
    let (operators, held) = source::open_source(directory, &host)?;
    let operators = Arc::new(operators);
    let source = operators.source().clone();
    let source_identity = source.source_identity();
    let mut session = operators.diagnostic_full_convolution_session(1, None)?;
    let base = session.binding();
    let baseline = host.arbiter.snapshot(&host.resource)?.reserved;
    let precancel = matches!(session.observe_initial_core_with(11,Some(&AtomicBool::new(true)),|_|Err(Error::State("pre-cancel reached observer".into()))),Err(Error::State(ref w)) if w.contains("cancelled"));
    need(
        precancel
            && session.binding() == base
            && host.arbiter.snapshot(&host.resource)?.reserved == baseline,
        "true initial-prefix pre-cancel admission/root",
    )?;
    let mut pressure = Vec::new();
    loop {
        let left = host.arbiter.snapshot(&host.resource)?.remaining;
        if left == 0 {
            break;
        }
        pressure.push(source.reserve_workspace(left.min(32 * budget::MIB) as usize)?);
    }
    let full = host.arbiter.snapshot(&host.resource)?.reserved;
    let pressure_refused = matches!(
        session.observe_initial_core_with(11, None, |_| Err(Error::State(
            "pressure reached callback".into()
        ))),
        Err(Error::Limit(_))
    );
    need(
        pressure_refused
            && session.binding() == base
            && host.arbiter.snapshot(&host.resource)?.reserved == full,
        "real exhausted prefix admission before compute",
    )?;
    drop(pressure);
    need(
        host.arbiter.snapshot(&host.resource)?.reserved == baseline,
        "pressure owners released",
    )?;
    let (mut callbacks, mut sampled) = (0, baseline);
    let (mut reader, mut old_reader) = (None, None);
    let (mut inner_report, mut prefix_report) = (None, None);
    let result = session.observe_initial_core_with(11, None, |a| {
        callbacks += 1;
        sampled = sampled.max(host.arbiter.snapshot(&host.resource)?.reserved);
        let checked = (|| -> Proof<()> {
            if reader.is_none() {
                reader = Some(reference::initial_core::Inner::open(&accepted)?);
            }
            if old_reader.is_none() {
                old_reader = Some(reference::layer9::Prefix::open(&old)?);
            }
            let prefix = old_reader
                .as_ref()
                .ok_or("completed old prefix")?
                .compare(0, a)?;
            let inner = reader
                .as_ref()
                .ok_or("completed inner reader")?
                .compare(a, source_identity)?;
            let prefix_failure = prefix.refusal().map(str::to_owned);
            let first = inner["first"]
                .as_object()
                .map(|_| inner["first"].to_string());
            prefix_report = Some(prefix.into_report());
            inner_report = Some(inner);
            if let Some(why) = first.or(prefix_failure) {
                return Err(format!("initial core exact diagnostic: {why}").into());
            }
            Ok(())
        })();
        checked
            .map_err(|e| Error::State(format!("completed original initial-core comparison: {e}")))
    });
    let failure = result.err().map(|e| e.to_string());
    let unchanged = session.binding() == base;
    need(
        unchanged,
        "unpublished prefix must never modify original root",
    )?;
    drop(session);
    drop(operators);
    drop(source);
    if let Some(r) = &reader {
        r.verify_after()?;
    }
    if let Some(r) = &old_reader {
        r.verify_after()?;
    }
    super::verify_source_paths(directory, &held)?;
    drop(reader);
    drop(old_reader);
    drop(held);
    drop(control);
    let end = host.arbiter.snapshot(&host.resource)?;
    need(
        end.reserved == 0 && !end.fenced,
        "prefix source/outputs/tape tickets end0",
    )?;
    write_new(
        report,
        &json!({"schema":"zg.original27b-rust-initial-core-prefix-diagnostic.v1",
        "status":if failure.is_none(){"PASS_UNPUBLISHED_PREFIX_COMPARISON_ONLY"}else{"FAIL_DIAGNOSTIC_NUMERICAL_OR_EXECUTION"},
        "failure":failure,"snapshot":source::SNAPSHOT,"source_identity":format!("{source_identity:?}"),
        "actual_source_roles":851,"requested_prefix_scope":{"used_source_roles":135,"layers":10,"candidate_state_domains":18},
        "used_source_roles":completed_scope(callbacks,135),"completed_prefix_layers":completed_scope(callbacks,10),"completed_candidate_state_domains":completed_scope(callbacks,18),"published_domains":0,
        "source_headers_and_all_bytes_verified":true,"actual_source_codec":"BF16","full_forward":false,"final_or_head_executed":false,
        "accepted_original_reference":accepted,"old_prefix_reference":old,"completed_callbacks":callbacks,"committed_calls":0,
        "inner_comparison":inner_report,"prefix_comparison":prefix_report,"failed_root_unchanged":unchanged,
        "pre_cancel_refused":precancel,"actual_pressure_refused":pressure_refused,
        "host_limit":budget::HOST,"derived_reservation_upper":derived,"max_observed_boundary_reserved_not_peak":sampled,
        "final_host_reserved":end.reserved,"host_fenced":end.fenced,"full_hf_conv4_comparison_requested":true,
        "mkl_exp_producer_qualified":false,"whole_model_numerical_qualified":false,"native_serving_performance_qualified":false,
        "elapsed_seconds_not_benchmark":start.elapsed().as_secs_f64()}),
    )?;
    if let Some(why) = failure {
        return Err(why.into());
    }
    Ok(())
}

// Partial private execution before callback is not established as complete or0.
fn completed_scope(callbacks: usize, count: usize) -> Option<usize> {
    if callbacks == 1 { Some(count) } else { None }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incomplete_private_work_has_unknown_completed_scope() {
        for count in [135, 10, 18] {
            assert_eq!(
                serde_json::to_value(completed_scope(0, count)).unwrap(),
                serde_json::Value::Null
            );
            assert_eq!(completed_scope(1, count), Some(count));
            assert_eq!(completed_scope(2, count), None);
        }
    }
}
