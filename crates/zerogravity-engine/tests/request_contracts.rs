use zerogravity_core::task::TaskKind;
use zerogravity_engine::{Error, TaskInput, TaskRequest, scheduler::SchedulerLimits};
#[test]
fn malformed_selector_future_task_and_overflow_remain_distinct_refusals() {
    let mut request = TaskRequest {
        model: "".into(),
        input: TaskInput::Text("test".into()),
        max_output: 1.try_into().unwrap(),
    };
    assert!(matches!(request.validate(1024), Err(Error::Malformed(_))));
    request.model = "unloaded".into();
    request.input = TaskInput::Unavailable(TaskKind::SpeechSynthesis);
    assert!(matches!(request.validate(1024), Err(Error::Unsupported(_))));
    assert!(matches!(
        SchedulerLimits {
            max_active: usize::MAX,
            max_queued: Some(1),
            max_input_bytes: 1,
        }
        .validate(),
        Err(Error::Limit(_))
    ));
}
