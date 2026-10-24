use super::*;
use cubecl_common::profile::{Instant, ProfileTicks};
use std::{cell::Cell, rc::Rc, time::Duration};

#[test]
fn original_error_and_panic_abandon_exactly_once() {
    for panics in [false, true] {
        let abandoned = Rc::new(Cell::new(0));
        let count = abandoned.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let _token = OpenToken::new(17, move |n| {
                assert_eq!(n, 17);
                count.set(count.get() + 1);
            });
            if panics {
                panic!("original actor panic");
            }
            Err(Error::State("original node cut".into()))
        }));
        assert_eq!(abandoned.get(), 1);
        if panics {
            assert_eq!(
                outcome.unwrap_err().downcast_ref::<&str>(),
                Some(&"original actor panic")
            );
        } else {
            assert!(matches!(outcome.unwrap(),Err(Error::State(s)) if s=="original node cut"));
        }
    }
}
#[test]
fn end_success_or_returned_failure_never_double_abandons() {
    for fails in [false, true] {
        let abandoned = Rc::new(Cell::new(0));
        let count = abandoned.clone();
        let ended = Cell::new(0);
        {
            let mut token = OpenToken::new(9, move |_| count.set(count.get() + 1));
            let out = token.finish(|n| {
                assert_eq!(n, 9);
                ended.set(ended.get() + 1);
                if fails {
                    Err(failure("end failed"))
                } else {
                    Ok(23)
                }
            });
            assert_eq!(out.is_err(), fails);
            assert!(token.finish(|_| Ok(0)).is_err());
        }
        assert_eq!(ended.get(), 1);
        assert_eq!(abandoned.get(), 0);
    }
}
#[test]
fn panic_during_end_still_abandons_without_replacing_original_panic() {
    let abandoned = Rc::new(Cell::new(0));
    let count = abandoned.clone();
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut token = OpenToken::new(7, move |_| {
            count.set(count.get() + 1);
            abandon_safely(|| panic!("secondary abandon"));
        });
        let _ = token.finish(|_| -> Result<()> { panic!("original end panic") });
    }));
    assert_eq!(
        out.unwrap_err().downcast_ref::<&str>(),
        Some(&"original end panic")
    );
    assert_eq!(abandoned.get(), 1);
}
fn duration() -> ProfileDuration {
    ProfileDuration::new_device_time(async {
        let now = Instant::now();
        ProfileTicks::from_start_end(now, now + Duration::from_micros(12))
    })
}
#[test]
fn exactly_whole_and_head_slots_required_and_one_use() {
    let mut slots = Slots::default();
    assert!(slots.take().is_err());
    assert!(slots.begin(2).is_err());
    slots.begin(WHOLE).unwrap();
    assert!(slots.begin(WHOLE).is_err());
    slots.durations[WHOLE] = Some(duration());
    assert!(slots.take().is_err());
    slots.begin(HEAD).unwrap();
    slots.durations[HEAD] = Some(duration());
    let taken = slots.take().unwrap();
    assert_eq!(taken.len(), 2);
    assert!(slots.take().is_err());
    assert!(slots.begin(HEAD).is_err());
}
#[test]
fn failed_window_cannot_become_successful_or_zero_measurement() {
    let mut slots = Slots {
        seen: [true; 2],
        durations: [Some(duration()), Some(duration())],
        failed: true,
        ..Default::default()
    };
    assert!(slots.take().is_err());
    let now = Instant::now();
    assert!(
        futures_lite::future::block_on(resolve_device(ProfileDuration::new_system_time(now, now)))
            .is_err()
    );
    assert!(
        futures_lite::future::block_on(resolve_device(ProfileDuration::new_device_time_maybe(
            async { None }
        )))
        .is_err()
    );
    assert!(valid_duration(Duration::from_secs(3600)).is_err());
    assert!(valid_duration(Duration::from_secs(3601)).is_err());
    assert_eq!(valid_duration(Duration::ZERO).unwrap(), 0);
    assert!(futures_lite::future::block_on(resolve_device(duration())).unwrap() > 0);
}
#[test]
fn inactive_head_observer_keeps_original_result_and_calls_once() {
    assert!(ACTIVE.with(|s| s.borrow().is_none()));
    let calls = Cell::new(0);
    let result = head(|| {
        calls.set(calls.get() + 1);
        Err::<(), _>(Error::State("real head error".into()))
    });
    assert!(matches!(result,Err(Error::State(s))if s=="real head error"));
    assert_eq!(calls.get(), 1);
}
