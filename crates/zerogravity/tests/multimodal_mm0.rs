//! MM0: source-free executable falsification tests for v5 media/task contracts.
//! No model checkpoint, GPU, tokenizer, codecs or production model execution.
//! These mock TaskDrivers use the EXISTING TaskCoordinator as sole root writer.
use std::collections::{BTreeMap, BTreeSet};
use zerogravity::state::{PreparedDomain, TaskCoordinator};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rate {
    num: u64,
    den: u64,
}
impl Rate {
    fn new(num: u64, den: u64) -> Result<Self, &'static str> {
        if num == 0 || den == 0 {
            return Err("zero media clock rate");
        }
        let g = gcd(num, den);
        Ok(Self {
            num: num / g,
            den: den / g,
        })
    }
}
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cursor {
    track: &'static str,
    ticks: u64,
    rate: Rate,
}
/// Convert a monotonic rational-clock timestamp to a second clock without
/// silently dropping fractional samples. Uses checked u128 rather than f64.
fn convert_floor(source: Cursor, target_rate: Rate) -> Result<(u64, u128, u128), &'static str> {
    let numerator = u128::from(source.ticks)
        .checked_mul(u128::from(source.rate.den))
        .and_then(|x| x.checked_mul(u128::from(target_rate.num)))
        .ok_or("media clock arithmetic overflow")?;
    let denominator = u128::from(source.rate.num)
        .checked_mul(u128::from(target_rate.den))
        .ok_or("media clock denominator overflow")?;
    let ticks = numerator / denominator;
    Ok((
        u64::try_from(ticks).map_err(|_| "media cursor overflow")?,
        numerator % denominator,
        denominator,
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ValueType {
    Tensor {
        dims: Vec<usize>,
        axes: Vec<&'static str>,
    },
    Samples {
        channels: u8,
        rate: Rate,
    },
    Frame {
        width: usize,
        height: usize,
        rate: Rate,
    },
    DecisionLogits {
        options: usize,
    },
}
impl ValueType {
    fn valid(&self) -> bool {
        match self {
            Self::Tensor { dims, axes } => {
                !dims.is_empty() && dims.len() == axes.len() && dims.iter().all(|&n| n > 0)
            }
            Self::Samples { channels, rate } => *channels > 0 && rate.num > 0,
            Self::Frame {
                width,
                height,
                rate,
            } => *width > 0 && *height > 0 && rate.num > 0,
            Self::DecisionLogits { options } => *options >= 2 && *options <= 1024,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Port {
    component: &'static str,
    port: &'static str,
    value: ValueType,
}
fn verify_assembly(components: &[&str], edges: &[(Port, Port)]) -> Result<(), &'static str> {
    let c: BTreeSet<&str> = components.iter().copied().collect();
    if c.len() != components.len() || c.is_empty() {
        return Err("duplicate component");
    }
    for (source, sink) in edges {
        if !c.contains(source.component)
            || !c.contains(sink.component)
            || !source.value.valid()
            || !sink.value.valid()
        {
            return Err("model assembly port missing/invalid");
        }
        if source.value != sink.value {
            return Err("model component codec/clock mismatch");
        }
        if source.component == sink.component {
            return Err("self edge unsupported");
        }
    }
    Ok(())
}

fn prepared(name: &str, schema: u64, frontier: u64, bytes: Vec<u8>) -> PreparedDomain {
    PreparedDomain {
        name: name.into(),
        schema,
        frontier,
        bytes,
    }
}

/// A dummy source-registered finite-choice TaskDriver. No token generation:
/// 20 independent questions share one deterministic encoding pass and score.
fn score_decisions(questions: usize, options: usize) -> Result<Vec<Vec<f32>>, &'static str> {
    if questions == 0 || questions > 128 || !(2..=32).contains(&options) {
        return Err("decision schema dimensions invalid");
    }
    let mut answer = Vec::with_capacity(questions);
    for q in 0..questions {
        let logits = (0..options)
            .map(|i| ((i as f32) - (q % options) as f32) * 0.25)
            .collect::<Vec<_>>();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = logits.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
        let total = probabilities.iter().sum::<f32>();
        for p in &mut probabilities {
            *p /= total;
        }
        answer.push(probabilities);
    }
    Ok(answer)
}

/// Dummy registered iterative flow solver (NOT a real denoiser or image
/// sampler); a completed step is published only after all required state
/// domains have prepared exactly the same global task frontier.
fn flow_step(
    coordinator: &mut TaskCoordinator,
    step: u64,
    latent: i64,
    rng: u64,
    ready: bool,
) -> Result<(i64, u64), String> {
    if !ready {
        return Err("predictor/device not ready".into());
    }
    let next = latent + (step as i64) * (rng % 3 + 1) as i64;
    let rng_next = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let root = coordinator.root().id;
    coordinator
        .publish(
            root,
            step,
            vec![
                prepared("solver", 1, step, next.to_le_bytes().to_vec()),
                prepared("rng", 2, step, rng_next.to_le_bytes().to_vec()),
            ],
            next.to_le_bytes().to_vec(),
        )
        .map_err(|e| e.to_string())?;
    Ok((next, rng_next))
}

/// MM0 mock multi-track facade. The state authority remains TaskCoordinator;
/// this validator only prepares two clock domains if all track events have
/// completed and timestamps are consistent. No independent root publication.
struct MockAV {
    owner: TaskCoordinator,
    frames: u64,
    audio_samples: u64,
}
impl MockAV {
    fn new() -> Self {
        Self {
            owner: TaskCoordinator::new(901, [("video".into(), 1), ("audio".into(), 2)]).unwrap(),
            frames: 0,
            audio_samples: 0,
        }
    }
    fn publish_frame(
        &mut self,
        frame: u64,
        samples: u64,
        video_ready: bool,
        audio_ready: bool,
    ) -> Result<(), String> {
        if !video_ready || !audio_ready {
            return Err("AV publication must await both physical ready events".into());
        }
        if frame != self.frames + 1 || samples <= self.audio_samples {
            return Err("nonmonotonic media track cursor".into());
        }
        let frame_clock = Rate::new(30000, 1001).unwrap();
        let audio_clock = Rate::new(48000, 1).unwrap();
        let expected = convert_floor(
            Cursor {
                track: "video",
                ticks: frame,
                rate: frame_clock,
            },
            audio_clock,
        )
        .map_err(str::to_owned)?
        .0;
        if expected != samples {
            return Err("audio/video timestamp mismatch".into());
        }
        let root = self.owner.root().id;
        self.owner
            .publish(
                root,
                frame,
                vec![
                    prepared("video", 1, frame, frame.to_le_bytes().to_vec()),
                    prepared("audio", 2, frame, samples.to_le_bytes().to_vec()),
                ],
                format!("frame={frame},audio={samples}").into_bytes(),
            )
            .map_err(|e| e.to_string())?;
        self.frames = frame;
        self.audio_samples = samples;
        Ok(())
    }
}

#[test]
fn mm01_rational_audio_video_clocks_are_exact_over_many_frames() {
    let video = Rate::new(30000, 1001).unwrap();
    assert_eq!(
        video,
        Rate {
            num: 30000,
            den: 1001
        }
    );
    let pcm = Rate::new(48000, 1).unwrap();
    let mut previous = 0;
    for frame in 1..=600 {
        let c = Cursor {
            track: "video",
            ticks: frame,
            rate: video,
        };
        let (audio, residue, denominator) = convert_floor(c, pcm).unwrap();
        let expected = frame * 48000 * 1001 / 30000;
        assert_eq!(audio, expected);
        assert!(residue < denominator);
        assert!(audio > previous);
        previous = audio;
    }
    assert_eq!(previous, 960960);
    assert_eq!(
        convert_floor(
            Cursor {
                track: "video",
                ticks: 5,
                rate: video
            },
            pcm
        )
        .unwrap()
        .0,
        8008
    );
    assert!(Rate::new(0, 1).is_err());
    assert!(Rate::new(3, 0).is_err());
    assert_eq!(Rate::new(48000, 2).unwrap().num, 24000);
    let huge = convert_floor(
        Cursor {
            track: "x",
            ticks: u64::MAX,
            rate: Rate::new(1, 1).unwrap(),
        },
        Rate::new(u64::MAX, 1).unwrap(),
    );
    assert!(huge.is_err());
}

#[test]
fn mm02_source_component_ports_reject_incompatible_media_formats() {
    let f = Rate::new(24, 1).unwrap();
    let film = ValueType::Frame {
        width: 512,
        height: 512,
        rate: f,
    };
    let source = Port {
        component: "predictor",
        port: "video",
        value: film.clone(),
    };
    let sink = Port {
        component: "vae",
        port: "latent",
        value: film,
    };
    assert!(verify_assembly(&["predictor", "vae"], &[(source.clone(), sink.clone())]).is_ok());
    let mut wrong = sink.clone();
    wrong.value = ValueType::Samples {
        channels: 2,
        rate: Rate::new(48000, 1).unwrap(),
    };
    assert!(verify_assembly(&["predictor", "vae"], &[(source.clone(), wrong)]).is_err());
    assert!(verify_assembly(&["predictor"], &[(source.clone(), sink)]).is_err());
    assert!(verify_assembly(&["predictor", "predictor"], &[]).is_err());
    let tensor = ValueType::Tensor {
        dims: vec![2, 128, 128],
        axes: vec!["b", "h", "w"],
    };
    assert!(tensor.valid());
    assert!(ValueType::DecisionLogits { options: 3 }.valid());
    assert!(!ValueType::DecisionLogits { options: 1 }.valid());
}

#[test]
fn mm03_dummy_flow_solver_uses_existing_atomic_state_root() {
    fn run_once() -> (i64, u64, u64) {
        let mut c = TaskCoordinator::new(33, [("solver".into(), 1), ("rng".into(), 2)]).unwrap();
        let mut latent = 4i64;
        let mut rng = 17u64;
        for step in 1..=7 {
            (latent, rng) = flow_step(&mut c, step, latent, rng, true).unwrap();
        }
        (latent, rng, c.root().id)
    }
    assert_eq!(run_once(), run_once());
    let mut c = TaskCoordinator::new(33, [("solver".into(), 1), ("rng".into(), 2)]).unwrap();
    let old = c.root().clone();
    assert!(flow_step(&mut c, 1, 7, 17, false).is_err());
    assert_eq!(c.root(), &old);
    let (latent, rng) = flow_step(&mut c, 1, 7, 17, true).unwrap();
    assert!(flow_step(&mut c, 1, latent, rng, true).is_err());
    assert_eq!(c.root().id, 1);
    // Failure in one prepared domain: do not publish partially updated latent.
    let bad = c.publish(1, 2, vec![prepared("solver", 1, 2, vec![1])], vec![]);
    assert!(bad.is_err());
    assert_eq!(c.root().id, 1);
}

#[test]
fn mm08_dummy_non_token_decisions_publish_schema_atomically() {
    let distributions = score_decisions(20, 3).unwrap();
    assert_eq!(distributions.len(), 20);
    for probs in &distributions {
        assert_eq!(probs.len(), 3);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    assert!(score_decisions(0, 3).is_err());
    assert!(score_decisions(20, 1).is_err());
    let mut c =
        TaskCoordinator::new(909, [("decision".into(), 1), ("provenance".into(), 3)]).unwrap();
    let original = c.root().clone();
    assert!(
        c.publish(0, 1, vec![prepared("decision", 1, 1, vec![3])], vec![])
            .is_err()
    );
    assert_eq!(c.root(), &original);
    let first = c
        .publish(
            0,
            1,
            vec![
                prepared(
                    "decision",
                    1,
                    1,
                    (distributions.len() as u64).to_le_bytes().to_vec(),
                ),
                prepared("provenance", 3, 1, b"mock-score-schema-1".to_vec()),
            ],
            b"schema-typed-decisions".to_vec(),
        )
        .unwrap();
    assert_eq!((first.root_id, first.sequence, first.epoch), (1, 1, 909));
    assert!(
        c.publish(
            0,
            2,
            vec![
                prepared("decision", 1, 2, vec![1]),
                prepared("provenance", 3, 2, vec![2])
            ],
            vec![]
        )
        .is_err()
    );
    assert_eq!(c.root().id, 1);
}

#[test]
fn mm07_av_synchronization_is_single_root_and_fail_closed() {
    let mut c = MockAV::new();
    let first = c.owner.root().clone();
    assert!(c.publish_frame(1, 1601, true, false).is_err());
    assert_eq!(c.owner.root(), &first);
    assert!(c.publish_frame(1, 1602, true, true).is_err());
    assert_eq!(c.owner.root(), &first);
    c.publish_frame(1, 1601, true, true).unwrap();
    assert_eq!(c.owner.root().id, 1);
    assert!(c.publish_frame(1, 1601, true, true).is_err());
    c.publish_frame(2, 3203, true, true).unwrap();
    c.publish_frame(3, 4804, true, true).unwrap();
    c.publish_frame(4, 6406, true, true).unwrap();
    c.publish_frame(5, 8008, true, true).unwrap();
    assert_eq!(c.owner.root().id, 5);
    assert_eq!(c.audio_samples, 8008);
    // No stale task worker can publish after a later commit.
    assert!(
        c.owner
            .publish(
                0,
                6,
                vec![
                    prepared("video", 1, 6, vec![1]),
                    prepared("audio", 2, 6, vec![2]),
                ],
                vec![]
            )
            .is_err()
    );
    let d: BTreeMap<_, _> = c
        .owner
        .root()
        .domains
        .iter()
        .map(|(k, v)| (k.clone(), v.bytes.len()))
        .collect();
    assert_eq!(d.len(), 2);
}

/// MM-06 adversarial duplex lifecycle: queued old-turn GPU audio can never
/// publish after a newer user turn; cancelling it also cannot free in-flight
/// buffers until an actual completion signal (mocked here).
#[test]
fn mm06_duplex_barge_in_device_epoch_and_audio_lease_liveness() {
    use zerogravity::device_events::{BufferRole, DeviceAccess, DeviceEventLedger};
    let mut device = DeviceEventLedger::new(72, 1024).unwrap();
    let old_audio = device.admit(256, BufferRole::Output).unwrap();
    let old_event = device
        .submit(0, &[(old_audio, DeviceAccess::Write)])
        .unwrap();
    device.request_cancel(old_event).unwrap();
    assert!(device.release_lease(old_audio).is_err());

    // The next user turn has a new epoch; previous audio is still physically
    // pending, but its old task state is never publishable in the new epoch.
    device.advance_epoch(73).unwrap();
    let mut coordinator =
        TaskCoordinator::new(73, [("transcript".into(), 3), ("audio".into(), 5)]).unwrap();
    let new_audio = device.admit(128, BufferRole::Output).unwrap();
    let new_event = device
        .submit(0, &[(new_audio, DeviceAccess::Write)])
        .unwrap();
    device.report_physical_ready(new_event).unwrap();
    let committed = device
        .publish_ready(
            new_event,
            &mut coordinator,
            1,
            vec![
                prepared("transcript", 3, 1, b"new turn transcript".to_vec()),
                prepared("audio", 5, 1, b"new turn audio".to_vec()),
            ],
            b"new turn playable packet".to_vec(),
        )
        .unwrap();
    assert_eq!((committed.root_id, committed.epoch), (1, 73));
    assert!(
        device
            .publish_ready(
                old_event,
                &mut coordinator,
                2,
                vec![
                    prepared("transcript", 3, 2, vec![1]),
                    prepared("audio", 5, 2, vec![2]),
                ],
                vec![]
            )
            .is_err()
    );
    assert!(device.release_lease(old_audio).is_err());
    // Only after the old physical event finishes does the old buffer retire.
    device.report_physical_ready(old_event).unwrap();
    device.release_lease(old_audio).unwrap();
    device.release_lease(new_audio).unwrap();
    assert_eq!(device.resident_bytes(), 0);
    assert_eq!(coordinator.root().id, 1);
}
