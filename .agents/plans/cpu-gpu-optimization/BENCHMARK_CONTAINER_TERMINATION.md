# l5 benchmark container termination

Author: ROOT / GPT-6.1 Sol xhigh

<!-- Previous incident labels: UD6, exit137. Exit status137 remains data. -->

The24Oct incident belongs to the local l5 .8B C16 quiet-repeat benchmark,
not the x2/Qwen27B capture. Evidence points to the benchmark owner's five-second
container stop escalating against a server without a serving shutdown handler.
This is a causal inference; the historical fatal signal and sender were not
recorded. OOM is unconfirmed. No benchmark or capture is qualified by this review.

## Established facts

- Original source21 package SHA256:
  `988c724567f877e0461a49956af62720d35739eae743306e4f1c105818b452cb`.
  Native binary SHA256:
  `d5cb1a0522b44de0118bc702ccdb69afb9f9540819e1946868cd331a7d025d88`.
  Image config:`sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe`.
  Binary/image values are original launch pins; their payloads were not reread.
- Native container:
  `cd678fc957c9428a0f33de347a90efb0002898b44ccacce8e949738642a1a28a`.
  Nonce:`07e0a0b37142376e3d2b90ed8fc1874e`. Foreground Podman PID3452517;
  sampled native payload host PID3452550; bootstrap namespace PID1 is preserved
  by exec. No original NSpid translation is retained. Stop actor PID3453765.
- Original bootstrap verifies that binary and replaces itself with it using
  `os.execv`; the launch has no init wrapper. Sixteen original CLI/serve/pull
  Rust files and the Cargo manifest were recovered or verified byte-for-byte
  against their source21 manifest hashes (13Git blobs;4identical current files). The application serving path blocks in `listener.incoming()` and has
  no signal/shutdown handler. Pull's signal registration is outside that path.
- Original `owned_container_v4.cleanup_container` verifies the owned
  name/nonce/CID, synchronously invokes `podman stop --time 5`, and proves
  container absence before foreground-group cleanup in this successful recorded
  cleanup. Its successful stop output,
  return code and start/end timestamps were discarded. Later foreground TERM/
  KILL flags describe Podman CLI cleanup, not the preceding payload death.
- Both original owner receipts close with EOF, no cleanup errors and no surviving
  container/group. Native foreground exit status is137; stock exits0. Healthy
  owner closure does not establish a successful benchmark or graceful shutdown.

## New timeline evidence

Fresh bounded journal queries on26Oct authenticate the original l5 machine
`8a88878aea7d4ea9bfa96178f9047d0f` and same boot
`650ed4ea-2acd-4c20-bb7d-1171a7426b51`. Journal metadata identifies Podman5.8.8.
The first query's dashed boot-ID match was invalid and returned no matches;
it is retained separately and proves no absence. The corrected query returned
17records, including a stop-process scope creation absent from the old16-record
CID-only query. A second query of the exact stop/payload PIDs and native/conmon
units, without a message filter, found only the startup line and removal event.
All queries closed with complete output; neither invokes Podman or changes its store.

|24Oct UTC, journal receipt time|Authenticated event|
|---|---|
|10:54:20.739345|Native container started|
|10:54:44.015826|Native serving startup message|
|10:54:46.405129|`podman-3453765.scope` started|
|10:54:51.693739|Native container died; original journal status137|
|10:54:51.694319|Container scope reports1.8GiB peak|
|10:54:51.797603|The same stop actor removed the exact native container|

Death follows stop-scope creation by5.288610s on both wall and boot-monotonic
clocks (520825590630→520830879240µs). Removal's process-relative clock
is5.426095554s. This ordering supports stop-timeout escalation, rather than the
later foreground-group kill, as the likely mechanism. Podman documents stop
signal followed by SIGKILL after the configured timeout; Linux namespace PID1
requires an installed handler for ordinary ancestor signals. These mechanisms
fit the pinned application and observed timing, but are not a recorded signal
trace. [Podman stop](https://docs.podman.io/en/latest/markdown/podman-stop.1.html),
[Linux PID namespaces](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html).

## Limits and next work

- No retained delivered-signal/sender record, configured image stop signal,
  runtime signal disposition, or original cgroup OOM-counter delta establishes
  the final cause. The12GiB cap,1.8GiB peak and empty earlier kernel query prove
  neither OOM nor its absence. Current counters cannot reconstruct a historical
  delta. [Linux cgroup memory events](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).
- Do not alter sealed source21 or promote its historical measurement to current
  qualification. Preserve measured, terminal, numerical, physical and cleanup
  outcomes independently, including every rejected/interrupted scope.
- Before a future applicable run: implement and validate serving shutdown at
  the actual owner boundary; retain each stop command's identity, argv, monotonic
  timestamps, return code and full output, plus pre-removal inspect/OOM evidence.
  Explicitly distinguish a graceful exit, owned escalation and an unknown kill.
- x2 capture still needs its own current resources, exact inputs, caller ABI and
  capture gates. This local incident supplies no x2 host, SDK or model permission.

Evidence: [archive/source review](../../runs/benchmark-container-termination-20261026/archive/REPORT.md),
[fresh journal query](../../runs/benchmark-container-termination-20261026/root/journal-corrected-result.json),
[exact-actor query](../../runs/benchmark-container-termination-20261026/root/actor-journal-result.json).
Raw stdout/stderr and original rejected/closed receipts remain in those run directories.
