# Local HTTP/1 server framing patch

Upstream: Hyper 1.13.0, MIT; registry archive SHA-256
`229c6ff5584df4c6becbfe00567128bac768c8cd32ebc8a8366bd81ed072e8f3`.
Upstream source revision is recorded in `.cargo_vcs_info.json`.

ZeroGravity requires strict rejection of ambiguous HTTP/1 request framing.
Hyper otherwise normalizes mixed Transfer-Encoding/Content-Length and
identical repeated Content-Length before the application sees headers.
The sole production source change is in the existing server request parser
(`src/proto/h1/role.rs`): reject either ordering of mixed framing and
repeated Content-Length, before application dispatch. Client parsing and
encoding are unchanged. The matching upstream server decoder assertions
are updated to reflect this policy.

Remove this patch when a maintained upstream parser offers the same strict
server policy. No second HTTP parser, runtime-specific fork or nested
Cargo.lock. Production loopback tests in zerogravity-serving qualify the
actual patched dependency and public ingress.
