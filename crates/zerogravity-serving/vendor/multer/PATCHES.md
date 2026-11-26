# Strict HTTP multipart form policy

Upstream: multer 3.1.0 (MIT); registry archive SHA-256
`83e87776546dc87511aa5ee218730c92b666d7264ab6ed41f9d215af9cd5224b`. Source revision: `.cargo_vcs_info.json`.

ZeroGravity's bounded HTTP form subset rejects MIME preambles/epilogues:
the initial delimiter starts the body; the closing delimiter is followed
only by optional transport padding, optional CRLF and decoded HTTP EOF. The upstream state machine
in src/multipart.rs is the sole boundary parser; it must not report
success while trailing bytes, framing errors or a missing HTTP EOF remain.
src/buffer.rs removes the unused preamble search helper.
src/helpers.rs rejects repeated part headers rather than overwriting them.
The upstream preamble test now expects rejection. build.rs declares the
existing nightly cfg to current Rust's check-cfg lint.
No changes to field payload parsing, size constraints or field streaming.

The serving wrapper supplies a bounded, cooperative frame stream and
charges parser headroom and retained part backing. Replace the local patch
when an upstream strict form policy provides these guarantees.

README.md has trailing whitespace stripped for repository diff hygiene.
