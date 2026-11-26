# Embedded Web control UI

SolidJS / Vite, no Node runtime in the product. The optional Rust CLI
feature **web** embeds exactly the committed production assets in
dist/; the default product includes this feature. The native and Web
clients share the same foreground runtime and active configuration
selection and one host-owned action slot. Home / Running / Serve support
explicit resident load, invoke, load-and-run, unload, and inference start/
stop. Models / Downloads / Devices also expose original cache, acquisition
and physical-device owners. Profiles / Config / Logs use the shared
configuration owner and bounded foreground operational observations.

Run **zerogravity web** (or **zerogravity --profile NAME --offline web**).
The host prints a private loopback URL; open it in your browser. Use
--listen 127.0.0.1:PORT for a fixed port, or the default ephemeral port.
The native TUI Home/Serve **b** action exposes the same-process private
URL without launching a second runtime. Closing a browser tab does not
retire its host. Ctrl-C/SIGTERM or confirmed native process quit joins
the Web connections, cancel/join the shared action worker, then retire
inference transport and runtime.

The random 256-bit control credential is transferred in the URL fragment,
removed from the visible URL, and held in tab-scoped session storage until
lock/tab close. It is independent of the OpenAI inference key. Loopback,
numeric Host, same-origin checks and explicit bearer are required for
owner data; public static assets contain no credentials. There is no
remote bind. POST also requires the exact Origin and application/json.

The wire accepts only typed actions: POST /zg/control/actions admits one
explicit action and returns a decimal string identity. GET
/zg/control/actions/ID observes the exact active or latest retained Web
receipt, including a bounded generated-text prefix and original final
usage. SSE /zg/events invalidates owner/action observations. No action
queue or automatic retry on ambiguous acceptance. Cancellation targets
an exact active Web identity; native private action results never appear
on the Web. Browser closure/lock does not cancel the process-owned action.
All source/admission/lifecycle decisions use the original resident owner.
Browser model input accepts cached selectors or blank for the trusted
selected document, not arbitrary local paths or remote URLs; acquisition
is never implicit. Unload and stop require explicit confirmation and the
observed registration or generation. The inference key is host-configured
and never accepted or exposed by the control API.

Authenticated GET /zg/control/inventory/models and /devices are cold
owner observations, not execution/admission leases. Relevant cache
mutation and terminal download events invalidate model inventory;
load/unload events invalidate device observations. Resync and explicit
refresh reread the visible shelf. Progress and token events do not scan
the cache.

Cached snapshots support exact-digest inspect, verify, load, load-and-run,
load for serving, additive aliases and preview/confirmed removal. The
original cache decides busy/referenced/incomplete/durable outcomes.
Listing is metadata only; a partial checkpoint key cannot authorize a
transfer. Repeat the exact pinned acquisition intent to revalidate a
partial. Explicit acquisition uses hf:/gh:OWNER/REPO@FULL_COMMIT plus
exact relative filenames or quant:NAME, or an approved catalog recipe.
Preview is non-mutating; actual pull requires confirmation and respects
offline and original admission/publication policy. No automatic download,
quota increase, eviction or implicit load. Direct exact-operation
pause/resume/cancel addresses the original transfer, even while the
shared worker owns the pull. The operation ID is distinct from an action
receipt ID. Device selection requires an exact observed physical ID and
cached digest for a policy check or explicit load, not a persistent
configuration change or implicit fallback.

Authenticated GET /zg/control/inventory/profiles and /config/NAME (or
/config/@active) expose the original profile list and a bounded typed
configuration observation. A browser retains only the latest exact observed
ConfigFile snapshot through an opaque token; another document observation
can require a refresh, and changed bytes refuse a stale draft. Typed
create/copy/validate/activate/set/unset/replace/confirmed removal use the
same action slot and atomic file owner as the native editor. Default,
portable and currently active documents cannot be removed. Activation
changes only future actions. The browser can edit safe scalar/physical
selector policy fields, not arbitrary paths, model graphs, prompts or
credentials. Raw YAML view and replacement are limited to wholly safe,
validated documents, normalized without comments; unrestricted raw repair
is native-only. An existing trusted document can be copied or activated
without disclosing its protected contents. No selected-profile pointer is
persisted.

GET /zg/control/inventory/logs exposes one bounded 128-entry operational
ring per foreground host (native or Web standalone). Action and owner
outcomes have safe correlations, not model text, prompts, filesystem paths
or credentials. Search/level/component and pause/follow are view-local;
collection continues, old rows can expire, and there is no durable log.
SSE configuration/logs invalidations refresh only relevant visible shelves.
A refused cold read may be retried finitely; mutations are never retried
implicitly.

## Development / reproducible build

From this directory:

    npm ci --offline --no-audit --no-fund
    npm run test
    npm run build

Use the checked-in lockfile and already cached dependencies; do not update
versions implicitly. Review/commit source and generated dist/ together.

After building the Rust CLI, run the real Firefox smoke:

    node src/browser-smoke.mjs PATH_TO_CLI

Use installed Firefox in a private loopback network namespace. The existing
cached model release regression can additionally exercise actual browser
model/cache/device actions and load/run/serve/unload with ZG_TEST_BROWSER=1
and the original approved fixture (no download). The existing ignored
pull monitor child can also exercise live transfer buttons in Firefox:

    node src/browser-smoke.mjs PATH_TO_COMBINED_TEST_BINARY --download-fixture

The child uses a deterministic protected transfer, not a remote provider
or live-cache mutation. For actual simultaneous TUI and attached Firefox
pause/resume and cancellation, use the same combined test binary:

    python3 -B ../tests/fixtures/tui_shell.py PATH_TO_COMBINED_TEST_BINARY --shared-download-fixture

The private launch capability is passed to the owned browser fixture over
stdin; both clients observe the original transfer and native quit joins
the host. The usual Rust gate runs embedded HTTP and native terminal
fixtures without requiring Node or Firefox.

The bundled third-party notice is served at /ui/licenses.txt.
