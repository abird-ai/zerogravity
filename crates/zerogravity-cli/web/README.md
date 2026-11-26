# Embedded Web control UI

SolidJS / Vite, no Node runtime in the product. The optional Rust CLI
feature **web** embeds exactly the committed production assets in
dist/; the default product includes this feature. The native and Web
clients share the same foreground runtime and active configuration
selection and one host-owned action slot. Home / Running / Serve support
explicit resident load, invoke, load-and-run, unload, and inference start/
stop. Models / Downloads / Devices and Profiles / Config / Logs mutation
surfaces follow in UX-9C–D.

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
load/run/serve/unload with ZG_TEST_BROWSER=1 and the original approved
fixture (no download). The usual Rust gate runs the embedded HTTP and native terminal fixtures without
requiring Node or Firefox.

The bundled third-party notice is served at /ui/licenses.txt.
