# Embedded Web control UI

SolidJS / Vite, no Node runtime in the product. The optional Rust CLI
feature **web** embeds exactly the committed production assets in
dist/; the default product includes this feature. The native and Web
clients share the same foreground runtime and active configuration
selection. UX-9A exposes read-only snapshots and event invalidations.

Run **zerogravity web** (or **zerogravity --profile NAME --offline web**).
The host prints a private loopback URL; open it in your browser. Use
--listen 127.0.0.1:PORT for a fixed port, or the default ephemeral port.
The native TUI Home/Serve **b** action exposes the same-process private
URL without launching a second runtime. Closing a browser tab does not
retire its host. Ctrl-C/SIGTERM or confirmed native process quit joins
the Web connections before runtime retirement.

The random 256-bit control credential is transferred in the URL fragment,
removed from the visible URL, and held in tab-scoped session storage until
lock/tab close. It is independent of the OpenAI inference key. Loopback,
numeric Host, same-origin checks and explicit bearer are required for
owner data; public static assets contain no credentials. There is no
mutation route or remote bind in this slice.

## Development / reproducible build

From this directory:

    npm ci --offline --no-audit --no-fund
    npm run test
    npm run build

Use the checked-in lockfile and already cached dependencies; do not update
versions implicitly. Review/commit source and generated dist/ together.

After building the Rust CLI, run the real Firefox smoke:

    node src/browser-smoke.mjs PATH_TO_CLI

Use installed Firefox in a private loopback network namespace. The usual
Rust gate runs the embedded HTTP and native terminal fixtures without
requiring Node or Firefox.

The bundled third-party notice is served at /ui/licenses.txt.
