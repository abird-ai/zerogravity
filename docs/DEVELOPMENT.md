# Development, builds and release definitions

Use [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution and AI-assisted plan
authorship, [AGENTS.md](../AGENTS.md) for repository boundaries, and the
[plan index](../.agents/plans/TODO.md) for current implementation and test work.
The architecture is described in the [engineering specification](ENGINEERING_SPECIFICATION.md).
This guide describes the checked-in build interfaces; it does not certify every
platform, model or release target.

## Local Rust development

The [toolchain](../rust-toolchain.toml) pins Rust 1.95.0 with rustfmt and Clippy.
The virtual [workspace](../Cargo.toml) has one Cargo.lock and one package-version
authority. Its default members are the library facade and CLI; default inference
features select CPU, Qwen3.5 and text without the GPU dependency closure.

From the repository root, with the locked crate sources already available:

```sh
cargo build --release --locked --offline -p zerogravity-cli
cargo run --locked --offline -p zerogravity-cli -- --help
cargo run --locked --offline -p zerogravity-cli -- --version
cargo fmt --all -- --check
```

Source installation uses `cargo install --locked --offline --path crates/zerogravity-cli`.
The sole product executable is `zerogravity`; argument handling and text output
belong to the [CLI package](../crates/zerogravity-cli/README.md).

After moving or copying a checkout, regenerate affected build/test outputs or
use a fresh Cargo target directory. Integration tests can embed an absolute
`CARGO_BIN_EXE_zerogravity` path from the old checkout even when the current
binary runs correctly. Such caches are disposable build inputs, not retained
test evidence; keep model and SDK directories outside any cache cleanup scope.

Run the complete repository gate with the approved cached inputs and actual
user/network isolation:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

[check.sh](../scripts/check.sh) selects the CPU owners, optional cache tests and
feature-isolation checks explicitly. A workspace-wide `--all-features` check also
selects GPU diagnostics and is not the default CPU gate. Trained-model and
physical-device tests require their actual fixtures, permissions and qualified
scope; ordinary builds and source-free tests do not replace them.

GPU products are explicit builds, for example the Vulkan text frontend:

```sh
cargo build --release --locked --offline -p zerogravity-cli \
  --no-default-features --features model-qwen35,text,vulkan
```

CUDA and HIP use their explicit `cuda` or `hip` feature and the required existing
native runtime/toolchain setup. Follow the [GPU library runbook](../crates/zerogravity-gpu/README.md)
for backend-specific builds, source/physical limits and downstream Cargo patches.
The [qualification tool](../tools/gpu-qualify/README.md) is separate from the
product. There is no independent probe workspace or second lockfile. Optional
backends, native LLVM code generation and vendor SDKs have distinct requirements;
an offline Cargo flag alone cannot prevent a dependency build script from using
the network. Keep actual network isolation and applicable acquisition authority.

## Nix development and native checks

The [flake](../flake.nix) pins nixpkgs, crane and rust-overlay. Native dependency
builds, packages and checks are separate derivations. Its default package and
app select the CPU text frontend.

```sh
nix develop
nix build .#zerogravity
nix run .# -- --version
nix flake check
```

Nix can fetch declared inputs and toolchains that are not already present.
These commands are not an instruction to acquire unapproved models or SDKs.
The [source filter](../nix/source.nix) and Git flake inputs determine the included
files; local untracked changes are not automatically part of a Git flake source.
Local Cargo success does not establish a Nix, cross-target or installed-service
pass.

## CPU cross-release definitions

On an x86_64 Linux build host, [nix/release.nix](../nix/release.nix) defines six
individual CPU assets and SHA-256 sidecars:

| Asset | Rust target |
| --- | --- |
| `zerogravity-linux-x86_64` | `x86_64-unknown-linux-musl` |
| `zerogravity-linux-aarch64` | `aarch64-unknown-linux-musl` |
| `zerogravity-windows-x86_64.exe` | `x86_64-pc-windows-gnu` |
| `zerogravity-windows-aarch64.exe` | `aarch64-pc-windows-gnullvm` |
| `zerogravity-macos-x86_64` | `x86_64-apple-darwin` |
| `zerogravity-macos-aarch64` | `aarch64-apple-darwin` |

With the required platform inputs authorized and available:

```sh
nix build .#release-all
./scripts/build-release-artifacts.sh dist
```

The [helper](../scripts/build-release-artifacts.sh) materializes and checks the
assets plus VERSION and PLATFORMS.txt. Linux assets use static musl; Windows
uses MinGW/LLVM-MinGW, and macOS uses an explicitly fetched Apple SDK plus
LLVM/LLD. The macOS definitions produce unsigned binaries. GPU backends are not
bundled in this six-target set. Cross-compilation, native execution on each
target, model numerics and platform signing/notarization are separate gates;
their definitions are not a claim of completed qualification.

The [CI workflow](../.github/workflows/build-binaries.yml) runs native quality
checks on PRs and main pushes. The six-target build runs on manual dispatch or
version tags after quality checks. Publication runs only for a tag matching
`v<workspace.package.version>` and refuses to overwrite an existing release.
Publishing or creating tags still requires the applicable explicit user action.

## Release installers

[install.sh](../install.sh) and [install.ps1](../install.ps1) select an OS/CPU
asset, verify its SHA-256 and reported binary version, and replace the executable
under a user-selected directory. They install no service. They require an
explicit release repository or base URL; no remote repository is guessed.
For an authorized, existing release:

```sh
ZEROGRAVITY_REPO=OWNER/zerogravity ZEROGRAVITY_VERSION=0.1.0 \
  ZEROGRAVITY_INSTALL_DIR="$HOME/.local/bin" ./install.sh
```

```powershell
.\install.ps1 -Repository "OWNER/zerogravity" -Version "0.1.0"
```

The base URL override is `ZEROGRAVITY_RELEASE_BASE_URL` or PowerShell
`-ReleaseBaseUrl`. A checksum fetched from the same release host detects
corruption; it does not authenticate against a compromised host. Release signing
and Windows/macOS signing remain separately governed work. The offline installer
fixture is `./scripts/test-install.sh ./target/release/zerogravity` after building
the local executable.

## NixOS service definition

The flake exports `nixosModules.default` and `nixosModules.zerogravity`. After
importing the module, keep its service disabled until serving is implemented:

```nix
services.zerogravity.enable = false;
```

The [module](../nix/module.nix) defines package, absolute workingDirectory,
extraArgs, non-secret environment and optional runtime environmentFile settings.
An enabled unit invokes `zerogravity serve`, which currently refuses as
unimplemented; `Restart = "no"` prevents a restart loop. The unit uses
DynamicUser, private temporary storage, state/cache directories, journal output,
a restricted umask and no automatic firewall exposure. It is a service
definition, not a working server. Keep secrets out of the Nix store and ordinary
environment attributes; a future serving change needs its own integration tests.
