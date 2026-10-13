{
  pkgs,
  system,
  nixpkgs,
  version,
  mkTarget,
  mkCross,
}: let
  lib = pkgs.lib;

  mkDist = {
    package,
    asset,
    binary ? "zerogravity",
  }:
    pkgs.runCommand "zerogravity-${asset}-distribution" {
      nativeBuildInputs = [pkgs.coreutils];
    } ''
      mkdir -p "$out"
      cp "${package}/bin/${binary}" "$out/${asset}"
      chmod 0755 "$out/${asset}"
      (cd "$out" && sha256sum "${asset}" > "${asset}.sha256")
    '';

  linuxX86_64 = mkCross {
    localSystem = system;
    crossSystem = {
      config = "x86_64-unknown-linux-musl";
      libc = "musl";
    };
    target = "x86_64-unknown-linux-musl";
    rustFlags = "-C target-feature=+crt-static";
  };

  linuxAarch64 = mkCross {
    localSystem = system;
    crossSystem = {
      config = "aarch64-unknown-linux-musl";
      libc = "musl";
    };
    target = "aarch64-unknown-linux-musl";
    rustFlags = "-C target-feature=+crt-static";
  };

  windowsX86_64 = mkCross {
    localSystem = system;
    crossSystem = {
      config = "x86_64-w64-mingw32";
      libc = "msvcrt";
    };
    target = "x86_64-pc-windows-gnu";
  };

  # LLVM/MinGW UCRT supports the Rust Windows ARM64 gnullvm target.
  llvmMingwBase = pkgs.callPackage (nixpkgs + "/pkgs/applications/emulators/wine/llvm-mingw.nix") {};
  llvmMingw = llvmMingwBase.overrideAttrs (old: {
    buildInputs =
      (old.buildInputs or [])
      ++ [pkgs.zstd pkgs.libxml2_13 pkgs.xz pkgs.ncurses];
  });

  windowsAarch64 = mkTarget {
    inherit pkgs;
    target = "aarch64-pc-windows-gnullvm";
    nativeBuildInputs = [llvmMingw];
    extraArgs = {
      CARGO_TARGET_AARCH64_PC_WINDOWS_GNULLVM_LINKER = "${llvmMingw}/bin/aarch64-w64-mingw32-clang";
      CC_aarch64_pc_windows_gnullvm = "${llvmMingw}/bin/aarch64-w64-mingw32-clang";
      CXX_aarch64_pc_windows_gnullvm = "${llvmMingw}/bin/aarch64-w64-mingw32-clang++";
      AR_aarch64_pc_windows_gnullvm = "${llvmMingw}/bin/aarch64-w64-mingw32-ar";
      RANLIB_aarch64_pc_windows_gnullvm = "${llvmMingw}/bin/aarch64-w64-mingw32-ranlib";
    };
  };

  # Pinned Apple SDK, adapted from the existing read-only build reference.
  # macOS cross builds require downloading Apple's SDK; source/license policy
  # and platform execution tests are independent of flake evaluation.
  fetchMacosSdk = pkgs.callPackage (nixpkgs + "/pkgs/by-name/ap/apple-sdk/common/fetch-sdk.nix") {};
  macosSdk = fetchMacosSdk {
    urls = [
      "https://swcdn.apple.com/content/downloads/14/48/052-59890-A_I0F5YGAY0Y/p9n40hio7892gou31o1v031ng6fnm9sb3c/CLTools_macOSNMOS_SDK.pkg"
      "https://web.archive.org/web/20250211001355/https://swcdn.apple.com/content/downloads/14/48/052-59890-A_I0F5YGAY0Y/p9n40hio7892gou31o1v031ng6fnm9sb3c/CLTools_macOSNMOS_SDK.pkg"
    ];
    version = "14.4";
    hash = "sha256-QozDiwY0Czc0g45vPD7G4v4Ra+3DujCJbSads3fJjjM=";
  };

  mkMacos = target: let
    targetEnv = lib.toUpper (lib.replaceStrings ["-"] ["_"] target);
    cc = pkgs.writeShellScript "zerogravity-${target}-cc" ''
      export PATH="${lib.makeBinPath [pkgs.llvmPackages.lld]}:$PATH"
      exec ${pkgs.llvmPackages.clang-unwrapped}/bin/clang \
        --target=${target} -isysroot ${macosSdk} \
        -mmacosx-version-min=11.0 -fuse-ld=lld "$@"
    '';
  in
    mkTarget {
      inherit pkgs target;
      nativeBuildInputs = [
        pkgs.llvmPackages.clang-unwrapped
        pkgs.llvmPackages.lld
        pkgs.llvmPackages.llvm
      ];
      extraArgs =
        {
          "CARGO_TARGET_${targetEnv}_LINKER" = cc;
          MACOSX_DEPLOYMENT_TARGET = "11.0";
          SDKROOT = macosSdk;
        }
        // {
          "CC_${lib.replaceStrings ["-"] ["_"] target}" = cc;
        };
    };

  macosX86_64 = mkMacos "x86_64-apple-darwin";
  macosAarch64 = mkMacos "aarch64-apple-darwin";

  linuxX86_64Dist = mkDist {
    package = linuxX86_64.package;
    asset = "zerogravity-linux-x86_64";
  };
  linuxAarch64Dist = mkDist {
    package = linuxAarch64.package;
    asset = "zerogravity-linux-aarch64";
  };
  windowsX86_64Dist = mkDist {
    package = windowsX86_64.package;
    asset = "zerogravity-windows-x86_64.exe";
    binary = "zerogravity.exe";
  };
  windowsAarch64Dist = mkDist {
    package = windowsAarch64.package;
    asset = "zerogravity-windows-aarch64.exe";
    binary = "zerogravity.exe";
  };
  macosX86_64Dist = mkDist {
    package = macosX86_64.package;
    asset = "zerogravity-macos-x86_64";
  };
  macosAarch64Dist = mkDist {
    package = macosAarch64.package;
    asset = "zerogravity-macos-aarch64";
  };

  releaseAll =
    pkgs.runCommand "zerogravity-release-all" {
      nativeBuildInputs = [pkgs.coreutils];
    } ''
          mkdir -p "$out"
          cp "${linuxX86_64Dist}"/zerogravity-* "$out/"
          cp "${linuxAarch64Dist}"/zerogravity-* "$out/"
          cp "${windowsX86_64Dist}"/zerogravity-* "$out/"
          cp "${windowsAarch64Dist}"/zerogravity-* "$out/"
          cp "${macosX86_64Dist}"/zerogravity-* "$out/"
          cp "${macosAarch64Dist}"/zerogravity-* "$out/"
          printf '%s\n' '${version}' > "$out/VERSION"
          cat > "$out/PLATFORMS.txt" <<'EOF'
      zerogravity-linux-x86_64: Linux x86_64 (static musl)
      zerogravity-linux-aarch64: Linux ARM64 (static musl)
      zerogravity-windows-x86_64.exe: Windows x86_64 (GNU)
      zerogravity-windows-aarch64.exe: Windows ARM64 (LLVM-MinGW)
      zerogravity-macos-x86_64: macOS x86_64 (unsigned)
      zerogravity-macos-aarch64: macOS ARM64 (unsigned)
      All assets select the CPU/Qwen3.5 text frontend; GPU backends are not bundled.
      Cross-compilation does not establish runtime or numerical qualification on a target.
      EOF
    '';
in {
  cross-linux-x86_64 = linuxX86_64.package;
  cross-linux-aarch64 = linuxAarch64.package;
  cross-windows-x86_64 = windowsX86_64.package;
  cross-windows-aarch64 = windowsAarch64.package;
  cross-macos-x86_64 = macosX86_64.package;
  cross-macos-aarch64 = macosAarch64.package;

  dist-linux-x86_64 = linuxX86_64Dist;
  dist-linux-aarch64 = linuxAarch64Dist;
  dist-windows-x86_64 = windowsX86_64Dist;
  dist-windows-aarch64 = windowsAarch64Dist;
  dist-macos-x86_64 = macosX86_64Dist;
  dist-macos-aarch64 = macosAarch64Dist;

  release-all = releaseAll;
}
