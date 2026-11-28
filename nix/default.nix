{
  inputs,
  sourceRoot,
}: let
  inherit (inputs) self nixpkgs crane rust-overlay uv2nix pyproject-nix pyproject-build-systems;
  systems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
  forAllSystems = f: nixpkgs.lib.genAttrs systems f;
  overlay = import rust-overlay;

  version = (builtins.fromTOML (builtins.readFile (sourceRoot + "/Cargo.toml"))).workspace.package.version;

  mkCraneLib = pkgs: target:
    (crane.mkLib pkgs).overrideToolchain (p: let
      base = p.rust-bin.fromRustupToolchainFile (sourceRoot + "/rust-toolchain.toml");
    in
      if target == null
      then base
      else base.override {targets = [target];});

  # The portable CPU artifacts remain independent of native GPU packages.
  gpuPkgs = import nixpkgs {
    system = "x86_64-linux";
    overlays = [overlay];
    config.allowUnfree = true;
  };
  gpuRuntime = import ./gpu-runtime.nix {
    pkgs = gpuPkgs;
    inherit sourceRoot;
  };

  native = forAllSystems (system:
    import ./native.nix {
      pkgs = import nixpkgs {
        inherit system;
        overlays = [overlay];
      };
      inherit sourceRoot version mkCraneLib;
    });

  full = import ./native.nix {
    pkgs = gpuPkgs;
    runtime = gpuRuntime;
    inherit sourceRoot version mkCraneLib;
  };
  selected = system:
    if system == "x86_64-linux"
    then full
    else native.${system};

  cross = import ./cross.nix {inherit nixpkgs overlay sourceRoot version mkCraneLib;};

  development = forAllSystems (system:
    import ./dev.nix {
      native = native.${system};
      inherit sourceRoot;
      gpuPkgs =
        if system == "x86_64-linux"
        then gpuPkgs
        else null;
      runtime =
        if system == "x86_64-linux"
        then gpuRuntime
        else null;
    });
  vllm = forAllSystems (system:
    import ./vllm.nix {
      inherit sourceRoot uv2nix pyproject-nix pyproject-build-systems;
      pkgs = native.${system}.pkgs;
    });
in {
  nixosModules = {
    default = import ./module.nix {inherit self;};
    zerogravity = self.nixosModules.default;
  };

  packages = forAllSystems (system:
    {
      default = (selected system).package;
      zerogravity = (selected system).package;
      portable = native.${system}.package;
      deps = (selected system).cargoArtifacts;
    }
    // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
      gpu-runtime-headers = gpuRuntime.cudaPath;
      vllm-cuda = vllm.${system}.cuda.provider;
      vllm-rocm = vllm.${system}.rocm.provider;
    }
    // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") (import ./release.nix {
      inherit system nixpkgs version;
      inherit (cross) mkTarget mkCross;
      inherit (native.${system}) pkgs;
    }));

  apps = forAllSystems (system:
    {
      default = {
        type = "app";
        program = "${(selected system).package}/bin/zerogravity";
      };
      zerogravity = self.apps.${system}.default;
    }
    // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
      vllm-cuda = {
        type = "app";
        program = "${vllm.${system}.cuda.provider}/bin/zerogravity-vllm-cuda";
      };
      vllm-rocm = {
        type = "app";
        program = "${vllm.${system}.rocm.provider}/bin/zerogravity-vllm-rocm";
      };
    });

  checks = forAllSystems (system:
    (import ./checks.nix {
      inherit sourceRoot;
      native = native.${system};
      module = self.nixosModules.default;
    })
    // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
      native-package = full.package;
    });

  devShells = forAllSystems (system:
    {
      default = development.${system}.shell;
    }
    // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
      gpu = development.${system}.gpu;
      vllm-cuda = vllm.${system}.cuda.shell;
      vllm-rocm = vllm.${system}.rocm.shell;
    });

  formatter = forAllSystems (system: development.${system}.formatter);
}
