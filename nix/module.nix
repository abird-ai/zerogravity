# Optional foreground ZeroGravity HTTP service. The host retains runtime
# ownership; systemd collects continuous operational stderr in the journal.
{self}: {
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.zerogravity;
in {
  options.services.zerogravity = {
    enable = mkEnableOption "zerogravity foreground model-serving service";
    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      description = "The zerogravity executable package.";
    };
    workingDirectory = mkOption {
      type = types.str;
      default = "/var/lib/zerogravity";
      example = "/srv/models";
      description = "Absolute working directory for the model-serving service.";
    };
    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Arguments for 'zerogravity serve'; select a supported model and execution policy.";
    };
    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      description = "Non-secret environment variables for the service. Never embed secrets here or in the Nix store.";
    };
    environmentFile = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "/run/secrets/zerogravity-env";
      description = "Optional runtime environment file, kept outside the world-readable Nix store.";
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = lib.hasPrefix "/" cfg.workingDirectory;
        message = "services.zerogravity.workingDirectory must be absolute.";
      }
    ];

    systemd.services.zerogravity = {
      description = "ZeroGravity AI runtime";
      wantedBy = ["multi-user.target"];
      after = ["network.target"];
      environment = cfg.environment;
      serviceConfig =
        {
          Type = "simple";
          DynamicUser = true;
          StateDirectory = "zerogravity";
          CacheDirectory = "zerogravity";
          WorkingDirectory = cfg.workingDirectory;
          ExecStart = lib.concatStringsSep " " (
            [(lib.getExe cfg.package) "serve"]
            ++ map lib.escapeShellArg cfg.extraArgs
          );
          # Explicit service restart policy; do not loop on admission or config failures.
          Restart = "no";
          StandardInput = "null";
          StandardOutput = "journal";
          StandardError = "journal";
          UMask = "0077";
          NoNewPrivileges = true;
          PrivateTmp = true;
        }
        // lib.optionalAttrs (cfg.environmentFile != null) {
          EnvironmentFile = cfg.environmentFile;
        };
    };
  };
}
