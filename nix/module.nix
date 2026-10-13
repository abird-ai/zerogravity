# NixOS service wiring for the future zerogravity server.
# The CLI deliberately rejects serve until the server is implemented.
# Keep disabled by default and avoid a systemd restart loop until implemented.
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
    enable = mkEnableOption "zerogravity (server command is currently unimplemented)";
    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      description = "The zerogravity executable package.";
    };
    workingDirectory = mkOption {
      type = types.str;
      default = "/var/lib/zerogravity";
      example = "/srv/models";
      description = "Absolute working directory for a future model-serving service.";
    };
    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Extra arguments for a future 'zerogravity serve' command; currently not implemented.";
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
      description = "Zerogravity AI runtime (future service; serve is not implemented)";
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
          # The bootstrap serve command exits 2; do not restart it indefinitely.
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
