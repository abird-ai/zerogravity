{
  pkgs,
  module,
}: let
  lib = pkgs.lib;
  evaluate = value:
    import (pkgs.path + "/nixos/lib/eval-config.nix") {
      inherit pkgs;
      system = pkgs.stdenv.hostPlatform.system;
      modules = [
        module
        {system.stateVersion = "25.11";}
        value
      ];
    };

  disabled = evaluate {};
  enabled = evaluate {
    services.zerogravity.enable = true;
  };
  configured = evaluate {
    services.zerogravity = {
      enable = true;
      workingDirectory = "/srv/models";
      extraArgs = ["--future" "value with spaces"];
      environment.ZEROGRAVITY_TEST = "1";
      environmentFile = "/run/secrets/zerogravity-env";
    };
  };
  bad = evaluate {
    services.zerogravity = {
      enable = true;
      workingDirectory = "relative/path";
    };
  };

  valid = x: lib.all (a: a.assertion) x.config.assertions;
  execStart = configured.config.systemd.services.zerogravity.serviceConfig.ExecStart;
  checks = [
    (!(disabled.config.systemd.services ? zerogravity))
    (valid enabled)
    (enabled.config.systemd.services.zerogravity.serviceConfig.DynamicUser)
    (enabled.config.systemd.services.zerogravity.serviceConfig.Restart == "no")
    (enabled.config.systemd.services.zerogravity.serviceConfig.StateDirectory == "zerogravity")
    (lib.hasInfix "zerogravity" enabled.config.systemd.services.zerogravity.serviceConfig.ExecStart)
    (valid configured)
    (lib.hasInfix "serve" execStart)
    (lib.hasInfix (lib.escapeShellArg "value with spaces") execStart)
    (configured.config.systemd.services.zerogravity.serviceConfig.WorkingDirectory == "/srv/models")
    (configured.config.systemd.services.zerogravity.serviceConfig.EnvironmentFile == "/run/secrets/zerogravity-env")
    (!valid bad)
  ];
in
  assert lib.all (x: x) checks;
    pkgs.runCommand "zerogravity-nixos-module-eval" {} "touch $out"
