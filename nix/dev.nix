{native}: let
  inherit (native) pkgs;
in {
  shell = native.craneLib.devShell {
    checks = {inherit (native) tests clippy;};
    packages = [
      pkgs.alejandra
      pkgs.shellcheck
      pkgs.actionlint
      pkgs.python3
      pkgs.curl
    ];
    RUST_BACKTRACE = "1";
  };

  formatter = pkgs.alejandra;
}
