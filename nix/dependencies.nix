# Crane may stub workspace crates, but registry path patches are real
# dependencies of other crates. Keep their complete sources in the dependency
# build, without invalidating its cache for ordinary workspace source edits.
{
  lib,
  craneLib,
  sourceRoot,
}: args: let
  manifest = builtins.fromTOML (builtins.readFile (sourceRoot + "/Cargo.toml"));
  patches = map (patch: patch.path) (builtins.filter (patch: patch ? path) (builtins.attrValues (manifest.patch.crates-io or {})));
  patched = lib.cleanSourceWith {
    inherit (args) src;
    name = "zerogravity-patched-dependencies";
    filter = path: type: let
      relative = lib.removePrefix "${toString sourceRoot}/" (toString path);
    in
      lib.any (root:
        relative
        == root
        || lib.hasPrefix "${root}/" relative
        || (type == "directory" && lib.hasPrefix "${relative}/" root))
      patches;
  };
in
  craneLib.buildDepsOnly (args
    // {
      extraDummyScript =
        lib.concatMapStringsSep "\n" (root: ''
          rm -r -- "$out"/${lib.escapeShellArg root}
          cp -R --no-preserve=ownership "${patched}"/${lib.escapeShellArg root} "$out"/${lib.escapeShellArg root}
        '')
        patches;
    })
