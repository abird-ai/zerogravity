# CPU quality never selects the GPU/qualification workspace members or `all`.
{lib}: let
  cpuPackages = [
    "zerogravity-core"
    "zerogravity-artifacts"
    "zerogravity-cpu"
    "zerogravity-model-qwen35"
    "zerogravity-model-gemma4"
    "zerogravity"
    "zerogravity-cli"
  ];
  cpuFeatures = ["cpu" "model-qwen35" "text"];
  diagnosticFeatures = cpuFeatures ++ ["diagnostic-cpu-checkpoint" "diagnostic-gdn-trace"];
  qualityFeatures = (map (name: "zerogravity-cli/${name}") diagnosticFeatures) ++ ["zerogravity/model-gemma4" "zerogravity-artifacts/diagnostic-cache"];
in {
  inherit cpuPackages;
  cliArgs = "--locked --offline --no-default-features -p zerogravity-cli --bin zerogravity --features ${lib.concatStringsSep "," cpuFeatures}";
  qualityArgs = "--locked --offline --no-default-features ${lib.concatMapStringsSep " " (name: "-p ${name}") cpuPackages} --features ${lib.concatStringsSep "," qualityFeatures}";
}
