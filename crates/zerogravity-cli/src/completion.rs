//! Static completion generation from the same canonical command tree. No
//! filesystem/model/network queries at completion time.
use crate::args::{Shell, TREE};

const GLOBALS: &str = "--help --version --quiet --verbose --format --color --progress --interactive --offline --yes --config --profile";

pub(super) fn write(shell: Shell) {
    let commands = TREE.iter().map(|n| n.name).collect::<Vec<_>>().join(" ");
    match shell {
        Shell::Bash => {
            println!("# zerogravity / zg completion\n_zerogravity() {{");
            println!(
                "    local cur=\"${{COMP_WORDS[COMP_CWORD]}}\" choices='{commands} {GLOBALS}' word"
            );
            println!(
                "    local i\n    for ((i=1; i<COMP_CWORD; i++)); do\n        word=\"${{COMP_WORDS[i]}}\"\n        case \"$word\" in"
            );
            println!(
                "            --format|--color|--progress|--interactive|--config|--profile) ((i+=1)); continue ;;"
            );
            for node in TREE {
                if !node.children.is_empty() {
                    println!(
                        "            {}) choices='{} {GLOBALS}'; break ;;",
                        node.name,
                        node.children.join(" ")
                    );
                } else {
                    println!("            {}) choices='{GLOBALS}'; break ;;", node.name);
                }
            }
            println!(
                "        esac\n    done\n    COMPREPLY=( $(compgen -W \"$choices\" -- \"$cur\") )\n}}\ncomplete -F _zerogravity zerogravity zg"
            );
        }
        Shell::Zsh => {
            println!(
                "#compdef zerogravity zg\n_zerogravity() {{\n    local word choices='{commands} {GLOBALS}'\n    local -i i\n    for ((i=2; i<CURRENT; i++)); do\n        word=\"$words[i]\"\n        case \"$word\" in"
            );
            println!(
                "            --format|--color|--progress|--interactive|--config|--profile) ((i+=1)); continue ;;"
            );
            for node in TREE {
                let children = node.children.join(" ");
                println!(
                    "            {}) choices='{children} {GLOBALS}'; break ;;",
                    node.name
                );
            }
            println!(
                "        esac\n    done\n    _values 'zerogravity' ${{=choices}}\n}}\n_zerogravity \"$@\""
            );
        }
        Shell::Fish => {
            for binary in ["zerogravity", "zg"] {
                for node in TREE {
                    println!(
                        "complete -c {binary} -f -n '__fish_use_subcommand' -a '{}' -d '{}'",
                        node.name, node.description
                    );
                    if !node.children.is_empty() {
                        println!(
                            "complete -c {binary} -f -n '__fish_seen_subcommand_from {}' -a '{}'",
                            node.name,
                            node.children.join(" ")
                        );
                    }
                }
                for flag in GLOBALS.split_whitespace() {
                    println!("complete -c {binary} -l {}", &flag[2..]);
                }
            }
        }
        Shell::Pwsh => {
            println!(
                "Register-ArgumentCompleter -Native -CommandName zerogravity,zg -ScriptBlock {{\n    param($wordToComplete, $commandAst, $cursorPosition)\n    $choices = '{commands} {GLOBALS}'\n    $skip = $false\n    foreach ($element in ($commandAst.CommandElements | Select-Object -Skip 1)) {{\n        if ($skip) {{ $skip = $false; continue }}\n        $word = $element.Extent.Text\n        if ($word -in @('--format','--color','--progress','--interactive','--config','--profile')) {{ $skip = $true; continue }}\n        switch ($word) {{"
            );
            for node in TREE {
                println!(
                    "            '{}' {{ $choices = '{} {GLOBALS}'; break }}",
                    node.name,
                    node.children.join(" ")
                );
            }
            println!(
                "        }}\n    }}\n    $choices -split ' ' | Where-Object {{ $_ -and $_.StartsWith($wordToComplete) }} | ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }}\n}}"
            );
        }
    }
}
