#!/usr/bin/env bash
# =============================================================================
# Copy one image's examples (the .py source and its .ipynb) into ~/examples,
# with the target's generated examples.md as ~/examples/README.md.
#
# Runs at BUILD time at the end of every stage, with the repository's examples/
# and the stage's generated targets/<target>/ folder bind-mounted. examples.txt
# there lists one example name per line ('#' starts a comment). ~/examples is
# rebuilt from scratch, so a child stage replaces its parent's set with its own
# list, which already includes the parent's examples.
#
# Usage: select_examples.sh <examples-dir> <target-dir>
# =============================================================================
set -euo pipefail

usage="usage: select_examples.sh <examples-dir> <target-dir>"
source_dir="${1:?${usage}}"
target_dir="${2:?${usage}}"
dest="${HOME}/examples"

rm -rf "${dest}"
mkdir -p "${dest}"
count=0
while read -r name; do
    case "${name}" in '' | '#'*) continue ;; esac
    cp "${source_dir}/${name}.py" "${source_dir}/${name}.ipynb" "${dest}/"
    count=$((count + 1))
done < "${target_dir}/examples.txt"
cp "${target_dir}/examples.md" "${dest}/README.md"
echo "✓ ${count} examples in ${dest}"
