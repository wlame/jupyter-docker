#!/usr/bin/env bash
# =============================================================================
# Copy one image's examples (the .py source and its .ipynb) into ~/examples.
#
# Runs at BUILD time at the end of every stage, with the repository's examples/
# bind-mounted and the stage's generated targets/<target>/examples.txt (one
# example name per line, '#' starts a comment). ~/examples is rebuilt from
# scratch, so a child stage replaces its parent's set with its own list, which
# already includes the parent's examples.
#
# Usage: select_examples.sh <examples-dir> <examples.txt>
# =============================================================================
set -euo pipefail

usage="usage: select_examples.sh <examples-dir> <examples.txt>"
source_dir="${1:?${usage}}"
list="${2:?${usage}}"
dest="${HOME}/examples"

rm -rf "${dest}"
mkdir -p "${dest}"
count=0
while read -r name; do
    case "${name}" in '' | '#'*) continue ;; esac
    cp "${source_dir}/${name}.py" "${source_dir}/${name}.ipynb" "${dest}/"
    count=$((count + 1))
done < "${list}"
echo "✓ ${count} examples in ${dest}"
