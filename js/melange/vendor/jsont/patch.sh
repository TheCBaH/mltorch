#!/bin/bash
# Copy an upstream jsont source out of the opam switch and apply the
# melange-only rename, refusing to do either if the source is not the exact
# version the patch was produced against.
#
#   patch.sh <src> <sha256> <patch> <dst>
#
# The checksum is the whole point. `%{lib:jsont:...}` resolves to whatever
# version is installed, and this patch renames an exception at line offsets and
# in syntactic positions that only hold for jsont 0.2.0. Applying it blind to a
# later release could rewrite the wrong construct and still compile. A mismatch
# must stop the build and say so, not fuzz its way through -- hence --fuzz=0 as
# well.
set -euo pipefail

src=$1
expected=$2
patch_file=$3
dst=$4

actual=$(sha256sum "$src" | cut -d' ' -f1)
if [ "$actual" != "$expected" ]; then
  cat >&2 <<EOF
jsont melange patch: refusing to patch an unexpected source.

  file:     $src
  expected: $expected
  actual:   $actual

The devcontainer pins jsont#0.2.0; if that pin moved, regenerate
$patch_file against the new source (rename only the exception -- NOT the
module and NOT the result constructor; let the compiler find the exception
positions) and update the checksum in this directory's dune file.
EOF
  exit 1
fi

cp "$src" "$dst"
chmod +w "$dst"
patch --silent --fuzz=0 "$dst" < "$patch_file"
