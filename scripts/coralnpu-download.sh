#!/usr/bin/env bash
# Fetch, verify and extract one of this host's Coral NPU archives pinned in a
# lock file: the emulators (`make coralnpu.download`) or the full subsystem
# (`make coralnpu.subsystem.download`). Prints the extracted archive root,
# which is what CORALNPU_DATA (or CORALNPU_SUBSYSTEM_DATA) names.
#
# Usage: coralnpu-download.sh <coralnpu.lock> <data-dir> [emulators|subsystem]
set -euo pipefail

lock=$1
data_dir=$2
kind=${3:-emulators}
case $kind in
emulators | subsystem) ;;
*)
	echo "coralnpu.download: unknown archive $kind" >&2
	exit 2
	;;
esac

arch=$(uname -m | sed -e 's/^amd64$/x86_64/' -e 's/^arm64$/aarch64/')
name=coralnpu-$kind-linux-$arch
tag=$(sed -n 's/^tag //p' "$lock")
repo=$(sed -n 's/^repo //p' "$lock")
sum=$(grep "  $name.tar.gz\$" "$lock" || true)
if [ -z "$tag" ] || [ -z "$repo" ] || [ -z "$sum" ]; then
	echo "coralnpu.download: $lock pins no $name.tar.gz" >&2
	exit 1
fi

dest=$data_dir/$tag
mkdir -p "$dest"
if ! test -f "$dest/$name.tar.gz"; then
	curl -fsSL -o "$dest/$name.tar.gz.part" \
		"https://github.com/$repo/releases/download/$tag/$name.tar.gz"
	mv "$dest/$name.tar.gz.part" "$dest/$name.tar.gz"
fi
(cd "$dest" && echo "$sum" | sha256sum -c --quiet -)

# Extract beside the old tree and swap, so an interrupted run leaves no half
# archive under the name the tests read.
rm -rf "$dest/$name.new"
mkdir "$dest/$name.new"
tar -xzf "$dest/$name.tar.gz" -C "$dest/$name.new"
test -f "$dest/$name.new/$name/hardware-configurations.json" || {
	echo "coralnpu.download: $name.tar.gz has no $name/hardware-configurations.json" >&2
	exit 1
}
rm -rf "$dest/$name"
mv "$dest/$name.new/$name" "$dest/$name"
rmdir "$dest/$name.new"
echo "$dest/$name"
