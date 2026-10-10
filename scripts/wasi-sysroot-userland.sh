#!/bin/sh
# Fetch Debian WASI userland packages over HTTPS, verify size/SHA-256 from the
# package index, and unpack without root. Upstream Clang supplies the compiler.
set -eu
[ "$#" -ge 1 ] && [ "$#" -le 2 ] || { echo "usage: $0 DEST [SUITE]" >&2; exit 2; }
dest=$1
suite=${2:-trixie}
case "$suite" in *[!a-zA-Z0-9_-]*|'') echo 'invalid Debian suite' >&2; exit 2 ;; esac
mirror=https://deb.debian.org/debian
major=$(clang --version | sed -n 's/.*clang version \([0-9][0-9]*\).*/\1/p' | head -n 1)
[ -n "$major" ] || { echo 'cannot identify Clang version' >&2; exit 2; }
host=$(dpkg --print-architecture)
case "$host" in *[!a-zA-Z0-9_-]*|'') echo 'invalid Debian architecture' >&2; exit 2 ;; esac
mkdir -p "$dest/debs" "$dest/root"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
curl --fail --location --proto '=https' --proto-redir '=https' \
  "$mirror/dists/$suite/main/binary-$host/Packages.xz" -o "$scratch/Packages.xz"
xz -dc "$scratch/Packages.xz" > "$scratch/Packages"
for package in wasi-libc "libclang-rt-$major-dev-wasm32"; do
  awk -v wanted="$package" '
    BEGIN { RS=""; FS="\n" }
    { delete fields; for (i=1; i<=NF; i++) {
        colon=index($i, ": "); if (colon>0 && substr($i,1,1)!=" ")
          fields[substr($i,1,colon-1)]=substr($i,colon+2)
      }
      if (fields["Package"]==wanted) {
        print fields["Filename"]; print fields["Size"];
        print fields["SHA256"]; print fields["Version"]; exit
      }
    }' "$scratch/Packages" > "$scratch/metadata"
  { IFS= read -r filename && IFS= read -r size && IFS= read -r sha && IFS= read -r version; } < "$scratch/metadata" || {
    echo "$package: missing complete metadata in $suite index" >&2; exit 2;
  }
  case "$filename" in pool/*) ;; *) echo 'invalid package filename' >&2; exit 2 ;; esac
  case "$filename" in *..*|*[!a-zA-Z0-9_./+~-]*) echo 'invalid package path' >&2; exit 2 ;; esac
  case "$size" in ''|*[!0-9]*) echo 'invalid package size' >&2; exit 2 ;; esac
  case "$sha" in *[!0-9a-f]*|'') echo 'invalid package digest' >&2; exit 2 ;; esac
  [ "${#sha}" -eq 64 ] || { echo 'invalid package digest length' >&2; exit 2; }
  output=$dest/debs/${filename##*/}
  if [ ! -f "$output" ]; then
    curl --fail --location --proto '=https' --proto-redir '=https' \
      "$mirror/$filename" -o "$scratch/package.deb"
    candidate=$scratch/package.deb
  else candidate=$output; fi
  [ "$(wc -c < "$candidate" | tr -d ' ')" = "$size" ] || { echo "$package: size mismatch" >&2; exit 2; }
  [ "$(sha256sum "$candidate" | cut -d ' ' -f 1)" = "$sha" ] || { echo "$package: digest mismatch" >&2; exit 2; }
  [ "$candidate" = "$output" ] || mv "$candidate" "$output"
  dpkg-deb -x "$output" "$dest/root"
  echo "unpacked $package $version"
done
echo "SYSROOT=$dest/root/usr"
