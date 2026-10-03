#!/usr/bin/env python3
"""Unpacks a wasm32 C library and compiler runtime into a directory, without
root: the Debian packages `wasi-libc` (headers, libc, libm) and
`libclang-rt-<N>-dev-wasm32` (the compiler's builtins), for the installed
Clang's major version. The installed Clang and `wasm-ld` already target wasm32;
what they lack is a libc, which is why stock generated C fails at `math.h`.

usage: wasi-sysroot-userland.py <dest-dir> [suite]
Prints the sysroot and the resource directory to pass to Clang:
  SYSROOT=<dest>/root/usr   (use --sysroot=$SYSROOT/.. -- see the Makefile)
"""
import lzma, os, re, subprocess, sys, urllib.request

dest = sys.argv[1]
suite = sys.argv[2] if len(sys.argv) > 2 else "trixie"
mirror = "http://deb.debian.org/debian/"
clang = subprocess.check_output(["clang", "--version"], text=True)
major = re.search(r"clang version (\d+)", clang).group(1)
want = ["wasi-libc", f"libclang-rt-{major}-dev-wasm32"]

index = None
for arch in ("all", "arm64", "amd64"):
    pass
def packages(arch):
    url = f"{mirror}dists/{suite}/main/binary-{arch}/Packages.xz"
    text = lzma.decompress(urllib.request.urlopen(url).read()).decode()
    out = {}
    for block in text.split("\n\n"):
        d = {}
        for line in block.split("\n"):
            if not line.startswith(" ") and ":" in line:
                k, v = line.split(":", 1)
                d[k] = v.strip()
        if "Package" in d:
            out[d["Package"]] = d
    return out
host = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
tables = [packages(host)]
try:
    tables.append(packages("all"))
except Exception:
    pass
os.makedirs(os.path.join(dest, "debs"), exist_ok=True)
root = os.path.join(dest, "root")
os.makedirs(root, exist_ok=True)
for name in want:
    d = next((t[name] for t in tables if name in t), None)
    if d is None:
        sys.exit(f"{name}: not in the {suite} index")
    out = os.path.join(dest, "debs", os.path.basename(d["Filename"]))
    if not os.path.exists(out):
        urllib.request.urlretrieve(mirror + d["Filename"], out)
    subprocess.check_call(["dpkg-deb", "-x", out, root])
    print("unpacked", name, d["Version"])
print("SYSROOT=" + os.path.join(root, "usr"))
