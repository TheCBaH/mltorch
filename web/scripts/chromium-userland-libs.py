#!/usr/bin/env python3
"""Unpacks the system libraries playwright's Chromium needs into a directory,
without root: for a machine where `playwright install --with-deps` cannot run
(no sudo, no apt lists). Resolves each package's dependencies from the Debian
`Packages` index, downloads only what `dpkg -l` does not already list, and
extracts the .debs with `dpkg-deb -x`. Prints the LD_LIBRARY_PATH to use.

usage: chromium-userland-libs.py <dest-dir> [suite] [arch]
       (defaults: Debian trixie, the host's dpkg architecture)

Continuous integration, which can use root, runs `npm run install:chromium`
instead; this exists so the browser check is reproducible in a container
without it.
"""
import os, re, subprocess, sys, urllib.request, lzma

dest = sys.argv[1]
suite = sys.argv[2] if len(sys.argv) > 2 else "trixie"
arch = sys.argv[3] if len(sys.argv) > 3 else subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
mirror = "http://deb.debian.org/debian/"
want = ["libglib2.0-0t64", "libnspr4", "libnss3", "libdbus-1-3", "libatk1.0-0t64",
        "libatk-bridge2.0-0t64", "libatspi2.0-0t64", "libx11-6", "libxcomposite1",
        "libxdamage1", "libxext6", "libxfixes3", "libxrandr2", "libgbm1", "libxcb1",
        "libxkbcommon0", "libasound2t64", "libcups2t64", "libpango-1.0-0", "libcairo2"]

os.makedirs(os.path.join(dest, "debs"), exist_ok=True)
index = lzma.decompress(urllib.request.urlopen(f"{mirror}dists/{suite}/main/binary-{arch}/Packages.xz").read()).decode()
pk = {}
for block in index.split("\n\n"):
    d = {}
    for line in block.split("\n"):
        if not line.startswith(" ") and ":" in line:
            k, v = line.split(":", 1)
            d[k] = v.strip()
    if "Package" in d:
        pk[d["Package"]] = d
        for prov in d.get("Provides", "").split(","):
            name = prov.strip().split(" ")[0]
            if name:
                pk.setdefault("provided:" + name, d)
installed = {l.split()[1].split(":")[0] for l in subprocess.run(["dpkg", "-l"], capture_output=True, text=True).stdout.splitlines() if l.startswith("ii")}

seen, order = set(), []
def visit(n):
    n = n.split(":")[0] if not n.startswith("provided:") else n
    if n in seen:
        return
    seen.add(n)
    d = pk.get(n) or pk.get("provided:" + n)
    if not d or d["Package"] in installed:
        return
    for field in ("Pre-Depends", "Depends"):
        for alt in d.get(field, "").split(","):
            m = re.split(r"[ (]", alt.split("|")[0].strip())[0]
            if m:
                visit(m)
    order.append(d)
for w in want:
    visit(w)
root = os.path.join(dest, "root")
os.makedirs(root, exist_ok=True)
for d in order:
    out = os.path.join(dest, "debs", os.path.basename(d["Filename"]))
    if not os.path.exists(out):
        urllib.request.urlretrieve(mirror + d["Filename"], out)
    subprocess.check_call(["dpkg-deb", "-x", out, root])
libdirs = [os.path.join(root, "usr", "lib", t) for t in os.listdir(os.path.join(root, "usr", "lib")) if "-linux-" in t]
libdirs += [os.path.join(root, "lib", t) for t in os.listdir(os.path.join(root, "lib")) if "-linux-" in t] if os.path.isdir(os.path.join(root, "lib")) else []
print("LD_LIBRARY_PATH=" + ":".join(libdirs))
