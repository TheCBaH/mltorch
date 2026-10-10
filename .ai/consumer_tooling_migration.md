# Consumer tooling and reference ownership

Consumer model references are acquired from pinned producer publications;
PyTorch computations stay in producer environments. Active Transformers tools
never invoke producer Python, install Python ML packages or recompute references.
Upstream ATen torchgen and its required build packages remain the accepted Python
exception. No claim is made about a Python-free clean build or external vendored
and producer development tools.

## Project setup tools

Inline-test discovery reads tracked Dune files in OCaml. It recognizes comments,
quoted atoms and nested stanzas, defaults inline-test mode to best, retains all
explicit modes, filters producer/vendored directories and filters Wasm/CompCert
conditions on both the library and runner. Malformed input or git discovery
failure stops timing before cleaning/building. The timing shell stores discovery
output first instead of losing a failing process substitution.

WASI userland acquisition uses installed shell/system tools. It fetches Debian
package metadata and packages over HTTPS, validates filename/suite/architecture,
checks package size/SHA-256 from metadata before extraction, and rejects corrupt
cached packages. It unpacks without root and preserves the Makefile sysroot
layout. This is transport-authenticated package metadata; it does not add Debian
release-signature verification or claim an independently pinned toolchain.

## Cram setup and assertions

Fifteen project cram files use test-only OCaml helpers for serialization and JSON
observations. Hand-authored graph witnesses preserve non-default convolution,
normalization, attention, reshape, padding, slicing, activation and unbind
parameters. Their invalid variants preserve negative/zero/symbolic dimensions,
missing metadata, operand/output errors, unsupported options, list cardinality
and backend-domain refusals. The helper prints actual CLI JSON observations;
capabilities, diagnostics, source namespaces, parameters, tensor shapes, edge
slots, generated JS, flow, verification, fusion and graph equality remain the
original assertions, with original expected values.

Payload-backed cases still require explicit PT2_DATA and downloaded artifacts.
The former MobileNetV1 traceback after a resource refusal is replaced with a
partial-file check and a normal-profile conversion/verification assertion. The
normal-profile case admits canonical and Native4D graphs with 115 nodes, removes
batch norms and reports no refutation. Migration verification includes forced
execution of all fifteen crams with Python unavailable after build setup.

No project Python setup/reference files or cram interpreter invocations remain.
Consumer dependency scope continues to exclude invoking vendored/producer Python
through checkout paths. Historical evidence and external Python source are data.
