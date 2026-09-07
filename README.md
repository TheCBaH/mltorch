# mltorch

OCaml tooling for [PyTorch](https://pytorch.org/) export artifacts, built on [Jsont](https://erratique.ch/software/jsont) and [Yamlt](https://github.com/TheCBaH/ocaml-yamlt).

Generates a typed OCaml decoder from the PyTorch export schema YAML, reads `.pt2` export archives without libtorch, and runs the exported graph end-to-end on real ATen ops.

[![build](https://github.com/TheCBaH/mltorch/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/TheCBaH/mltorch/actions/workflows/build.yml)
[![pages](https://github.com/TheCBaH/mltorch/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/TheCBaH/mltorch/actions/workflows/pages.yml)

## What this is

Given a `.pt2` export archive, mltorch:

1. Decodes `model.json` (the serialized exported graph) into typed OCaml values.
2. Executes supported graphs through either a direct interpreter using C++ ATen bindings or a Native graph IR with pure OCaml evaluators.
3. Rewrites the Native graph through layout conversion, constant folding, and fusion passes, with symbolic verification available to check the transformations. The browser viewer exposes lowering stages and per-operator symbolic computations.

It is not a training framework and does not implement autograd; the target is running and understanding an already-exported inference graph.

## What's different here

Native operators share one per-output-element computation between concrete evaluation and symbolic expression building. Written against an abstract numeric domain, that computation produces either tensor values or expressions containing indices, loads, reductions, and scans. Verification, fusion planning, and visualization all use this symbolic representation.

- **Symbolic rewrite checks.** The verifier compares local computations under corresponding inputs. An exhaustive proof covers all input values for the checked shapes and relation; reports distinguish proofs from numerical agreement, sampled-coordinate coverage, counterexamples, and unproved obligations. Budgets and unsupported cases can leave a rewrite unproved, and the release policy rejects counterexamples without requiring every obligation to be proved.
- **Inspectable computations.** Canonical Native operators expose their symbolic computation in the browser, including reductions, index arithmetic, and coordinate loads. Detail availability depends on successful symbolic evaluation and size limits.
- **A pure OCaml execution path.** Native tensor operations and graph evaluation run without libtorch, and the same code compiles to JavaScript: `make jsoo.pt2.run` runs a real model's inference under node and checks the ranking against the release reference. The browser viewer does not run inference — it decodes, lowers, and verifies. The separate ATen interpreter and parity tests use generated bindings to a locally built C++ subset that includes the required upstream kernels.
- **Generated schema plumbing.** The decoder and ATen bindings are generated from the checked-out PyTorch schema and operator registry. Supporting a new operation can still require kernel sources, dispatch selection, and a Native implementation.

Design notes for each of these live in [`.ai/`](.ai/) — the tracked design record, indexed by topic rather than by a fixed list.

## Try it in a browser

**[MLTorch Model Explorer](https://thecbah.github.io/mltorch/)** — pick a model from the built-in catalogue, or load your own exported `model.json`, and inspect the graph at each stage of lowering.

The viewer's decoding, Native lowering, and verification run client-side in JavaScript: local files are never uploaded. The C++ ATen interpreter is separate from this browser path. Published by [`pages.yml`](.github/workflows/pages.yml) from `main`, from a version tag, and on a published release.

## Get started

* [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=TheCBaH/mltorch)
* run
  * `make runtest` build and run the hermetic test suite (no model downloads)
  * `make pt2.download-cram && make pt2.runtest` download a few real models and run the `.pt2` loader/interpreter tests against them
  * `make inference` download every supported model (~5.3 GB) and run the interpreter on each, printing per-image timing

Model fixtures (`.pt2` exports, images, expected results) are downloaded from release
assets of [`TheCBaH/devcontainer.pytorch-image-models`](https://github.com/TheCBaH/devcontainer.pytorch-image-models),
the companion repo that produces them from [`pytorch-image-models`](https://github.com/huggingface/pytorch-image-models).
