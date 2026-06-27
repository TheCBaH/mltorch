# mltorch

OCaml tooling for [PyTorch](https://pytorch.org/) export artifacts, built on [Jsont](https://erratique.ch/software/jsont) and [Yamlt](https://github.com/TheCBaH/ocaml-yamlt).

Generates a typed OCaml decoder from the PyTorch export schema YAML, reads `.pt2` export archives without libtorch, and runs the exported graph end-to-end on real ATen ops.

[![build](https://github.com/TheCBaH/mltorch/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/TheCBaH/mltorch/actions/workflows/build.yml)

## Get started
* [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=TheCBaH/mltorch)
* run
  * `make runtest` build and run the hermetic test suite (no model downloads)
  * `make pt2.download-cram && make pt2.runtest` download a few real models and run the `.pt2` loader/interpreter tests against them
  * `make inference` download every supported model (~5.3 GB) and run the interpreter on each, printing per-image timing
