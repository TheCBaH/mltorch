# mltorch

OCaml tooling for [PyTorch](https://pytorch.org/) export artifacts, built on [Jsont](https://erratique.ch/software/jsont) and [Yamlt](https://github.com/TheCBaH/ocaml-yamlt).

Generates a typed OCaml decoder from the PyTorch export schema YAML, reads `.pt2` export archives without libtorch, and runs the exported graph end-to-end on real ATen ops.

[![build](https://github.com/TheCBaH/mltorch/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/TheCBaH/mltorch/actions/workflows/build.yml)
[![pages](https://github.com/TheCBaH/mltorch/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/TheCBaH/mltorch/actions/workflows/pages.yml)

## Try it in a browser

**[MLTorch Model Explorer](https://thecbah.github.io/mltorch/)** — pick a model from the built-in catalogue, or load your own exported `model.json`, and inspect the graph at each stage of lowering.

The whole pipeline is compiled to JavaScript and runs on your device: local files are never uploaded. Published by [`pages.yml`](.github/workflows/pages.yml) from `main`, from a version tag, and on a published release.

## Get started
* [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=TheCBaH/mltorch)
* run
  * `make runtest` build and run the hermetic test suite (no model downloads)
  * `make pt2.download-cram && make pt2.runtest` download a few real models and run the `.pt2` loader/interpreter tests against them
  * `make inference` download every supported model (~5.3 GB) and run the interpreter on each, printing per-image timing
