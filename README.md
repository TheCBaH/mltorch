# mltorch

OCaml tooling for [PyTorch](https://pytorch.org/) export artifacts, built on [Jsont](https://erratique.ch/software/jsont) and [Yamlt](https://github.com/TheCBaH/ocaml-yamlt).

Generates a typed OCaml decoder from the PyTorch export schema YAML, reads `.pt2` export archives without libtorch, and runs the exported graph end-to-end on real ATen ops.

[![ci](https://github.com/TheCBaH/mltorch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TheCBaH/mltorch/actions/workflows/ci.yml)
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

Transformers graph admission consumes the pinned source submodule:

```sh
git submodule update --init modules/devcontainer.transformers
make transformers.admission
make transformers.admission.normalized # explicit empty-cache normalization
```

Only the top-level producer checkout is needed for its JSON catalogue. Admission
is offline, verifies the checkout pin and every catalogue file, and needs no
Python or ML packages after the usual OCaml build setup. Reports are written to
`_build/transformers-admission/`; set `TRANSFORMERS_ADMISSION_OUT` to retain them
elsewhere. `TRANSFORMERS_SOURCE` can select an investigation checkout, subject to
the same pin and integrity checks. These tiny random graphs are separate from
the released checkpoint cohort. A source pin update must include an inventory
diff and regenerated admission evidence; release and checkpoint pins are
independent.

Regenerate or check the selected release metadata and bounded adapter manifests
with OCaml:

```sh
make transformers.cohort.fetch transformers.assets.fetch # explicit acquisition
make transformers.cohort.check transformers.assets.check # offline verification
```

The selections in `data/transformers/` pin the publication and list artifact and
adapter IDs. These commands check cached bundle members and actual processor,
tokenizer and config bytes without executing producer code or downloading model
weights. `.generate` variants regenerate the derived files offline. Metadata
compatibility does not establish numerical or task acceptance.

Replay the selected published cases and verify retained runs offline:

```sh
make transformers.replay TRANSFORMERS_ARTIFACTS='ARTIFACT_ID'
make transformers.matrix
```

Each replay creates a new directory under `TRANSFORMERS_REPORTS` containing a
content-based run manifest, schema-3 reports and a completion record. The
manifest pins the consumer workspace, producer inventory, release members,
reference environment and effective execution policy. Acquisition errors are
retained too. `TRANSFORMERS_DOTS=binary32-sequential` and
`TRANSFORMERS_CASTS=saturating` select separate opt-in rows.

The OCaml matrix checks immutable report digests and complete cases/outputs
against verified cached contracts. It retains failures and conflicts, excludes
legacy or different-workspace runs as historical, and shows all four policies,
including “not run.” JSON and Markdown go to `TRANSFORMERS_MATRIX_JSON` and
`TRANSFORMERS_MATRIX_MARKDOWN`. It exits nonzero for missing, failing, conflicting
or invalid coverage. A complete matrix is separate from task acceptance and
does not imply transformed, Kernel or streaming execution.

Consume pinned producer task fixtures without Python ML packages:

```sh
make transformers.tasks.fetch # explicit acquisition of six recipe bundles
make transformers.tasks.adapters # offline token/pixel comparisons
make transformers.tasks.models # model outputs after matching inputs
make transformers.tasks.generation # bounded SmolLM2 chain, two independent prompts
make transformers.consumer.check # restricted PATH without Python, after build setup
```

Task runs retain fixture pins, every case outcome and workspace/executable identity
under `TRANSFORMERS_TASK_REPORTS`. The supported adapter boundaries are ASCII BERT
WordPiece and the Pillow TinyCLIP/MobileViT PPM recipes. Unicode BERT, torchvision
recipes, raw-text generation and additional host/tower boundaries remain explicit
refusals or deferred coverage. Generation consumes published prompt IDs, checks
all logits and K/V transitions, and stops after the single history-4 decode step.
No task is promoted by matching token IDs or fixture integrity alone.

`make transformers.sweep` measures all 22 checkpoint components and declared
opt-in numerical policies separately, then verifies the matrix. Numerical
failures keep a failing exit status. CI offers this full sweep and the T5
prefill normalized/saturating-cast gate through manual workflow inputs; local
results do not establish hosted CI success.
