# Lowering inspector — design record

Status: implemented (CLI, browser pane, Machine IR, assembly, Coral, Native4D, overlay
request). Supersedes the `generatedJs` explorer option (see the generated-JS doc in
this directory for the pass-naming mechanism and the dropped ideas).

## Question it answers

For one node of a model: what does it compile to on every backend this repository has,
and how do two variants differ? One producer library serves the CLI
(`native_graph lower`) and the browser (a lazily spawned worker and a code pane).

## Layers

```
lib/lowering_key     vocabulary: dialect, node, variant (family derived from the variant),
                     one spelling `family[:segment[/segment...]]`; default segments elided
lib/lowering_view    invocation -> Artifact: Loop, Js, C, Wat (Loop route), Ssa, C/Js/Wat via
                     SSA, Machine_ir (target x stage), Asm (GNU text), Coral (estimate,
                     transfer plan, RVV/Zvt kernel text); Summary (per-node metrics)
lib/lowering_diff    Myers line diff, hunks, unified text; bounded search, then a correct
                     but unminimised script (`minimal = false`)
lib/model_explorer_export   protocol: Build_lowering, Build_lowering_summary, Me_artifact,
                     `max_artifact_bytes`, `Me_export.canonical_bundle ~dialect`
js/webapp/lowering_worker.ml   the second worker; one-entry memo of the canonical bundle
web/app/code_pane.js           the pane; names come from the bridge's vocabulary
bin/native_graph_lower.ml      the CLI
```

`model_explorer_export` must not depend on `lowering_view`: the session worker would
gain SSA, Machine IR and Rivet. The handler joining them lives in the lowering worker
and the CLI (`Me_export.handle ?lower ?summary ?bundle`).

## Decisions that are easy to get wrong

- **The unit is the bundle invocation**, one per canonical Native node. A Region-authored
  node lowers through a different path than a plain node, so per-node text must come from
  the bundle, not from `Loop_node_program.lower` alone. Max pool with indices is two
  invocations (one per output); only a grouped Region node (Lstm) has several outputs in
  one invocation.
- **Defaults are what the drivers run**: C and Wasm Loop route at `simd_fp32_relaxed`
  (neon128 / wasm128); JS is binary64 scalar and refuses an fp32 policy; SSA is opt-in.
  Machine IR uses the scalar fp32 plan every target accepts.
- **The key carries the node, so the family is derived from the variant**; a family with
  another family's axes is unrepresentable.
- **Granting**: the canonical bundle grants nothing, so `granted` is `fun _ -> false` in
  both consumers; a node needing a grant fails the bundle build.
- **Every family emits or returns a typed refusal**; payloads are data (`{kind; data}`
  on the wire), rendered to sentences only at the edge (the pane).
- **Native4D** offers only the families with Native4D gates (C, JS, Loop, SSA, WAT);
  the others return `Unsupported_dialect`. The table is `Lowering_key.reaches`.
- **Rebuild cost dominates**: source -> canonical graph is ~2 s in node for
  mobilenetv2_050 (bundle build 0.3 s), so the worker keeps a one-entry memo keyed by
  source digest, stages, fold, limits and dialect (memoised requests 17-32 ms).
- **Overlay metrics** cost differs by an order of magnitude in node (mobilenetv2_050:
  scalar_loops 0.5 s, vector_loops 2.9 s, coral_cycles 28 s); only the requested metric
  is computed and progress is emitted every 16 invocations.

## Fidelity and proofs

- The inspector's text equals the whole-model builders' kernels for every invocation of
  mobilenetv2_050 in C, JS and WAT through the Loop route and the SSA planned route;
  mutating the invocation choice turns all cases red. SSA Exact/Representation are not
  covered.
- Printed x86-64 and AArch64 assembly digest equals the manifest `code_digest`; RV32
  text reassembles through Rivet's text route to identical `.text` bytes (GNU `as` was
  not available, so this compares Rivet text against Rivet typed).
- Pass-subset, stage, dialect, metric and refusal behaviour each have a mutation that
  turns a test red.

## JS-backend consequences

The lowering closure (SSA, Machine IR, Rivet module builders, Coral libraries except
`coralnpu_rv32`, which links `unix`) runs under js_of_ocaml. Defects found on the way:
`load_imm`'s `value > 0xFFFF_FFFF` test and two `land 0xFFFF_FFFF` masks in the RV32
kernel DSL, and a `0x80000000` literal in the x86-64 expression table, all truncated at
32 bits. `Loop_programs.overflow_kernel` differs natively and under JS by design (the
SSA index domain is i32; the reference bound is host `int`). The Rivet module builders
were split from their process halves so they link purely. Pure machine suites run under
`(modes best js)`; `machine_ir` (libm), `machine_model` (direct C), `machine_x86_64`
(NaN payloads) and `machine_source` stay native only.

## Publishing

`lowering_view` links Rivet unconditionally, so `webapp.build` installs it first
(`make rivet.install`); Pages and the CI browser job inherit that. The published site is
about 56 MB of debug bundles. Not verified from a branch run: the Pages deploy, passwordless
sudo for the install inside the Pages container, and the submodule fetch of the vendored
Rivet.

## Limits and follow-ups

- The pane is offered on canonical `g/native` nodes only (not operator detail roots or
  Kernel values); numerics are reachable only through the URL spelling.
- Per-pass transformation diffs and a lowering kind in the flow are a separate track.
- A Rivet RISC-V `pp_gnu` would replace `Instruction.pp`; the pin bump it needs
  (err_trace 0.2, `rivet-native-exec`) is its own change series.
