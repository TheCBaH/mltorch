# Inference comparison coordinator: engine pairs, composition, certification

How real engines meet the coordinator. The coordinator layers
(`inference_compare_{plan,extract,policy,core}`) never see an engine name; an
engine is an ordinary graph runner plus a dialect, and a pair is two such
endpoints, two graph descriptions and one mapping artifact.

## Where a mapping comes from

- **Nothing is matched by name or value.** A mapping is written by the party
  that knows the correspondence: the importer (`Import_map`, ATen to Native),
  a transformation pass (`Pass_map`, from `Map_export`'s plain-data clusters
  of a `Graph_map`), or the identity (two routes over one graph).
- **Pass maps are coarsened, not repaired.** A pass says which nodes became
  which, not where an execution boundary is legal. `Pass_map` makes the
  nodes of a cluster one unit, the producers of one value cluster one unit, a
  value only one side exposes internal (merging its producers and consumers)
  and any strongly connected set of units one unit. The planner still has the
  last word.
- **Composition concatenates, never collapses.** `Compose.compose` joins maps
  through a shared intermediate graph: conversion steps keep their order,
  claims are joined, a direction with no single-source route is unavailable
  (a pack is refused, not guessed). A missing route is a pinned constant only
  when the later map's tensor relates to nothing the first map binds; a route
  from a tensor no unit binds is no route. Regrouping three maps gives the
  same meaning (`pair_cram.t`).

## Engines

- A model outside the Native4D domain is a refusal naming the tensor; there
  is no fall back to Native. The same holds for either compiled route.
- `Graph_dialect`/`Graph_runner` serve any Native-family graph model; an engine
  supplies only compile and execute. Import, export and the input inventory
  are shared.
- The embedded CompCert and the direct assembly routes
  (`lib/inference_compare_compiled`) are optional: they need
  `MLTORCH_COMPCERT` for the host ISA and Rivet's libraries, and a missing
  prerequisite leaves their gate pending. Both end in Rivet's assembler and
  loader, so `Ancestry.shared` names what an agreement between them does not
  vouch for.

## Certification

`inference_compare certify` runs the reference trajectory, then (only after a
complete agreement) the tested trajectory from fresh resources with unchanged
identities, then the tested engine's complete graph (the mapping regrouped into
one unit, `Whole_graph`), then the contract's end-to-end checks, and only then
writes the golden record.

- Each stage is persisted before the next starts; a mismatch, error or
  refusal anywhere ends the workflow and no golden exists. The golden holds the
  tested engine's own bytes.
- Peer-only references, donor overrides, lossy comparison conversions, a
  missing final policy and a required acceptance with no frozen reference are
  refused before anything runs.
- `Exact` requires bitwise local, fidelity and final checks; anything else that
  passes is `Within_declared_contract`.
- A passing local comparison does not bound the end-to-end error: the final
  policies and the frozen reference are separate gates.

The variant words, the registry of engines and the coverage manifest are in the
variants doc; this one covers the coordinator side.

## Running it

Each command below is exercised by a cram in `test/inference_compare_pt2`
(gated on `PT2_DATA`, run by `make pt2.runtest`) or `test/inference_compare/cli`.

```sh
M=data/pt2-functional/test_convnext2/test_convnext2.pt2
I=data/pt2-functional/test_convnext2/inputs.pt
inference_compare export-pt2 --model $M --inputs $I --right native4d --out exp
inference_compare validate --experiment exp/experiment.json
inference_compare run --experiment exp/experiment.json --evidence-root ev --run-id r1
inference_compare replay --run ev/run-r1                      # or --unit UNIT
inference_compare measure --experiment exp/experiment.json --evidence-root ev2

# two-pass certification against the producer's own outputs
inference_compare export-pt2 --model $M --inputs $I --certify declared \
  --reference-outputs data/pt2-functional/test_convnext2/outputs.pt --out cert
inference_compare certify --workflow cert/validation.json --evidence-root ev3

# embedded CompCert against direct assembly (aarch64 hosts with the embedding)
make compcert.compare.runtest
```

Exit codes: 0 agreement or certified, 1 numerical disagreement, 2 invalid
input, map, policy or contract, 3 refused or failed stage, 4 incomplete.

## Known limits

- `csatv2` stops in the ATen to Native importer (not at Native4D).
- The compiled routes are verified on aarch64 only; the x86_64 embedding is
  present but this host cannot run it.
- Multi-source (pack) routes are not composed.
- ATen's segmented run is not bitwise ATen's complete graph on
  `test_convnext2`, so exact partition fidelity is refused for ATen; a declared
  contract certifies.
