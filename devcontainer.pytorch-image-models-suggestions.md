# Suggestions for `devcontainer.pytorch-image-models`

These are integration-oriented suggestions from consuming the `v0.0.3`
functional-ATen release in `mltorch`. They are proposals for the producer; they
do not imply that the current release is malformed.

## Publish one signed, machine-readable release index

Publish `manifest.json` alongside release assets, containing the producer
commit, tag, supported model IDs, archive URLs, SHA-256 digests, archive sizes,
and required members. Consumers currently need to maintain those facts in a
separate manifest, which can drift when a model is added, renamed, or removed.

The index should also make model retirement explicit. In particular, a consumer
that previously used an ID such as `resnet18` can select a current default or
offer a useful migration message instead of discovering the removal through a
missing file.

## Describe each runnable payload contract

For every model, include a small metadata record with:

- input and output map keys, dtypes, shapes, and layout/stride expectations;
- the number of shipped samples and lexical sample-key order;
- preprocessing provenance and model class count;
- the `expected.json` ranking contract and numeric comparison tolerances.

`inputs.pt` and `outputs.pt` are excellent self-contained execution fixtures.
Describing their map-level contract eliminates consumer guesses about whether
they hold one tensor, a batch, or multiple named samples.

## Distinguish graph compatibility from runnable compatibility

Generate a compatibility report per release that records at least:

- graph target histogram and configuration variants;
- captured tensor dtypes, including non-float tensors;
- an interpreter execution result for every supplied input;
- a native-lowering/execution result where that backend is part of the release
  promise; and
- the first unsupported target or dtype for each failed backend.

A target list alone is not enough evidence of whole-model execution. For
example, a model may load its graph successfully while reaching a target such
as `aten.stack.default` only during an actual run, or may depend on an `int64`
captured tensor. Classifying these models as `runnable`, `graph_only`, or
`unsupported` would let consumers choose appropriate CI cohorts without
treating every published graph as a runnable claim.

## Make the embedded graph easy to verify

Keep the committed producer `model.json` byte-for-byte canonical with the graph
inside each `.pt2`, or publish a canonical decoded-JSON digest for both. This
makes release-artifact verification cheap and unambiguous for downstream
consumers, while retaining the `.pt2` as the authoritative executable archive.

## Publish a stable model catalogue for browser consumers

Include a catalogue with display name, default model, aliases/deprecations, and
the graph JSON URL for each selected model. A browser app can then build its
picker from the producer's advertised inventory rather than embedding an old
model name as a default.

## Keep model roles explicit

If models are intentionally included to exercise a particular behavior—such as
adaptive pooling to `[1,1]` or `[7,7]`—label that role separately from a full
interpreter or native-execution promise. This preserves valuable edge-case
fixtures without accidentally turning them into broad compatibility claims.

## Scope boundary

Implementing missing ATen operators, native lowering rules, or JavaScript
runtime support is a consumer responsibility. The producer can make those
decisions far easier by publishing complete, reproducible capability evidence
and a stable release catalogue.
