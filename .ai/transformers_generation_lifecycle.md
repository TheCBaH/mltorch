# Bounded Transformers cache chaining

Pure history metadata decodes int64 values and bounds history/capacity at one
million before narrowing on native or JavaScript. Static snapshots require a
state block, unique binding names, nonnegative history, capacity covering the
snapshot and attention length above history. Cache-shape history narrowing is
bounded; chaining compares int64 lengths without narrowing untrusted extents.

`History.chain_tensors` adds exact dtype compatibility to ordered cache-name,
rank, batch, head, head-dimension and history checks. Host-side
`Transformers_metadata.Chaining` compares model identity and the complete
checkpoint/config weight-source object before allowing those tensor checks.
Identical shapes cannot authorize tensors from different weights or configs.

The bounded SmolLM2 route has one prefill and one history-four decode. Consumer
task checks must retain every output/K/V comparison and independently validate
reset, masks, positions, greedy selection, EOS, capacity and stop transitions.
History-five output cannot feed the history-four graph. Shape/dtype compatibility
does not establish numerical or continuous-generation acceptance.
