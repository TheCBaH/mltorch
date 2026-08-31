# Region computation ownership: follow-up design question

## Question

Should an operation's Region program be part of that operation's computation
definition, rather than a separately named operation-adjacent module selected
by a generic `Regionizer`?

The current implementation deliberately keeps scalar execution and Region
execution separate, but the desired ownership is:

```text
RmsNorm owns:
  - its scalar Pixel computation;
  - its Region-program declaration.

The common Region runtime owns:
  - canonical region-key traversal;
  - traversal of every output owned by a key;
  - local evaluation, result conversion, and stores.
```

## Current call chain

`Norm.RmsNorm.Compute(S).pixel` is the authoritative scalar definition. It is
called through `Eval_op.Make(S).pixel`:

```text
Graph_ir.Rms_norm
  -> Eval_op.Make(Direct).pixel    -> scalar Direct evaluation
  -> Eval_op.Make(Symbolic).pixel  -> Pixel Expr.Value
                                      -> optional Region candidate
```

At the Symbolic boundary, `Eval_symbolic.run_regionized` retains the typed
operation, parameters, operands, output ordinal, shape, Pixel expression, and
synthetic optional operands just long enough for `Regionizer.try_regionize` to
dispatch. The dispatcher currently calls:

```text
Norm.RmsNorm.Region.try_regionize
Norm.LayerNorm.Region.try_regionize
Reduce.Softmax.Region.try_regionize
```

Those operation-owned constructors return a `Region_program.t`; they do not
execute it. `Kernel_adapt.Region_admission` checks limits and reconstruction
against the original Pixel expression before accepting it.

## Why scanning is not currently operation-specific

The Region program already declares the semantic scan domain:

```text
partition: Whole params.dims, Singleton on every other axis
locals:    evaluate once per region key
output:    evaluate once per coordinate owned by that key
```

`Region_execution.materialize` interprets that declaration generically:

```text
for key in Region_partition.fold_keys partition output_shape:
  values = evaluate_locals key
  for output in Region_partition.fold_outputs partition output_shape key:
    store output = emit values output
```

This scan is shared because it defines the Region language's canonical `Vec6`
traversal and must be identical for RMSNorm, LayerNorm, Softmax, and future
Region programs. Operation-specific loops would duplicate traversal, stores,
result conversion, and future tiling/vectorization policy.

An operation can and should describe *what a region is* through its partition
and locals. It should not describe *how the common Region representation is
physically scanned* unless the project deliberately adds a separate scheduling
interface (for example `Region_schedule` or Loop IR). That is outside the
scalar-Region slice.

## Possible API direction

The present shape is:

```ocaml
module RmsNorm = struct
  module Compute (S : Semantics.SEMANTICS) = struct
    val pixel : ...
  end

  module Region = struct
    val try_regionize : Region_context.t -> params -> ...
  end
end
```

This keeps `Region` outside `Compute(S)` because it constructs Native
`Expr.Value` and `Region_program.t`; it does not run over the abstract scalar
domain `S`.

If making Region construction visibly part of an operation's computation is a
better conceptual model, an API such as the following is worth considering:

```ocaml
module RmsNorm = struct
  module Compute = struct
    module Pixel (S : Semantics.SEMANTICS) = struct
      val run : ...
    end

    module Region = struct
      val try_program : Region_context.t -> params -> ...
    end
  end
end
```

That makes both forms explicitly `RmsNorm` computation while preserving a
single generic Region executor. It is an API/layout refactor, not a reason to
add a Region arm to `Semantics.SEMANTICS` or to duplicate the traversal inside
each operation.

## Decision deferred

Before changing the public operation-module structure, decide whether
`Compute` should mean only a polymorphic scalar computation or should be the
umbrella for Pixel and Region computation descriptions. The current runtime
and correctness contract work with either naming/layout choice.
