# Op config — no acronyms, no bare ints, shaped window config

Conventions for the `params` record an op functor takes (`Conv2d.params` today,
pool params too). Applied to `lib/native/ops/conv.ml`; the same guarded scalar
types are expected for new ops.

## Problem

`Conv2d.params` (`ops/conv.ml`, and the older sketch in `native_compute_design.md`
§2) is:

```ocaml
type params = { kh:int; kw:int; cin:int; sh:int; sw:int; ph:int; pw:int }
```

Three issues: (1) every field but `cin` is a compressed acronym; (2) all seven
are bare `int`, so `-1` stride or a negative pad type-checks; (3) `kh/kw`,
`sh/sw`, `ph/pw` are the same shape — a height/width pair — repeated three
times instead of named once.

## Convention 1 — no acronyms

Spell field names out in full: `kernel`, `in_channels`, `stride`, `pad`. A
single-letter axis name (`h`, `w`, matching `Axis.H`/`Axis.W`) is not an
acronym and stays short; a *compression of multiple words* (`kh`, `sw`, `cin`)
is, and gets expanded.

## Convention 2 — shared H/W pair or per-axis windows

Kernel size, stride, and padding are all "one value for H, one for W." Name
that shape once instead of repeating two fields per concept:

```ocaml
module Hw : sig
  type 'a t = { h : 'a; w : 'a }
end
```

Scoped to H/W on purpose — it's the shape conv/pool params actually have, not
a general N-axis vector (that's already `Vec6`, which is a different role:
tensor shapes/coords over all six axes, not a pair of op hyperparameters).
Revisit if a future op needs a third paired axis (e.g. conv3d's `D`).

For Conv2d, H and W now each carry a full axis window because TensorFlow/PyTorch
semantics need asymmetric padding and dilation per axis:

```ocaml
type axis_window = {
  kernel : Dim.extent Dim.t;
  stride : Op_config.Pos.t;
  pad_before : Op_config.Nonneg.t;
  pad_after : Op_config.Nonneg.t;
  dilation : Op_config.Pos.t;
}
```

## Convention 3 — non-negative by type, not by convention

`Dim` (`native_tensor_design.md` §1a) already guards tensor-shape quantities
(`extent`, `index`, `count`, `offset`) ≥ 0 via a phantom-tagged `private int`.
Two of the seven `params` fields *are* extents and should just reuse it:

- `kernel : Dim.extent Dim.t Hw.t` — a kernel size is literally the weight
  tensor's extent along `H`/`W`.
- `in_channels : Dim.extent Dim.t` — literally the input's extent along `C`.

`stride`, `pad_before`/`pad_after`, `dilation`, and `groups` are not
tensor-axis sizes — they're op hyperparameters with their own constraints, and
folding them into `Dim` would blur a module whose documented job is tensor
indexing math. They get their own guarded types, following the same `private
int` pattern as `Dim` — but they don't share *one* constraint:

- `pad` of 0 is a normal, common value (no padding) — `≥ 0` is the right
  guard.
- `stride`, `dilation`, and `groups` of 0 are never valid, so they need the
  stricter `Pos` type, not the same one as `pad`.

So two minimal types, not one — and, along with `Hw`, too trivial individually
to each earn a dedicated file, so all three live as submodules of one
`lib/native/op_config.ml`(`.mli`):

```ocaml
(* lib/native/op_config.ml(i) *)
module Nonneg : sig            (* ≥ 0, zero allowed *)
  type t = private int
  val of_int : int -> t        (* raises [Invalid_argument] if < 0 *)
  val to_int : t -> int        (* also the free coercion (x :> int) *)
end

module Pos : sig                (* ≥ 1, zero rejected *)
  type t = private int
  val of_int : int -> t        (* raises [Invalid_argument] if < 1 *)
  val to_int : t -> int        (* also the free coercion (x :> int) *)
end

module Hw : sig
  type 'a t = { h : 'a; w : 'a }
end
```

These are the "core types module" for op config: any future parameter that
must be ≥ 0 but isn't a `Dim` role (pad, output_padding, …) reuses
`Op_config.Nonneg.t`; anything that must be ≥ 1 (stride, dilation, groups, …)
reuses `Op_config.Pos.t`. Two types rather than one parameterized
"non-negative, optionally-strict" type, since the strictness is a property of
*which quantity* it is, fixed at the call site, not something callers choose
per use. (Callers go through the qualified `Op_config.Nonneg`/`Op_config.Pos`/
`Op_config.Hw` — the `wrapped false` unqualified-access convention is for the
library's top-level modules; submodules of one file are naturally qualified.)

## Result

```ocaml
type params = {
  h : axis_window;
  w : axis_window;
  in_channels : Dim.extent Dim.t;
  groups : Op_config.Pos.t;
}
```

Seven loose, acronym-named, unguarded `int`s become named, typed fields, each
guarded to the constraint it actually has. `MaxPool2d.params` and
`AvgPool2d.params` (`lib/native/ops/pool.ml`) still use the simpler
`kernel`/`stride`/`pad` H/W triples because pooling currently models symmetric,
undilated windows. `Linear.params` (`lib/native/ops/linear.ml`) has no spatial
kernel at all and reduces to `in_features : Dim.extent Dim.t`, reusing the same
"it's a tensor extent, not a bare int" reasoning as `Conv2d.in_channels`.
