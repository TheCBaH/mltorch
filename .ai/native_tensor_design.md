# Native tensors — 6D NHWC layout, GADT payloads, quantization

Pillars 1 & 2 of `native_inference_plan.md`. Covers the coordinate model, the
typed payload, and quantization. No ops here — this is the data substrate the
compute functors read and write.

## 1. The fixed 6-axis frame `N T D H W C`

Every tensor is logically **6-dimensional** over a fixed axis order, regardless
of its "real" rank. Lower-rank tensors embed by setting the unused **outermost**
axes to size 1 (a rank-r shape occupies the innermost r axes — §1d gives the
exact rule and its consequences for rank-changing ops). This makes every op a
uniform function of a 6D `coord`, and removes rank-polymorphism from the whole
engine.

| Axis | Meaning |
|---|---|
| `N` | batch |
| `T` | time / sequence (1 for plain images) |
| `D` | depth (1 for 2D data) |
| `H` | height |
| `W` | width |
| `C` | channels |

`C` is the **innermost / fastest-varying** axis: this is channels-last (a.k.a.
NHWC, generalized to 6D as `N T D H W C`). It matches the channels-last `.pt`
sample images the repo already loads, and keeps the per-output-channel vector
contiguous, which is the natural unit for the C-vectorized schedule later.

> Terminology note: the spec said "channels first," but NHWC and the axis order
> both place C **last** → channels-*last*. The whole design assumes C innermost.

### 1a. Scalar dimensional types — `int` is too loose

Plain `int` lets you write `numel t < 0` or pass a flat `offset` where a
cardinality is wanted; both are nonsense the type system should reject. Four
quantities are involved, all `≥ 0`, and they split into a *per-axis* pair and a
*flattened* pair — the flattened pair being the 1-D analogue of `shape`/`coord`:

| | per-axis (one component) | flattened (whole tensor) |
|---|---|---|
| **size / cardinality** | `extent` (axis length) | `count` (numel = ∏ extents) |
| **position** | `index` (`0 ≤ i < extent`) | `offset` (`0 ≤ o < count`) |

So **`numel` returns a `count`, `offset` returns an `offset`, and the two do not
unify** — exactly the distinction you flagged, mirroring shape≠coord one level
down. A fifth type, `delta` (**signed**), is the one thing that legitimately *can*
be negative: the difference of two indices, and the stencil/padding arithmetic
`i·stride + k − pad` that may land out of bounds before it is guarded.

All five share one representation but stay distinct via a phantom-tagged
`private int`, which is the key to keeping this cheap: construction is guarded
(non-negative, role-correct) but reading back out to a raw `int` — the one thing
`Bigarray`/FFI needs — is a *free* `:>` coercion, no unwrap call.

```ocaml
module Dim : sig
  type +'role t = private int          (* distinct per 'role; coercible to int via :> *)
  type extent and index and count and offset and delta   (* phantom role tags *)

  val extent : int -> extent t         (* checked: raises / clamps if < 0 *)
  val index  : int -> index  t         (* checked ≥ 0 *)
  (* `count`/`offset` are *derived*, never built from a raw int by callers:
     they come only out of the product / linearisation below, so they are ≥ 0
     by construction. *)

  val ( *@ ) : count t -> extent t -> count t        (* fold for numel *)
  val lin    : offset t -> extent t -> index t -> offset t  (* o*ext + i, the Horner step *)

  val to_delta : index t -> delta t
  val index_of : extent:extent t -> delta t -> index t option  (* Some iff 0 ≤ d < extent *)
  (* raw int only at the storage boundary, as a coercion: (off :> int) for data.{…} *)
end
```

`delta` answers "what about things that *can* be negative": those — `delta`,
quant `zero_point`, and the stored quantised values themselves — stay signed
(`zero_point` and quanta are plain `int`, not `Dim` types; they are domain data,
not sizes/positions).

### 1b. `shape` and `coord` are distinct types over a shared representation

Both a shape and a coord are "six ints over `N T D H W C`," but they must **not be
interchangeable**: a *shape*'s components are `extent`s; a *coord*'s are `index`es.
Adding two shapes, or using a shape as an index, is nonsense — the types should
forbid it. (Two plain records with the same field labels do *not* achieve this in
OCaml: the later-defined one shadows field access, so they'd silently unify.)

Encode the shared 6-component storage once, tagging the role *and* the component
type with phantoms so access is generic but the roles never unify — and `get`
hands back the right scalar dimensional type (`extent` for a shape, `index` for a
coord):

```ocaml
type axis = N | T | D | H | W | C        (* fixed order; C is innermost *)

module Vec6 : sig
  type ('role, 'comp) t                  (* abstract: shared 6-component record *)
  type shape = (shape_role, Dim.extent Dim.t) t   (* extents, each ≥ 0 *)
  type coord = (coord_role, Dim.index  Dim.t) t   (* indices, 0 ≤ aᵢ < extentᵢ *)

  val shape : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> shape  (* checks ≥ 0 *)
  val coord : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> coord
  val origin : coord                                 (* all zero *)

  val get : ('role, 'comp) t -> axis -> 'comp        (* extent for shape, index for coord *)
  val set : ('role, 'comp) t -> axis -> 'comp -> ('role, 'comp) t

  (* the ONLY operations bridging the two roles — each needs *both*, so the
     arguments can't be swapped, and the *result types* are distinct too: *)
  val numel     : shape -> Dim.count Dim.t           (* ≥ 0 by construction *)
  val offset    : shape -> coord -> Dim.offset Dim.t (* dense NHWC linear index *)
  val in_bounds : shape -> coord -> bool
  val iter      : shape -> (coord -> unit) -> unit
  val fold      : shape -> 'acc -> ('acc -> coord -> 'acc) -> 'acc  (* iter's accumulating sibling *)
end
```

Role-specific algebra lives at the specific type: coord translation/windowing in
`delta` space (stencils), broadcast-compatibility and product on `shape`. There is
deliberately **no** `coord_of_shape`/`shape_of_coord`; the only way one meets the
other is `offset`/`in_bounds`/`iter`/`fold`, exactly where index-within-extent is
meaningful.

`fold` is not yet implemented (design-only, same status as the rest of this
note where flagged) — `iter` covers every current use (`materialize`, `pp`),
but a whole-tensor reduction (e.g. comparing two tensors, or an op that wants
its own accumulator instead of writing through `materialize`) belongs at this
substrate level rather than reimplementing the 6-nested-loop order at each
call site.

### 1c. No stored strides — the layout is fixed and packed

Because the format is **fixed (NHWC) and packed (dense, no gaps)**, the strides
are a pure function of the shape — there is nothing for a tensor to *carry*. The
linear index is computed on demand:

```
offset shape coord =
  ((((c.n * shape.t + c.t) * shape.d + c.d) * shape.h + c.h) * shape.w + c.w) * shape.c + c.c
```

This is why `offset : shape -> coord -> offset` (it *uses* both roles distinctly,
and yields the flattened-position type, not a `count`). There is no `strides` field
and no per-tensor `offset` field — every tensor is its own dense buffer starting at 0. The two things explicit strides would have bought
are handled more naturally elsewhere, consistent with "access is a coordinate fed
to a functor":

- **Broadcast** is **coordinate reduction against the source shape at read time**,
  not a stride-0 trick: reading a source of shape `[1,1,1,1,1,C]` (a per-channel
  bias) with a full output coord maps every axis whose source extent is `1` to
  index `0`. The load primitive does this:

  ```ocaml
  val read_coord : shape -> coord -> coord     (* axis a: if shape.a = 1 then 0 else coord.a *)
  ```

  So broadcasting needs no storage metadata — only the source's own shape, which
  the tensor already has.

- **Slice / permute / reshape are index reindexings, not data with funny strides.**
  In a framework where computation *is* an index→value function, a layout op is a
  remap of the consumer's read indices (expressed in symbolic `index_expr`, see
  `native_symbolic_language.md`), so it fuses away. When a concrete packed tensor
  is genuinely required at a boundary (e.g. a graph output), the reindex is
  materialized into a fresh dense buffer by the iterator. (A `reshape` that
  refactors axes incompatibly with NHWC packing is the one case that *must*
  materialize — flagged, not hidden behind a stride.)

`tensor.ml` therefore provides just `numel`, `offset`, `read_coord`, `iter`, and a
`materialize` that runs the iterator to produce a packed buffer — no stride
algebra at all.

### 1d. Mapping ATen ranks to the 6-axis frame — **right-aligned, positional**

ATen/PyTorch shapes are *rank-bearing positional* lists (`[6, 7, 8]` is rank 3;
the names "H/W/C" are ours, not ATen's). The 6-axis frame is *fixed and
nameless-to-ATen*. The bridge between them (`aten_shape.ml`) is **purely
positional and right-aligned**:

> A rank-r ATen shape occupies the **innermost r axes** of `N T D H W C`; the
> outer `6 − r` axes are size 1. Equivalently, ATen dim `i` (`0 ≤ i < r`) maps to
> frame position `6 − r + i`.

```
of_aten [a;b;c]    = N1 T1 D1 H=a W=b C=c          (* rank 3 → innermost 3 *)
of_aten [n;h;w;c]  = N1 T1 D=n H=h W=w C=c          (* rank 4 → innermost 4 *)
to_aten ~rank:3 s  = [s.H; s.W; s.C]                (* reverse needs the rank *)
axis_of_dim ~rank:3 1 = W                            (* dim 1 of a rank-3 tensor *)
```

The reverse (`to_aten`) **takes the rank** because the frame alone cannot recover
it: a frame shape with `N=1` is ambiguous between "rank < 6, N unused" and "rank
6, batch happens to be 1." Rank is metadata the graph must carry alongside the
shape; the round-trip law is `to_aten ~rank:(length a) (of_aten a) = a`.

This is mechanical, not semantic. In particular it does **not** put batch on `N`
or perform NCHW↔NHWC: a rank-4 NHWC activation `[n;h;w;c]` lands with `n` on the
`D` axis (`N` stays 1), because `D H W C` are the innermost four. Placing batch on
`N`, or remapping channel-first inputs, is a *separate* semantic concern (a
load-time permutation), deliberately outside this positional bridge. The bridge's
only job is to be the invertible rank carrier.

#### Rank-changing reductions (`mean.dim` / `sum` / `amax`) and `keepdim`

A reduction over a set of ATen dims either keeps those dims at size 1
(`keepdim=true`, rank preserved) or removes them (`keepdim=false`, rank drops).
Under right-aligned embedding these are **genuinely different 6-axis shapes** — the
earlier assumption that `keepdim` is a no-op in this engine was wrong, and is
corrected here. With `keepdim=false` the surviving dims **re-pack toward the
inside**, so the named axes carrying the data shift inward:

```
input [H6 W7 C8]  (rank 3),  mean over W:
  keepdim=true   → ATen [6,1,8] → H6 W1 C8     (W collapses in place)
  keepdim=false  → ATen [6,8]   → H1 W6 C8     (W removed; H's data now on W)
```

So the value computation is layout-aware: with `keepdim=false` an op's output
coordinate addresses the *re-packed* axes, and `pixel` maps each output axis back
to the input axis whose data it now carries (the surviving-axis permutation), not
the same-named axis. `keepdim=true` is the identity permutation and recovers the
simple "collapse in place" form.

Crucially this needs **only the reduced axes and `keepdim` — not the rank**, so a
reduction stays a rank-free function of named axes (§1's premise). The reason: the
permutation is "drop the reduced axes, re-pack every survivor right-aligned." Under
this embedding the axes outside the data are already size 1, so re-packing *all*
survivors — real axes and size-1 padding axes alike — merely shuffles those 1s and
lands the real data on exactly the right-aligned axes. There is no need to know
where the real data begins. Concretely, for `mean` over `W` the survivors `N T D H
C` re-pack into the innermost five `T D H W C` (`H→W`, `C→C`, outer `N→1`),
whatever the true rank. Recovering the ATen output *rank* (e.g. to feed a
positional consumer) is the bridge's `to_aten`, computed from the original rank
the graph carries — not the op's concern.

## 2. Payload — a GADT over the storage formats

The format GADT ties together (a) the **storage** cell type (what `data.{i}`
returns), (b) the `Bigarray` kind backing it, and (c) a phantom tag saying whether
turning a stored cell into a real value needs **quantization metadata**
(`` `Quant ``) or is fixed by the format (`` `Real ``). The phantom is what lets
the type system *require* quant data on `int8`/`int16` and *forbid* it on the
reals. (Compare `lib/aten/aten_dtype.ml`, which ties OCaml type ↔ Bigarray kind
but has no quant dimension — and which already notes Half has no Bigarray kind.)

```ocaml
type ('elt, 'ba, 'q) fmt =
  | F32  : (float, Bigarray.float32_elt,       [ `Real  ]) fmt
  | I64  : (int64, Bigarray.int64_elt,         [ `Real  ]) fmt
  | I32  : (int32, Bigarray.int32_elt,         [ `Real  ]) fmt
  | F16  : (int,   Bigarray.int16_unsigned_elt,[ `Real  ]) fmt  (* IEEE half, raw bits *)
  | BF16 : (int,   Bigarray.int16_unsigned_elt,[ `Real  ]) fmt  (* bfloat16, raw bits  *)
  | I16  : (int,   Bigarray.int16_signed_elt,  [ `Quant ]) fmt
  | I8   : (int,   Bigarray.int8_signed_elt,   [ `Quant ]) fmt
```

Two things the storage/value split makes explicit, both new with this set:

- **`'elt` is the *stored* cell, not the compute value.** Every format decodes to
  the compute domain (`float`) on read; `'q` says only whether that decode needs
  *metadata*. `F32`/`I64`/`I32` decode trivially (identity / `to_float`);
  `I16`/`I8` are affine-dequantized (§3); `F16`/`BF16` are `` `Real `` (no
  metadata) but still need a **software float codec**, because no `Bigarray` kind
  holds a half. Their cell is a raw 16-bit word (`int16_unsigned_elt`, distinct
  from the *signed* 16-bit used by quantized `I16`):
  - `BF16` decode is just the top half of a float32: `(bits lsl 16)` reinterpreted
    as `float32` bits (and encode = round-to-nearest of the top 16 bits). Cheap.
  - `F16` decode is a full binary16 unpack (1·5·10) → float32; a small `Half`
    module, the one place reduced-precision reals touch bit-twiddling.

- `F16` and `BF16` share their type indices `(int, int16_unsigned_elt, `Real)` —
  they are told apart by the *constructor* (the decode differs), exactly as with
  any two same-typed GADT arms; pattern-matching `fmt` picks the codec.

`I64` exists mainly for **index/shape data** (argmax results, gather indices,
`SymInt` hints) that flows through the graph as tensors but is never quantized and
rarely enters float arithmetic.

Quant data is carried in a second GADT indexed by the same phantom, so a
`` `Real `` payload literally cannot hold quant params and a `` `Quant `` one
cannot omit them:

```ocaml
type _ quantization =
  | No_quant :          [ `Real  ] quantization
  | Quant    : quant -> [ `Quant ] quantization

type ('elt, 'ba, 'q) payload = {
  fmt   : ('elt, 'ba, 'q) fmt;
  quant : 'q quantization;                                   (* unit-like vs quant *)
  data  : ('elt, 'ba, Bigarray.c_layout) Bigarray.Array1.t;  (* flat store *)
}
```

A full tensor is just the 6D `shape` plus the payload — no strides, no offset
(§1b): every tensor is a dense NHWC buffer indexed by `Vec6.offset shape`.

```ocaml
type ('elt, 'ba, 'q) t = {
  shape   : Vec6.shape;
  payload : ('elt, 'ba, 'q) payload;
}

(* Existentially packed for heterogeneous storage (graph env, op inputs):
   the concrete format is hidden but recoverable by matching [fmt]. *)
type packed = Tensor : ('elt, 'ba, 'q) t -> packed
```

Pattern-matching `payload.fmt` refines `'elt`/`'ba`/`'q` together, so a generic
reader recovers the element type *and* learns whether to dequantize — the
compiler rejects "read an int8 tensor as float without quant params."

## 3. Quantization — per-tensor and per-channel along C

Affine (asymmetric) quantization: `real = scale * (q - zero_point)`, the
TFLite/PyTorch convention. Two granularities, chosen per tensor:

```ocaml
type quant =
  | Per_tensor  of { scale : float;       zero_point : int }
  | Per_channel of { scale : float array; zero_point : int array }   (* length = C *)
```

- **Per-tensor** — one (scale, zero_point) for the whole tensor (typical for activations).
- **Per-channel** — one (scale, zero_point) per channel along **C**, indexed by `Vec6.get c C`
  (typical for conv/matmul *weights*, where it materially improves accuracy).

`zero_point` is a plain **signed** `int`, not a `Dim` type — it is domain data in
`[qmin, qmax]` (genuinely negative for asymmetric quant), not a size or position.
Same for the stored quanta. The `Dim` discipline is for extents/indices/counts/
offsets only.

```ocaml
val dequantize : quant -> c:int -> q:int -> float      (* picks scale[c]/zp[c] if per-channel *)
val quantize   : quant -> c:int -> qmin:int -> qmax:int -> float -> int
(* quantize x = clamp (round (x /. scale) + zero_point) qmin qmax *)
```

Read/write rules used by the compute layer (storage cell → compute `float`):

- **Read** a `` `Quant `` input (`I8`/`I16`) → dequantize using `Vec6.get c C`.
- **Write** a `` `Quant `` output → requantize; `qmin/qmax` from the format
  (`I8`: −128..127, `I16`: −32768..32767).
- **`F16`/`BF16`** → decode via the `Half`/`Bf16` codec on read, round-encode on
  write (`` `Real ``, no metadata).
- **`F32`/`I64`/`I32`** → identity / `to_float` (exact reals; the integer ones used
  for accumulators, indices, `SymInt` data, never quantized).

### Quantized compute: default vs integer path

Default everywhere: **dequantize inputs → compute in float → requantize on
store.** Simple, and exact enough to match the reference. The faster
integer-accumulator path (int8×int8 → int32 accumulate, then a single
fixed-point requantize) is a Phase-4 option; it surfaces in the symbolic IR as a
**dequant∘quant pair that the codegen can fold/elide across fused ops**, so the
substrate above is forward-compatible with it without changing these types.

## 4. Dynamic shapes (PyTorch `SymInt`)

Torch export can leave an axis size symbolic: a `SymInt` is a shape variable
(`s0`, `s1`, …) or an expression over them, carried with a `ShapeEnv` of
constraints (per-symbol ranges, divisibility, equalities) and a runtime *hint*.
`lib/pt2` surfaces these from the `ExportedProgram` (`range_constraints` + the
`SymInt`/`SymExpr` arguments). The framework already keeps *sizes* out of the
algorithm — the per-pixel functor never names an extent; only the iterator and the
footprint analysis do — so dynamic shapes slot in by making an **extent a `size`**:

```ocaml
module Symint : sig
  type t                       (* var s_i | const | a*s+b | floordiv | mod … *)
  type env                     (* ShapeEnv: per-symbol range, divisibility, equalities *)
  val bind  : env -> (string -> int) -> env   (* specialise symbols to runtime ints *)
  val eval  : env -> t -> int option          (* Some n once bound and guards hold *)
  val proves : env -> t -> [ `Ge of int | `Divides of int | … ] -> bool
end

type size = Static of Dim.extent Dim.t | Sym of Symint.t
(* shape generalises to a 6-vector of `size`; reduction/tile bounds become `size` too *)
```

The split that makes this cheap is the same one as Direct-vs-Symbolic:

- **Direct execution requires fully-static extents.** You cannot allocate a
  `Bigarray` of symbolic length. When real inputs arrive their concrete sizes are
  `Symint.bind`ed into the `ShapeEnv`, the guards are checked (mirroring dynamo's
  guard check), and every `Sym` extent resolves to `Static`. `numel`/`offset` over
  a *bound* shape give concrete `Dim.count`/`offset` exactly as in §1 — dynamic
  shapes cost nothing once specialised.

- **Symbolic / footprint / codegen mode operates over `Sym` extents** — and this is
  where they pay off. The footprint analysis (`native_symbolic_language.md` §3)
  already evaluates indices in an affine interval domain with *symbolic* endpoints,
  so a `Sym` extent is just the interval `[0, s_i)`. Validating a tiling over `Sym`
  extents proves it correct for the **whole family of admissible shapes**, not one
  concrete case; the `ShapeEnv` guards discharge the side-conditions:
  - a no-remainder tiling needs `s_i mod tile = 0` provable in the env, else
    codegen emits a tail/remainder loop;
  - a fused producer's halo must stay `< s_i` for all admissible `s_i` (a range
    guard); `offset` becomes a symbolic affine form in the size variables, so
    codegen emits index math against the runtime extent, not a baked constant.

Invariants / boundaries:

- **Per-channel quant requires a static `C`** (the scale/zp arrays are length-C).
  In practice `C` and the kernel window are static; the dynamic axes are `N`
  (batch) and `T` (sequence) — consistent with how torch dynamic shapes are used.
- **Unbacked `SymInt`s** — a size that comes from tensor *data* (`nonzero`, a
  decoded length) — are the size-level analogue of a gather: they collapse to the
  `⊤` data-dependent footprint and are flagged untileable on that axis unless the
  `ShapeEnv` declares an upper bound (torch's "unbacked symint with a hint").

## 5. Tests

- Quant **round-trip**: `dequantize (quantize x) ≈ x` within one quantum, per
  format, per-tensor and per-channel.
- **Float codecs**: `F16`/`BF16` decode∘encode round-trips within the format's
  precision; `BF16` decode equals the high 16 bits of the `F32` bit pattern.
- **Offset** math: `Vec6.offset shape coord` matches the closed form and round-trips
  with `iter` order (C fastest); `numel = ∏ extents`.
- **Broadcast**: `read_coord` maps extent-1 axes to 0 and is the identity on
  matching axes; reading a `[1,…,1,C]` source over a full output coord never OOBs.
- **`Dim` non-negativity**: `Dim.extent (-1)` / `Dim.index (-1)` are rejected;
  `numel`/`offset` results are `≥ 0`; `(off :> int)` coercion is the only path to a
  raw int (no constructor takes a raw int for `count`/`offset`).
- **Type discipline** (compile-time, exercised by the test module type-checking;
  a `(* expect type error *)` block that must fail to compile, kept as a comment):
  `Vec6.offset` rejects `offset coord shape` (roles swapped); a `coord` cannot be
  passed where a `shape` is expected; a `count` (numel) cannot be passed where an
  `offset` is expected, nor vice-versa; an unguarded `delta` cannot index storage
  without going through `Dim.index_of`. Constructing `I8` requires a `Quant _`;
  `F32` requires `No_quant`.
- **Dynamic shape binding**: a shape with a `Sym` axis cannot be `materialize`d
  until bound; `Symint.bind` + guard check resolves it to `Static` and then
  `numel`/`offset` match the equivalent statically-built shape; a guard violation
  (e.g. an extent that breaks a divisibility constraint) is rejected at bind time.
