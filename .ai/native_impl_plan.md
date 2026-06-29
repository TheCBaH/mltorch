# Native engine — implementation plan (module order, tests, printers)

The build order for `lib/native/`, bottom-up: each module compiles and its expect
test is green before the next starts. This sequences the design in
`native_inference_plan.md` / `native_tensor_design.md` / `native_compute_design.md`
/ `native_symbolic_language.md` into concrete, individually-testable steps.

## Two hard rules

1. **Every public type ships a printer.** `val pp : Format.formatter -> t -> unit`
   (and a `to_string` derived from it). This is not cosmetic: the expect tests
   assert on printed output, so *the printer is the test's notion of the value* — a
   type with no `pp` cannot be expect-tested, and the symbolic IR is unreadable
   without one. New hand-written printers should use `Fmt` as the composition
   layer (`Fmt.pf`, `Fmt.list`, `Fmt.option`, boxes/separators) rather than
   ad-hoc separator refs or manual indentation; this keeps nested printers
   reusable and lets `Format` make normal line-breaking decisions. Plain
   ADTs/records may use `[@@deriving show]`; the **GADTs, `private`/abstract
   types, and existentials** (`Dim`, `Vec6`, `fmt`, `payload`, `expr`) need
   **hand-written** `pp` — derivers don't handle them. Distinct types must print
   distinctly (e.g. a `shape` in `[…]`, a `coord` in `(…)`) so a test reading the
   output can tell them apart.

2. **Every module lands with its expect test.** No module merges without a
   `test/native_<m>_test.ml`. Tests follow the repo convention exactly: a per-test
   `(library …)` in `test/dune` with `(inline_tests)` + `(preprocess (pps
   ppx_expect))`, `let%expect_test`, accept output with `dune promote`.

```scheme
; lib/native/dune
(library (name native) (wrapped false)
 (libraries core fmt))        ; pure OCaml + Bigarray; no libtorch, no ctypes

; test/dune — one stanza per module under test, e.g.
(library (name native_dim_test) (modules native_dim_test)
 (libraries native) (inline_tests) (preprocess (pps ppx_expect)))
```

## Build DAG

| # | module (`lib/native/`) | depends on | key types | test pins |
|---|---|---|---|---|
| 1 | `axis.ml` | — | `axis` | order, `pp` |
| 2 | `dim.ml` | — | `extent/index/count/offset/delta` | ≥0 rejection, arith, `index_of` guard, `:>`, `pp` |
| 3 | `vec6.ml` | axis, dim | `shape`, `coord` | offset closed-form, `numel`, broadcast, `iter` order, role-distinct `pp` |
| 4 | `half.ml` | — | (codecs) | bf16=hi16 of f32, f16 round-trip, `pp` |
| 5 | `quant.ml` | dim | `quant` | (de)quant round-trip, per-channel, `pp` |
| 6 | `payload.ml` | dim, half, quant | `fmt` GADT, `quantization`, `payload`, `packed_fmt` | per-format decode, GADT discipline, `pp` |
| 7 | `tensor.ml` | vec6, payload | `('e,'b,'q) t`, `packed` | build/read, broadcast read, `materialize`, `shift_in_bounds`, `pp` |
| — | **milestone: substrate (master Phase 0) green** | | | |
| 8 | `symint.ml` | dim | `Symint.t`, `env`, `size` | bind+guards, `eval`, unbacked, `pp` |
| 9 | `semantics.ml` | tensor, dim | `SEMANTICS`, `INDEX` sigs | (sig only — exercised via Direct) |
| 10 | `direct.ml` | semantics, tensor | `Direct` | per-primitive values, `tap` pad→0 |
| 11 | `schedule.ml` | vec6, tensor | `for_each`, `evaluate` | fills a tensor from a pixel fn; iter order |
| 12 | `ops/*.ml` | semantics | `Relu`/`Add`/`Reduce`/`Conv2d`/… functors | Direct results vs hand-computed / ATen golden |
| 13 | `expr.ml` + `symbolic.ml` | dim, payload, tensor | `Tensor_sig`, `index_expr`/`index`, `expr`, `Symbolic` | `pp` golden of built exprs; `eval(Symbolic) = Direct` |
| 14 | `footprint.ml` | expr, symint | `region`, `verdict` | the 4 stressors (elementwise/conv/softmax/gather) |
| 15 | `codegen.ml` | footprint, expr | loop-nest IR | fused == unfused; `pp` of emitted nest |
| 16 | `native_interp.ml` | tensor, ops, pt2 | driver | resnet18 vs `Interp.run` |

Steps 1–7 are "the basic modules"; detail below. 8–16 follow the same two rules and
are specified in the design docs — sketched at the end.

---

## 1. `axis.ml`

```ocaml
type axis = N | T | D | H | W | C        (* fixed order; C innermost *)
val all : axis list                       (* [N;T;D;H;W;C] *)
val pp : Format.formatter -> axis -> unit  (* "N" … "C" *)
```

```ocaml
let%expect_test "axis order + pp" =
  List.iter (fun a -> Format.printf "%a " Axis.pp a) Axis.all;
  [%expect {| N T D H W C |}]
```

## 2. `dim.ml`

The scalar dimensional discipline (native_tensor_design §1a). `private int` per
phantom role → distinct types, free `:>` to `int`, guarded construction.

```ocaml
type +'role t = private int
type extent and index and count and offset and delta
val extent : int -> extent t            (* raises Invalid_argument if < 0 *)
val index  : int -> index  t
val ( *@ ) : count t -> extent t -> count t
val lin    : offset t -> extent t -> index t -> offset t   (* o*ext + i *)
val to_delta : index t -> delta t
val index_of : extent:extent t -> delta t -> index t option (* Some iff 0≤d<extent *)
val pp : Format.formatter -> 'role t -> unit                (* the integer *)
```

`pp` prints the value; non-negativity and role separation are pinned by
construction tests and (commented) compile-failure blocks.

```ocaml
let%expect_test "dim: non-negative + guard + coercion" =
  Format.printf "%a@." Dim.pp (Dim.extent 224);
  [%expect {| 224 |}];
  Format.printf "%b@." (match Dim.index (-1) with exception Invalid_argument _ -> true | _ -> false);
  [%expect {| true |}];
  (* delta guarded into an extent: 3 in [0,5) ⇒ Some, -1 ⇒ None *)
  let g d = Dim.index_of ~extent:(Dim.extent 5) (Dim.delta d) in
  Format.printf "%b %b@." (g 3 <> None) (g (-1) = None);
  [%expect {| true true |}]
  (* compile error (kept as a comment, must NOT compile):
     let _ = Dim.( *@ ) (Dim.index 1) (Dim.extent 1)   (* count expected, got index *) *)
```

## 3. `vec6.ml`

Shared 6-component storage, parameterised by the single `Dim` role its components
carry (native_tensor_design §1b). `get` returns `extent` for a shape, `index` for
a coord; the three per-axis roles give three non-unifiable vector types.

```ocaml
type 'd t
type shape  = Dim.extent Dim.t t
type coord  = Dim.index  Dim.t t
type deltas = Dim.delta  Dim.t t
val shape  : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> shape
val coord  : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> coord
val deltas : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> deltas
val numel  : shape -> Dim.count Dim.t
val offset : shape -> coord -> Dim.offset Dim.t
val in_bounds : shape -> coord -> bool
val iter   : shape -> (coord -> unit) -> unit
val pp_shape : Format.formatter -> shape -> unit   (* [N=1 T=1 D=1 H=2 W=2 C=3] *)
val pp_coord : Format.formatter -> coord -> unit   (* (0,0,0,1,0,2)             *)
```

Distinct delimiters (`[…]` vs `(…)`) are the visual proof that the two roles are
different — and the printers are themselves type-checked against the role.
(Broadcast is no longer a `Vec6` read trick: `load` is strict and the op reduces
the coord against the source shape via `Pointwise.broadcast_coord` — see
native_compute_design §1.)

```ocaml
let%expect_test "vec6: offset, numel, pp" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3 in
  let c = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:1 ~w:0 ~c:2 in
  Format.printf "%a @ %a -> off %a, numel %a@."
    Vec6.pp_shape s Vec6.pp_coord c Dim.pp (Vec6.offset s c) Dim.pp (Vec6.numel s);
  [%expect {| [N=1 T=1 D=1 H=2 W=2 C=3] @ (0,0,0,1,0,2) -> off 8, numel 12 |}]
```

## 4. `half.ml`  (`Half` + `Bf16`)

Software float codecs — no Bigarray half kind (native_tensor_design §2). Raw
16-bit `int` ↔ `float`.

```ocaml
module Bf16 : sig val to_float : int -> float  val of_float : float -> int  val pp : … end
module Half : sig val to_float : int -> float  val of_float : float -> int  val pp : … end
```

```ocaml
let%expect_test "bf16 = high 16 bits of f32; half 1.0" =
  Format.printf "%g %g@." (Bf16.to_float 0x3F80) (Half.to_float 0x3C00);
  [%expect {| 1 1 |}];
  (* bf16 round-trip of a representable value *)
  Format.printf "%g@." (Bf16.to_float (Bf16.of_float 1.5));
  [%expect {| 1.5 |}]
```

## 5. `quant.ml`

Affine quant, per-tensor and per-channel along C (native_tensor_design §3).
`zero_point`/quanta are plain signed `int` (domain data), not `Dim`.

```ocaml
type quant = Per_tensor of { scale:float; zero_point:int }
           | Per_channel of { scale:float array; zero_point:int array }
val dequantize : quant -> c:int -> q:int -> float
val quantize   : quant -> c:int -> qmin:int -> qmax:int -> float -> int
val pp : Format.formatter -> quant -> unit   (* "Per_tensor s=0.5 zero_point=0" *)
```

```ocaml
let%expect_test "quant round-trip + pp" =
  let q = Quant.Per_tensor { scale = 0.5; zero_point = 0 } in
  Format.printf "%a deq(4)=%g req=%d@."
    Quant.pp q (Quant.dequantize q ~c:0 ~q:4)
    (Quant.quantize q ~c:0 ~qmin:(-128) ~qmax:127 2.0);
  [%expect {| Per_tensor s=0.5 zero_point=0 deq(4)=2 req=4 |}]
```

## 6. `payload.ml`

The format GADT + quantization GADT + payload (native_tensor_design §2). Decode
`storage cell -> float` dispatches on `fmt` (trivial / Half/Bf16 / Quant).

```ocaml
type ('elt,'ba,'q) fmt = F32 : … | F16 : … | BF16 : … | I64 : … | I32 : … | I16 : … | I8 : …
type _ quantization = No_quant : [`Real] quantization | Quant : quant -> [`Quant] quantization
type ('e,'b,'q) payload = { fmt : ('e,'b,'q) fmt; quant : 'q quantization; data : … Array1.t }
type packed_fmt = Fmt : ('e,'b,'q) fmt -> packed_fmt
val pp_fmt : Format.formatter -> ('e,'b,'q) fmt -> unit   (* "f32" "bf16" "i8" *)
val pp : Format.formatter -> ('e,'b,'q) payload -> unit    (* "i8[Per_tensor s=0.5 zero_point=0]×N" *)
```

```ocaml
let%expect_test "fmt names + quant discipline" =
  Format.printf "%a %a %a@." Payload.pp_fmt F32 Payload.pp_fmt BF16 Payload.pp_fmt I8;
  [%expect {| f32 bf16 i8 |}]
  (* compile error, kept as comment (must NOT compile):
     let _bad : (_,_,[`Real]) payload = { fmt = I8; quant = No_quant; data } *)
```

## 7. `tensor.ml`

`{ shape; payload }` — dense, no strides (native_tensor_design §1c). Read =
`Vec6.in_bounds` (strict) → `Vec6.offset` → decode. `pp` shows format + shape + a
bounded element preview.

```ocaml
type ('e,'b,'q) t = { shape : Vec6.shape; payload : ('e,'b,'q) Payload.payload }
type packed = Tensor : ('e,'b,'q) t -> packed
val read : packed -> Vec6.coord -> float
val shift_in_bounds : packed -> Vec6.coord -> (Axis.axis * Dim.delta Dim.t) list -> Vec6.coord option
val materialize : Vec6.shape -> (Vec6.coord -> float) -> packed   (* fresh dense f32 *)
val pp : Format.formatter -> packed -> unit   (* "tensor f32 [N=1…C=3] {0.5, 0.5, …}" *)
```

```ocaml
let%expect_test "tensor read + broadcast + pp" =
  let t = Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3)
            (fun c -> float_of_int (Dim.to_int (Vec6.get c C))) in   (* [0;1;2] over C *)
  Format.printf "%a@." Tensor.pp t;
  [%expect {| tensor f32 [N=1 T=1 D=1 H=1 W=1 C=3] {0, 1, 2} |}];
  (* broadcast read at a far N..W coord still hits the C lane *)
  Format.printf "%g@." (Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:5 ~w:9 ~c:2));
  [%expect {| 2 |}]
```

**Milestone:** steps 1–7 = master **Phase 0** (substrate). Everything below builds
on a fully-printed, fully-tested data layer.

---

## 8–16 (same two rules; specified in the design docs)

- **8 `symint.ml`** — `size = Static | Sym`, `ShapeEnv`. Test: `bind`+guard check
  resolves `Sym`→`Static` and matches the static shape; a divisibility-violating
  bind is rejected; `pp` renders `s0`, `s0 mod 8 = 0`, etc. (native_tensor §4).
- **9 `semantics.ml`** — `SEMANTICS`/`INDEX` signatures only; no test of its own,
  exercised through Direct.
- **10 `direct.ml`** — `Direct : SEMANTICS with type t = float`. Test each
  primitive and that `tap` returns 0 in the pad region (native_compute §3).
- **11 `schedule.ml`** — `for_each`/`evaluate`. Test: `evaluate` over a pixel
  function fills a tensor whose `pp` matches the hand-rolled `materialize`.
- **12 `ops/*.ml`** — `Relu`, `Add`, a `Reduce`, then `Conv2d`/`Matmul`/`MaxPool`.
  Test Direct outputs against hand-computed small cases, later element-wise vs the
  ATen path (gated, like `interp_cram`).
- **13 `expr.ml` + `symbolic.ml`** — `Tensor_sig`, the index/value IR, `Symbolic`.
  **The `expr` printer is the most important in the project** — infix/sexp so the
  IR is legible (`relu((conv(x) + y))`). Tests: `pp` golden of built exprs, then
  `eval (Symbolic.pixel) ~binding == Direct.pixel` per op (native_symbolic §2.4, §3).
- **14 `footprint.ml`** — interval `region` + `verdict`, with `pp` (`x.H ∈ [h0-1,
  h1+1)`, `C: full`, `table: ⊤`). Tests: the four stressors’ verdicts
  (native_symbolic §4–5).
- **15 `codegen.ml`** — lower `(expr, schedule)`→loop-nest IR; `pp` the nest. Test:
  fused result equals unfused.
- **16 `native_interp.ml`** — graph driver; gated cram vs `Interp.run` on resnet18.

## Cadence

Per module: write `.ml` (+ `pp`), write `test/native_<m>_test.ml`, `dune build`,
`dune promote` to accept the printed goldens, `make format`. A module is "done"
only when its printer renders every public value and its expect block is promoted.
