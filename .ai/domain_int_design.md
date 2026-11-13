# Domain-typed integers — one `int`, one meaning

Status: **implemented** — the mechanism (`Core.Tagged_int`, the signature ratchet in
`tools/int_signatures` with a `todo-domain` ceiling that only goes down, the convention in `CLAUDE.md`)
and the domain migrations below have landed. What stays a bare `int` is classified in §1;
The `Semantics` scale/constant arguments are deliberately left as dimensionless
scalars. `Dim` and the guarded op scalars (`Geometry`: `Pos`, `Nonneg`, `Hw`) live in
`core`, below `Expr`, so `Expr.Intrinsic.max_pool` takes a kernel, stride and padding as
the very types `native` holds; `native` re-exports them and adds the wire codecs.
`Slice`, `Select`, `Select_scatter` and `Split_with_sizes` (and their native4d twins) hold
fences, a `Dim.index` and `Dim.extent`s; the ATen importers resolve into them
(`Split_with_sizes.of_aten` refuses a non-positive size with the value as written), and
JSON decode refuses a negative one. A rejected value is still reported as written. Arithmetic (`Dim.product_bounded`, `div_exact`, `unlin`,
`fence`/`span`, `Delta.*`, `Dim_arith`) has landed and is used by `Direct`, `Tensor`,
`Const_ssa_symbolic` and the Native4D lowerers (`Wide_permute` keeps its atoms, blocks
and runs apart with `Tagged_int`). Identity and ordinals have landed: `Tensor_id`,
`Node_id`, `Group_id` and `Cluster_var` are `Tagged_int` applications with a typed `Next`
counter (which absorbed `check_room`), `Id_supply` holds a record of them, and the
output, region-emitter, topological-position, child-path and PT2 node-index ordinals
are their own singletons. Region scratch slots have their own phantom family, `Slot` (`extent`, `count`, `offset`), so a
slot offset cannot be mistaken for a tensor's `Dim.offset`. The ATen boundary is typed: `Rank.t`, and `Aten_int` (`Dim`, `Index`,
`Size`, `Step` — signed, as written) enter `Aten_shape`, which is the only place they are
resolved (slice bounds into fences, a select index into a `Dim.index`); the importers'
error payloads report them as written. Window and im2col products are bounded once (`Window_axis.product`, `Dim_arith.Extent.scale`); a ranked class is a `Dim.index`. Extends
`native_tensor_design.md` §1a, which introduced `Dim` for tensor sizes and positions,
to every integer that names an entity. Reads with `js_backends_design.md` (the 32-bit
`int` rule) and `error_handling_design.md` (payloads carry data, not prose).

## 1. The rule, and what it is for

**A bare `int` in a signature, record field, variant payload or shared mutable state
must be dimensionless or external. Any `int` that names an entity gets a type that no
other entity's `int` can be passed for.**

This is about *domain separation*, not validation. A tensor size and a tensor
coordinate are both "an int in `[0, 2^31)`" and would pass every range check; the type
exists so that swapping them is a compile error. Checked constructors are a
consequence (a domain usually has an invariant), not the goal. Two corollaries follow
from taking that seriously:

- **Labelled arguments are only partial protection.** `~c:int ~q:int` cannot be
  swapped positionally, but a tuple, a record with two same-typed fields, a list, a
  `ref`, or an unlabelled positional argument can. Those are where the defects hide
  (`(int * int) option` for a slot's offset and count; `int * int * int` for the
  tensor/node/group watermarks; `(int * int) list` mapping an atom id to an extent).
- **Unwrapping to do arithmetic and re-wrapping is domain erasure**, not a neutral
  convenience. `Dim.extent (ext a * ext b * ext c)` states nothing about what the
  product means and forfeits every check the type could have made. If a domain-correct
  operation does not exist, the operation is missing from the domain's module; the
  fix is to add it there, not to leave the domain.

### What stays a bare `int`

| Category | Examples | Why it is not a domain |
|---|---|---|
| Budget / limit | `max_size`, `max_depth`, `Hard.*`, `transition_cost` | One dimensionless quantity, always labelled by its purpose |
| Bit pattern / storage cell | `Half` bits, the `int` in `Bigarray` element types | A representation, not an entity |
| Compare / hash result | `compare`, `hash` | Fixed by the stdlib contract |
| Expression-tree literal | `Ceil_div_pos of _ * int` in `Expr` | Typed by the surrounding tree's role parameter |
| List length / diagnostic count | `expected`/`got` arity payloads | Dimensionless tally |
| Quantum | quantised values, `zero_point`, `qmin`, `qmax` | One domain by construction (`native_tensor_design.md` §3) |
| External format | `pt2`, ATen/C bindings, `Jsont.int` on the wire | The external representation *is* the int |
| Tally | statistics counters, report fields | Dimensionless |
| As written | `Non_positive_dim { value }`, a slice's decoded `start`/`stop`, `` `Invalid_k `` | A value reported exactly as supplied, because it failed the check that would have typed it; the domain type would have to admit the very values it exists to exclude |

Everything else — extents, positions, offsets, counts, ranks, ATen dim numbers, ids,
ordinals, next-free counters, slot offsets, arena indices, algorithm-local atom/block
numbers — is a domain.

### An exempt label is a claim

A category in the table says the `int` is not a domain, so it is read against the
reason above: what does this `int` name, and what could it be confused with? A
declaration keeps an exempt label only if *every* `int` in it earns it, and one
function carries one label wherever it is declared (sibling emitters and targets
included). Entries that name an entity are `todo-domain` whatever the code around
them does with the number. The audit of the exempt entries, by family:

- **Width in bits, element width, `sew`, `lmul`, `vlen`** (52 entries, was `tally`,
  `budget` or `wire`): a quantity with a unit, returned in tuples next to other
  `int`s and passed beside the typed `Vtype.Sew`/`Sz` that exist. `Sz.bits` and
  `Vtype.Sew.bits` are not the domain's own definition (`defn` is a constructor of
  a type that already exists): they return a bare width.
- **Lane count and lane index** (21 entries, was `tally` or `expr-literal`): an
  extent and a position of a vector, where `Ssa_type.Lanes.t` and
  `Mir_type.Lanes.t` already exist and are unwrapped (`lanes_of`, `lane_ok`,
  `lanes_per_register`, `registers_of_lanes`). The Loop vector record that holds
  `lanes` and `parts` beside its literal bounds is `todo-domain` as a whole.
- **Block rows, unroll factor, parts, block keys** (7 entries, was `tally` or
  `budget`): a row count is a domain, not a limit. The same `rows` was `budget` in
  `Ssa_opt_rows.program` and `tally` in `analyze` two declarations later; a
  declaration that also returns a count of blocked loops is not exempt either,
  since one of its `int`s is not. A limit (`max_depth`, `fuel`) stays `budget`.
- **Wasm local, buffer and site indices** (16 entries, was `wire` or `budget`): ordinals
  in a numbering private to one emitter. A wasm module does contain a local index,
  but the emitter's `get`, `set`, `fresh`, `buffer_param` and `reserve_local` are
  its own counters, not a field it reads from an external format; the Loop and SSA
  emitters now carry the same label (`todo-domain`) for each. The scan meter's pair
  of locals (`budget`) is two of them.

## 2. Taxonomy

Two mechanisms, chosen by whether the domains share arithmetic.

**Phantom-role families** (`'role t = private int`, as `Dim` today): for domains that
combine with one another under defined operations, so the algebra lives once and the
roles stay non-unifiable.

**Generative singletons** (`Tagged_int.Make ()`): for domains that are only compared,
keyed and printed — ids, ordinals, arena indices, local atom/block numbers. Each
application is a fresh, incompatible `type t = private int`.

| Family | Types | Mechanism | Owner |
|---|---|---|---|
| Algorithm-local | `Atom.t`, `Block.t` | singleton, private to the module | `native4d/wide_permute` |
| Arena | `Ground_expr.Node.t` | singleton | `native/transform/ground_expr` |
| As written (untrusted, signed) | `Aten_int.Dim.t`, `Aten_int.Index.t`, `Aten_int.Size.t` | singleton per kind | `native/aten_int`, normalised only by `Aten_shape` |
| Geometry | `extent`, `index`, `count`, `offset`, `delta` (existing); `fence` (new) | phantom role | `native/dim` |
| Identity | `Tensor_id`, `Node_id`, `Group_id`, `Cluster_var` | singleton (replaces four hand-rolled copies) | their current modules |
| Next-free | `Tensor_id.Next.t`, `Node_id.Next.t`, `Group_id.Next.t` | per id space | with the id |
| Ordinal | `Output_ordinal`, group member, topological position, child index | singleton per space | owner of the space |
| Rank | `Rank.t` (≥ 0) | private int | `native/rank` |
| Slot | `Slot.offset`, `Slot.count`, `Slot.extent`, `Slot.position` | phantom role | `native/region_slots` |

Notes on the less obvious rows:

- **`fence`** is a position with `0 ≤ f ≤ extent`, i.e. *one past the last index is
  legal*. A slice's resolved `start` and `stop` are fences, not indices (`stop` may
  equal the extent), which is why neither `Dim.index` nor `Dim.extent` fits them.
  `Dim.span` turns two fences into an `extent option` (`None` when empty — the
  engine has no empty tensors).
- **Strides** need no new role: a dense stride is a product of extents, so `Vec6.t`
  of `Dim.count`.
- **Slot** is deliberately not `Dim.offset`/`Dim.count`. A slot offset indexes a
  region's flat scratch array; a `Dim.offset` indexes a tensor. Conflating them is
  exactly the mistake the types exist to prevent.
- **As written** exists because the ATen boundary carries values that are signed,
  unvalidated and mean different things (`-1` as a dim number, a size, an index).
  Error payloads report them *as written* (`Aten_shape`'s existing convention); the
  type keeps a written dim from being treated as an extent.
- **`Expr` cannot see `native`'s `Dim`** (`expr` depends only on `core` and `fmt`).
  `Dim.index`/`Dim.delta` are manifest aliases of `Expr.Role`'s markers for that
  reason. Window geometry inside `Expr` (`Max_pool`'s eight `int` parameters) gets its
  own small role-typed vocabulary in `Expr`, mapped at the `native` seam.

## 3. Mechanism

### 3.1 `Tagged_int` (lib/core)

```ocaml
module type S = sig
  type t = private int
  val of_int : int -> t          (* owner-internal: not a validation point *)
  val to_int : t -> int          (* also the free coercion (x :> int) *)
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val succ : t -> t
  val pp : Format.formatter -> t -> unit
  module Map : Map.S with type key = t
  module Set : Set.S with type elt = t
end

module Make (_ : sig val prefix : string end) () : S   (* generative *)
```

- Lives in `lib/core` (Fmt only), so every JS-reachable library can use it, including
  those below `native`. `jsont` is not included; owners that need it add it with
  `Jsont.map`, which keeps `core` free of a Jsont dependency.
- `private int` makes the exit a **free coercion**, so hot paths pay nothing.
  Constructors are kept off per-element paths, as `Dim` already does.
- Adoption dedupes the four hand-rolled copies (`Tensor_id`, `Node_id`, `Group_id`,
  `Cluster_var`), which today each restate `compare`/`equal`/`pp`/`Map`/`Set`.

### 3.2 Arithmetic stays in the domain

Domain erasure happens because the operation is missing. The audit's arithmetic
rebrands are a short list; `Dim` gains exactly those:

```ocaml
(* same-domain, in Dim *)
val product_bounded :
  limit:int64 -> extent t list -> (extent t, product_witness) result
  (* divide-before-multiply, ~limit EXCLUSIVE — the Vec6.numel_bounded convention;
     returns the un-multiplied witness on failure, never the wrapped product *)
val div_exact : extent t -> by:extent t -> extent t option   (* exact quotient *)
val divides : by:extent t -> extent t -> bool
val unlin : offset t -> extent t -> offset t * index t        (* dual of [lin] *)
val fence_of_index : index t -> fence t
val span : fence t -> fence t -> extent t option

module Delta : sig                     (* the Semantics carrier's arithmetic *)
  val add : delta t -> delta t -> delta t
  val neg : delta t -> delta t
  val min : delta t -> delta t -> delta t
  val max : delta t -> delta t -> delta t
  val floor_div_pos : delta t -> by:extent t -> delta t
  val of_extent : extent t -> delta t
end
```

`Dim_arith` (a separate module placed after `Op_config`, because `Op_config` already
depends on `Dim`) holds the operations that mix domains: `Delta.scale ~by:Op_config.Pos.t`,
`Extent.scale ~by:Op_config.Pos.t`. This is the seam where a kernel size or stride
meets a position.

`Dim.to_int64` is the one named exit for the `Int64`-exact arithmetic that
`resize`/`pool` do on purpose; the coercion `(x :> int)` remains the exit for storage
indexing and wire encoders. After migration those are the only two kinds of exit, so a
grep for `:> int` is an audit of every place a domain is left.

### 3.3 Next-free counters

An id is typed; the counter that hands the next one out is not, which is how a tensor
watermark can be passed where a node watermark is wanted. Each id module gains a
`Next.t` (the first free id in that space) with `of_ids`/`alloc`/`alloc_n`. It absorbs
`Tensor_id.check_room` (whose two raw `int` arguments are the same defect one level
down) and replaces the hand-rolled `max (to_int id + 1)` folds, which exist in more
than one place. `Id_supply`'s `int * int * int` marks become a record of `Next.t`.

### 3.4 Gates

Raw-`int` constructors remain the sanctioned *entry* for trusted literals
(`Vec6.shape ~n ~t ~d ~h ~w ~c`, `Dim.extent`, `Op_config.Pos.of_int`), and their
checked twins (`Dim.extent_checked`, `Op_config.Bad.*`) the entry for untrusted input.
The rule is about what happens after entry: once a value has a domain it does not
round-trip through `int` inside the engine. The ATen boundary is the one place values
arrive *without* a domain, hence `Aten_int`.

## 4. Enforcement

Types alone do not stop the next `int`. Three layers:

1. **A signature allowlist ratchet.** Every `int` token in an `.mli` of an in-scope
   library must appear in an allowlist file with a category tag from the table in §1.
   A new bare `int` in a signature fails the build until it is either given a domain
   or allowlisted with a reason. The allowlist may only shrink. Signatures, not
   implementations, because a signature is the contract other code is written
   against and is precisely checkable (unlike domain inference).

   A library whose interface *is* its implementation (the emitters, the targets)
   has no `.mli` to read. An `ml:<dir>` line in the scope file also reads the
   interface the compiler infers for each `.ml` of that library without an `.mli`:
   the text `ocamlc -i` prints, through the same parser, with the library's object
   directories on the include path, its alias module opened as dune does, and its
   external libraries as `ocamlfind` packages. Nothing links against the compiler's
   own library, so the check does not follow a compiler release; the dune rule
   depends on the library's `check` alias, which keeps it hermetic. A bare `<dir>`
   line is unchanged. `ml:lib/ssa_c`, `ssa_js` and `ssa_wasm` are in scope. A
   library enters with an opening `todo-domain` balance (the three emitters: three
   records of value ids and name counters) that is part of the recorded ceiling;
   the ceiling only goes down from there. Judgement calls in that classification:
   the words of a host-ABI failure record are `wire`, but the ordinals of a private
   numbering (a wasm local index, a buffer parameter or index, a failure site) are
   `todo-domain`, as is a lane count; indent depths are `tally`, the literal an
   emitter prints `as-written`.

   The three machine targets (`machine_target_aarch64`, `_riscv32`, `_x86_64`)
   entered next, with an opening `todo-domain` balance of 66. It is register numbers
   (`x`, `gpr`, `rax`, `range`, `allocatable`, `view`, `sequence`, the unit
   constructors), the value ids the selectors unwrap to key their tables, and the
   instruction-form variants, which carry lane positions and bit indices. Judgement
   calls: a width in bits (`Sz.bits`, `unit_bits`, `shape`, `bitmask_immediate`) is
   a domain and a `todo-domain` (it is returned in tuples beside other `int`s, and the
   typed `Sz` already exists), a lane count and a lane index are `todo-domain`
   (`Mir_type.Lanes` exists and is unwrapped), the per-event index in `slot` is a `cell`, and the
   fuel and depth of the interpreter runs are `budget`. The balance only goes down:
   typing the register number as a `Mir_id.Unit`-style domain retires most of it.

   The four `coralnpu*` libraries entered after them, adding 75 `todo-domain`
   entries (rows, strides, panel and block extents, byte offsets, register
   numbers, output ordinals in the generators and the run harness). Judgement
   calls: the MMIO register offsets, the descriptor size and the mailbox layout
   are `wire` (a layout shared with the hardware or the guest program), the
   register bit masks are `bits`, instruction immediates in the kernel emitters
   are `wire`, the refusal payloads that report a budget or a count are
   `as-written`, element-width and register-group factor arguments are
   `todo-domain` (`Vtype.Sew` and `Vtype.Lmul` exist and are bypassed), and so are
   lane arguments.

   The eight `machine_rivet_*` libraries entered last, adding 163 `todo-domain`
   entries: the register-allocation intervals and positions the routes re-export,
   value ids unwrapped as keys, byte offsets and sizes in the data blocks and the
   outliner, the output ordinals of the run harnesses. Judgement calls: assembler
   operands and directives, instruction encoding fields, ELF and image padding,
   syscall numbers and the process exit code are `wire`, the typed register
   index's constructors are `defn`, a payload reporting a number as the check saw
   it is `as-written`, and a timeout or branch reach is a `budget`. A library
   re-exporting a type through `include` or a functor application lists its
   declarations once per file, so the opening balance counts each re-export. The
   interface read for these libraries needs the workspace's own build of a
   vendored library (`elf`) ahead of an installed one, so the tool also puts the
   `vendored` object directories on the include path.

   `js_ast` and `wasm_ir` entered next, the first of the libraries js_of_ocaml
   reaches, adding 3 `todo-domain` entries: the Wasm instruction type (label
   depths, function, global and local indices, lane indices), `Wasm.Module`'s
   `map_calls`, and the export `kind`. Judgement calls: a literal in a JavaScript
   expression tree (`Idx.const`, `scale`, an array length) is `expr-literal`,
   printed through `float_of_int`; a memory-argument alignment and offset, an
   opcode byte, a data-segment offset and the memory limits are `wire`, and the
   offsets are bounded to `[0, 2^31)` at the declaration so they fit a 32-bit
   `int`; a lane count is `todo-domain` and a census count is `tally`; the numbers a validation
   error reports are `as-written`. No 32-bit hazard was found: the only `int`s
   that could approach 2^31 are the memory offsets, bounded at the declaration,
   and the printer's literals, which go through `float_of_int`.

   `loop_ir` entered after them, adding 43 `todo-domain` entries: the ordinals
   and local indices of the Wasm and C emitters (a local's index, a failure
   site, a name's ordinal), the lane positions of the vector forms, the extents
   and offsets the lowering passes thread (`extent`, `width`, `steps`, a pool's
   element count and offset), and a statement's scan widths. Judgement calls: the
   `Loop_array`, `Loop_temp` and `Loop_var` ids are already `Tagged_int`
   instances, so their `of_int`/`to_int` are `defn`; the constants of an index
   tree and of an affine form are `expr-literal` (the index domain
   `[-2^31, 2^31)` is the contract, see `Loop_range.domain`); failure-record
   words, memory placements and the process exit codes are `wire`. Two
   32-bit-`int` hazards sit behind the `wire` label and are not hidden by it:
   `Loop_bundle_wasm` admits a memory of exactly `2^31` bytes (`memory_cap` is
   rejected with `>`), and narrows `Placement.total` with `Int64.to_int`, which
   wraps under js_of_ocaml at that one value; and its `table_alloc` narrows a
   table address with `Int64.to_int` before the total is checked. Both need a
   multi-gigabyte placement to trigger, so neither is reachable by a model that
   builds today.

   `pt2` entered last of the JS-reachable libraries, adding 3 `todo-domain`
   entries: `Pt2_tensor.product`, `numel` and `contiguous_strides`, which do
   extent and stride arithmetic on raw `int`s that `pt2_aten` later wraps in
   `Dim`. Judgement calls: the decoded tensor record, the pickle rebuild
   descriptor and the zip end-of-central-directory fields are `wire` (the archive's
   own representation), the zip `Limits` and their fields are `budget`, and the
   counts and paths a rejection reports are `as-written`. No 32-bit hazard was
   found: every decoded size is bounded when decoded, and `product` folds in
   `int64` and narrows only to the runtime's own `int` range, so on js_of_ocaml
   it bounds to 32 bits.
2. **A review question**, added to the conventions: *"what does this `int` name, and
   what could it be confused with?"*
3. **Exit hygiene**: the count of `(x :> int)`/`Dim.to_int` exits is tracked, and
   should fall once §3.2 lands. It is a measure, not a gate.

## 5. JavaScript backends

No representation changes: `private int` is `int`, and js_of_ocaml still sees 32 bits.
The design *helps* the 32-bit rule in two ways: `product_bounded` is the one place a
product of extents is formed and bounds each factor before multiplying (so the "check
on a wrapped result is not a bound" class cannot recur in new code), and a domain type
makes an aggregate visibly not the same type as its factors. Nothing here removes the
need for `make jsoo.runtest` and `make jsoo.inline-runtest`; both must stay green
through every step.

Probing that claim: `Graph_builder` refuses a tensor at or above `Hard.numel` when it is
built, and an op whose output is checked (reshape) refuses one at snapshot time, but
`Snapshot.create` accepts a hand-decoded graph whose `Expand` produces 2^35 elements,
so the graph view does not bound numel. No path from such a graph into a Native4D
relabel group was found (the recognizers decline it first), so the reliance was not a
demonstrated defect, only an unenforced one; `Extent_product.bounded` now enforces it.

The claim currently made in comments — "this product is at most the tensor's element
count, which graph construction bounds inside 32 bits" — is a convention, not a
type-enforced fact: the bound is applied by the importers and by
`Kernel.Bounds.signature`, and not by the validated graph view. Native4D code that
multiplies extents relies on it. `product_bounded` replaces that reliance with a check
at the point of use.

## 6. Alternatives considered

| Option | Cost | Verdict |
|---|---|---|
| Status quo plus comments | none | Leaves an unchecked claim repeated in several files, and does nothing for identity/ordinal mixups |
| Records with distinct field names | low | Does not stop swapping same-typed fields across records or through tuples/lists |
| **Targeted domains, phantom family for algebra, generative singletons for identity** | moderate, incremental | **Chosen** |
| Wrap every `int` | very high | The budget, tally, wire and hash categories gain nothing; most would re-raise on construction |
| A single `'role t` for everything | low code, high coupling | Unrelated domains (ids vs. extents) would share an algebra they must not have |
| Abstract (non-`private`) types | same | Loses the free `:>` coercion on hot paths (`Vec6.read_at` was profiled for exactly this) |

## 7. Cost and benefit

At time of writing roughly 1,800 `int` tokens sit in the source libraries; about a third
name a domain, and about half of the remainder are legitimate under §1. The domain
sites concentrate: the windowed ops and hyper-parameters are the largest block and are
mechanical once §3.2 exists; the ATen-as-written and identity/ordinal blocks are
smaller and structurally more valuable, because they are where same-typed values are
most easily swapped. The Native4D additions are few and self-contained.

Benefits: swaps become compile errors; the unwrap/re-wrap arithmetic disappears from
call sites; the bound on products is enforced where the product is formed; error
payloads report values in the domain of what they describe.

Costs: a wider `Dim` surface to maintain; test code that builds typed values needs
helpers; a parallel role vocabulary inside `Expr`; and a review burden on the
allowlist. The type-level change has no runtime cost on the profiled paths (exits are
coercions), which is to be confirmed by the timed inference run rather than assumed.

## 8. Migration principles

- **Order by dependency, not by count.** Arithmetic and mechanism first (nothing else
  can migrate without them), identity and as-written next (independent of each other),
  the windowed-op block after both, then the isolated libraries.
- **Every step is a normal, self-contained change** with `make precommit` green,
  `make jsoo.runtest` and `make jsoo.inline-runtest` green, and this document updated
  in the same change.
- **A claim that a check bites is proved by making it fail** (revert, watch red,
  restore) — for `product_bounded` this means a shape whose per-axis extents pass
  individually and whose product wraps a 32-bit `int`.
- **Do not widen a domain to make a call site compile.** A site that needs an
  operation the domain lacks either gets the operation added (§3.2) or is a
  genuine boundary and is allowlisted with a category.

## 9. Open questions

1. Is `lib/core` the right home for `Tagged_int`, or should it be a new library so
   `core` stays exactly "Fmt glue"? (`expr`'s own private wrappers, `Local_var` and
   `Reduce_var`, could adopt it only if it sits below `expr`.)
2. Should `Expr`'s geometry vocabulary be its own types, or should the phantom role
   family move down into `expr` so `Dim` and `Expr` share it (extending what
   `Dim.index`/`Dim.delta` already do)?
3. `Symint`'s coefficients (`Const of int`, `Scale of int * t`) mix extents and signed
   deltas in one variant; whether to type them or treat the symbolic polynomial as its
   own dimensionless algebra is undecided.
4. Test code: a `Test_util` of typed builders versus per-file `of_int` calls. Tests
   were not part of the survey.
