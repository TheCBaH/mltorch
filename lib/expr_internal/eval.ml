(* [index.Tensor]'s gather-value validation: a multi-field payload gets a
   named record rather than a bare tuple, per CLAUDE.md's "payloads carry
   data, not prose". Defined before [index_error] (which names it) and before
   [eval_index] (which calls [resolve_gather_index]). *)
module Gather_index_out_of_range = struct
  type t = { raw : int64; extent : int }
end

let pp_gather_index_out_of_range fmt
    ({ Gather_index_out_of_range.raw; extent } : Gather_index_out_of_range.t) =
  Fmt.pf fmt "gather index %Ld out of range [-%d, %d]" raw extent (extent - 1)

(* Validate an [index.Tensor] gather value in the Int64 domain BEFORE
   narrowing to [int]: [Int64.to_int] truncates to 63 bits on a 64-bit
   runtime, so checking AFTER narrowing would let a value near
   [Int64.min_int]/[Int64.max_int] wrap into a spuriously in-range [int] --
   e.g. a wrapped value landing on -1 would incorrectly select the final
   element. The interval mirrors ATen's own pre-normalization valid range for
   a single index value, [-extent, extent-1].

   No [Sys.int_size] branch is needed, unlike [float_of_index]'s: the
   validation must hold identically on every backend, so there is no "safe to
   skip" case. Once validated, [raw] is comfortably within native [int]'s
   range on every backend (32-bit js_of_ocaml included), since [extent] is
   already bounded far below either width by the engine's own extent ceiling
   -- so narrowing after the check is safe. Comparisons on [Int64.t] don't
   allocate (only operations that PRODUCE a new one do), so the steady-state
   per-call cost is exactly two [Int64.compare]s (no allocation) plus one
   [Int64.to_int] on the accepted path. *)
let resolve_gather_index (raw : int64) ~(extent : int) :
    (int, [> `Gather_index_out_of_range of Gather_index_out_of_range.t ]) Err.t
    =
  let extent64 = Int64.of_int extent in
  let lower = Int64.neg extent64 (* -extent *)
  and upper =
    Int64.sub extent64 1L
    (* extent - 1 *)
  in
  if Int64.compare raw lower < 0 || Int64.compare raw upper > 0 then
    Err.fail
      (`Gather_index_out_of_range { Gather_index_out_of_range.raw; extent })
  else
    let raw_i = Int64.to_int raw in
    Err.return (if raw_i < 0 then raw_i + extent else raw_i)

(* Only what index evaluation can actually raise. [Eval.error] widens this in
     stage 4, once intrinsics and loads exist -- a stage cannot publish a type
     equation naming a module that arrives in the next one.

   [`Data_index_unexpected_here] is the public [index]'s own "impossible
   resolver" tag (below): a [Data] node cannot appear in a caller's input by
   construction whenever that caller never builds one, so the stub resolver
   [index] instantiates [eval_index] with is provably dead code for them --
   this tag is what it fails with if that invariant were ever violated. It is
   also reused, unchanged, by [Kernel_eval.machine]'s structurally-dead
   virtual arm (native side): both call sites mean exactly the same thing, "a
   [Data] index reached a context that structurally cannot produce one", so
   one shared tag is more honest than two near-duplicates. *)
type index_error =
  [ Checked.error
  | `Data_index_unexpected_here
  | `Gather_index_out_of_range of Gather_index_out_of_range.t
  | `Index_not_exact_in_float of int
  | `Unbound_reducer of Reduce_var.t ]

let pp_index_error fmt : [< index_error ] -> unit = function
  | #Checked.error as e -> Checked.pp_error fmt e
  | `Data_index_unexpected_here ->
      Fmt.string fmt "a Data index reached a context that cannot resolve one"
  | `Gather_index_out_of_range e -> pp_gather_index_out_of_range fmt e
  | `Index_not_exact_in_float n ->
      Fmt.pf fmt "index %d is not exactly representable as a float" n
  | `Unbound_reducer v -> Fmt.pf fmt "unbound reducer %a" Reduce_var.pp v

(* The recursion escapes and the public entry point converts once, rather than
     allocating an [Ok] per AST node per output pixel. The design permits this
     explicitly -- no state is exposed and the result is the defined value -- and
     it matters: this runs inside the grounding loop, which the transform
     verifier drives for every random-walk config.

     The token carries an already-built [Err.Error.t], so the backtrace is the
     one captured where the failure was DETECTED, not where it was caught.

   Polymorphic in the caller's own error row ['e], constrained [> index_error]:
   two real callers resolve a [Data] source at two different rows ([value]'s
   own [error], [Ground_eval]'s own error), and neither narrows down to
   [index_error] in general -- see the [Index.Data] design record. [resolve_data]
   is a RAW fetch (returns the stored [int64], not an already-validated [int]):
   normalization/bounds-checking happens exactly once, uniformly, here in the
   [Data] arm, via [resolve_gather_index] above, regardless of which caller's
   resolver produced the raw value.

   [~widen] lifts a bare [index_error] into the caller's own ['e] -- an
   explicit function parameter rather than an inline [(_ :> 'e)] coercion,
   because ['e] is still an abstract, unresolved row at the point [chk]/
   [fail_with] are defined: OCaml can only check a [:>] coercion against a
   fully known target type, so coercing directly to the not-yet-resolved ['e]
   here would force it to unify with whatever narrow type is being coerced,
   defeating the polymorphism this function exists to have. Every real caller
   supplies [~widen] as [(fun (e : index_error) -> (e :> 'e))] AT ITS OWN call
   site, where ['e] (that caller's concrete row) is already known, so the
   coercion there is the ordinary, unproblematic kind. *)
let eval_index (esc : ([> index_error ] as 'e) Err.Escape.t)
    ~(widen : index_error -> 'e) ~(output : int Coord.t)
    ~(reducers : Reduce_var.t -> int option)
    ~(resolve_data : Source.t -> int Coord.t -> (int64, 'e) Err.t)
    (e : 'r Index.t) : int =
  let fail_with (k : index_error) = Err.Escape.throw esc (widen k) in
  let chk (r : (int, [< index_error ]) Err.t) : int =
    Err.Escape.or_throw esc
      (Err.map_error widen (r :> (int, index_error) Err.t))
  in
  let rec go : type r. r Index.t -> int = function
    | Index.Add (a, b) -> chk (Checked.add (go a) (go b))
    | Index.Assume_position a -> go a
    | Index.Ceil_div_pos (a, d) -> chk (Checked.ceil_div_pos (go a) d)
    | Index.Clamp_low a -> Stdlib.max 0 (go a)
    | Index.Const n -> n
    | Index.Data (src, coord, extent) ->
        let raw =
          Err.Escape.or_throw esc (resolve_data src (Coord.map go coord))
        in
        Err.Escape.or_throw esc (resolve_gather_index raw ~extent)
    | Index.Floor_div_pos (a, d) -> chk (Checked.floor_div_pos (go a) d)
    | Index.Max (a, b) -> Stdlib.max (go a) (go b)
    | Index.Min (a, b) -> Stdlib.min (go a) (go b)
    | Index.Of_position i -> go i
    | Index.Output a -> Coord.get output a
    | Index.Reduce v -> (
        match reducers v with
        | Some i -> i
        | None -> fail_with (`Unbound_reducer v))
    | Index.Scale (k, a) -> chk (Checked.mul k (go a))
    | Index.Zero -> 0
  in
  go e

(* The public entry point: instantiates [eval_index] at ['e = index_error]
   with a resolver that can never legitimately be called -- a [Data] node
   cannot appear in a hand-built expression that never constructs one, so this
   stub is provably dead code for every such caller, the same way
   [kernel_elab.ml]'s existing catch-all already treats an unmatched
   [Index.t] case as inadmissible rather than crashing. *)
let index ~output ~reducers e =
  Err.Escape.with_escape @@ fun esc ->
  eval_index esc ~widen:Fun.id ~output ~reducers
    ~resolve_data:(fun _ _ -> Err.fail `Data_index_unexpected_here)
    e

(* Exactness is a ROUND TRIP, not a magnitude threshold. binary64 stops
     representing CONSECUTIVE integers above 2^53, but plenty of larger ones are
     exact -- 2^53+2, 2^54 and [min_int] all are, while 2^53+1, 2^54+2 and
     [max_int] are not. A "reject above 2^53" rule would be wrong in both
     directions.

     The [Sys.int_size] guard is not a micro-optimisation: every representable
     js_of_ocaml [int] is already exact in binary64, so without it this would put
     two emulated [Int64] conversions per [Value_of_index] back on the JS
     per-pixel path -- the exact allocation cost the [int] index domain was
     chosen to avoid -- in exchange for a check that can never fire there. *)
let float_of_index i =
  let f = Stdlib.float_of_int i in
  if Sys.int_size <= 53 then Err.return f
  else if Int64.equal (Int64.of_float f) (Int64.of_int i) then Err.return f
  else Err.fail (`Index_not_exact_in_float i)

(* Everything a value can fail on. [`Unknown_source] and [`Coord_out_of_range]
     are raised by the host's [load], not here -- the language knows nothing
     about what a source is. [`Data_source_wrong_format] is raised by the
     host's [load_index] (below): the bound tensor isn't the [I64] format a
     [Data] source requires. Its payload is a bare [string] (the format's
     name), not the host's own structured format type -- this module must not
     depend on [native], which is where that type lives; see the analogous
     choice for [Index.Data]'s own [extent] field. *)
type error =
  [ `Coord_out_of_range of Source.t * Axis.t * int * int Coord.t
  | `Data_source_wrong_format of string
  | index_error
  | Intrinsic.error
  | `Unknown_source of Source.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Coord_out_of_range (s, a, v, c) ->
      Fmt.pf fmt "%a[%a] out of range on axis %a: %d" Source.pp s
        (Coord.pp Fmt.int) c Axis.pp a v
  | `Data_source_wrong_format name ->
      Fmt.pf fmt "Data source is not an I64 tensor (format %s)" name
  | #index_error as e -> pp_index_error fmt e
  | #Intrinsic.error as e -> Intrinsic.pp_error fmt e
  | `Unknown_source s -> Fmt.pf fmt "unknown source %a" Source.pp s

module Env = struct
  (* The whole boundary between the language and its host. [Expr] supplies
       evaluated coordinates and consumes a working float; everything about
       storage format, quantization and tensor ownership lives on the other
       side. This is what keeps the library independent of [native].
     [load_index] is the same boundary for a [Data] source's raw integer
     read -- the host resolves the bound tensor and its format; this module
     only threads the coordinate and normalizes/bounds-checks the result
     (in [eval_index]'s own [Data] arm) once it comes back. *)
  type t = {
    load : Source.t -> int Coord.t -> (float, error) Err.t;
    load_index : Source.t -> int Coord.t -> (int64, error) Err.t;
  }
end

(* Every value-level failure arrives as a [Err.t] from somewhere else -- the
     host's [load], the intrinsic geometry, the index evaluator -- so there is
     no direct-throw helper to go with [vchk]. It stays at module level so its
     ['a] generalises; an annotation on a [let] inside [value] would not. *)
let vchk esc : ('a, [< error ]) Err.t -> 'a = function
  | Ok v -> v
  | Error e -> Err.Escape.throw_error esc (e :> error Err.Error.t)

let value (env : Env.t) ~output e =
  Err.Escape.with_escape @@ fun esc ->
  let vchk r = vchk esc r in
  (* [eval_index] is polymorphic in the caller's error row, and [env.load_index]
     already sits at exactly this frame's own [error] row -- so [esc] is
     passed directly, with no narrowing view needed (unlike before [Data]
     existed, when the index evaluator's row was strictly narrower than this
     frame's). *)
  let idx reducers i =
    eval_index esc
      ~widen:(fun (e : index_error) -> (e :> error))
      ~output ~reducers ~resolve_data:env.Env.load_index i
  in
  let rec go reducers (e : Value.t) : float =
    match e with
    | Value.Binary (op, a, b) ->
        Value.apply_binary op (go reducers a) (go reducers b)
    | Value.Const x -> x
    | Value.Intrinsic i -> intrinsic reducers i
    | Value.Load (s, c) -> vchk (env.Env.load s (Coord.map (idx reducers) c))
    | Value.Reduce r ->
        let lo = idx reducers r.Reduction.lo
        and hi = idx reducers r.Reduction.hi in
        let combine, init =
          match r.Reduction.kind with
          | Reduction.Max -> (Max_op.apply Max_op.Float_max, Float.neg_infinity)
          | Reduction.Sum -> (( +. ), 0.)
        in
        (* The ordered half-open left fold the denotation specifies. Same seed
             and same association as the engine's own reduction -- a rewrite that
             reassociated this would change the answer, not just its shape. *)
        let rec fold i acc =
          if i >= hi then acc
          else
            let bound v =
              if Reduce_var.equal v r.Reduction.var then Some i else reducers v
            in
            fold (i + 1) (combine acc (go bound r.Reduction.body))
        in
        fold lo init
    | Value.Round_f32 a ->
        (* Convert to binary32 and widen back. The one value expression that
             changes a value without being arithmetic. *)
        Int32.float_of_bits (Int32.bits_of_float (go reducers a))
    (* Only the SELECTED branch is evaluated -- the other may divide by zero
         or read out of bounds, and guarding is what the caller built it for. *)
    | Value.Select (c, a, b) ->
        if guard reducers c then go reducers a else go reducers b
    | Value.Unary (op, a) -> Value.apply_unary op (go reducers a)
    | Value.Value_of_index i -> vchk (float_of_index (idx reducers i))
  and guard reducers = function
    | Bool.Index_eq (a, b) -> Int.equal (idx reducers a) (idx reducers b)
    | Bool.Value_lt (a, b) -> go reducers a < go reducers b
  and intrinsic reducers (Intrinsic.Max_pool d as i) =
    let open Intrinsic.Max_pool in
    let at a = idx reducers (Coord.get d.out a) in
    let w = vchk (Intrinsic.window i ~out_h:(at Axis.H) ~out_w:(at Axis.W)) in
    let read ih iw =
      vchk
        (env.Env.load d.source
           (Coord.of_fn (fun a ->
                if a = Axis.H then ih else if a = Axis.W then iw else at a)))
    in
    (* Value and index advance TOGETHER under one predicate. Updating them
         separately is how they fell out of step originally, which is why
         [Max_op.pool_better] is shared rather than open-coded. An ordinary tie
         keeps the incumbent; a NaN re-triggers, so the LAST NaN wins. *)
    let rec rows ih best best_ix =
      if ih >= w.Intrinsic.Window.hhi then (best, best_ix)
      else cols ih w.Intrinsic.Window.wlo best best_ix
    and cols ih iw best best_ix =
      if iw >= w.Intrinsic.Window.whi then rows (ih + 1) best best_ix
      else
        let v = read ih iw in
        let best, best_ix =
          if Max_op.pool_better ~best ~value:v then
            (v, vchk (Intrinsic.flat_index i ~ih ~iw))
          else (best, best_ix)
        in
        cols ih (iw + 1) best best_ix
    in
    let best, best_ix = rows w.Intrinsic.Window.hlo Float.neg_infinity 0 in
    match d.result with Value -> best | Index -> vchk (float_of_index best_ix)
  in
  go (fun _ -> None) e
