(* Only what index evaluation can actually raise. [Eval.error] widens this in
     stage 4, once intrinsics and loads exist -- a stage cannot publish a type
     equation naming a module that arrives in the next one. *)
type index_error =
  [ Checked.error
  | `Index_not_exact_in_float of int
  | `Unbound_reducer of Reduce_var.t ]

let pp_index_error fmt : [< index_error ] -> unit = function
  | #Checked.error as e -> Checked.pp_error fmt e
  | `Index_not_exact_in_float n ->
      Fmt.pf fmt "index %d is not exactly representable as a float" n
  | `Unbound_reducer v -> Fmt.pf fmt "unbound reducer %a" Reduce_var.pp v

(* The recursion escapes and the public entry point converts once, rather than
     allocating an [Ok] per AST node per output pixel. The design permits this
     explicitly -- no state is exposed and the result is the defined value -- and
     it matters: this runs inside the grounding loop, which the transform
     verifier drives for every random-walk config.

     The token carries an already-built [Err.Error.t], so the backtrace is the
     one captured where the failure was DETECTED, not where it was caught. It is
     an [index_error] token even though [value] below evaluates indices under
     its own frame at the wider [error] row: a token is invariant in its
     payload, so [value] passes [Err.Escape.map] of its own rather than sharing
     one directly. That derived token exits the same frame and allocates no
     second generative exception, which is what keeps a nested [with_escape] off
     the per-pixel path. *)
let eval_index (type r) (esc : index_error Err.Escape.t) ~(output : int Coord.t)
    ~(reducers : Reduce_var.t -> int option) (e : r Index.t) : int =
  let fail_with (k : index_error) = Err.Escape.throw esc k in
  let chk : (int, [< index_error ]) Err.t -> int = function
    | Ok v -> v
    | Error e -> Err.Escape.throw_error esc (e :> index_error Err.Error.t)
  in
  let rec go : type r. r Index.t -> int = function
    | Index.Add (a, b) -> chk (Checked.add (go a) (go b))
    | Index.Assume_position a -> go a
    | Index.Ceil_div_pos (a, d) -> chk (Checked.ceil_div_pos (go a) d)
    | Index.Clamp_low a -> Stdlib.max 0 (go a)
    | Index.Const n -> n
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

let index ~output ~reducers e =
  Err.Escape.with_escape @@ fun esc -> eval_index esc ~output ~reducers e

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
     about what a source is. *)
type error =
  [ `Coord_out_of_range of Source.t * Axis.t * int * int Coord.t
  | index_error
  | Intrinsic.error
  | `Unknown_source of Source.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Coord_out_of_range (s, a, v, c) ->
      Fmt.pf fmt "%a[%a] out of range on axis %a: %d" Source.pp s
        (Coord.pp Fmt.int) c Axis.pp a v
  | #index_error as e -> pp_index_error fmt e
  | #Intrinsic.error as e -> Intrinsic.pp_error fmt e
  | `Unknown_source s -> Fmt.pf fmt "unknown source %a" Source.pp s

module Env = struct
  (* The whole boundary between the language and its host. [Expr] supplies
       evaluated coordinates and consumes a working float; everything about
       storage format, quantization and tensor ownership lives on the other
       side. This is what keeps the library independent of [native]. *)
  type t = { load : Source.t -> int Coord.t -> (float, error) Err.t }
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
  (* The index evaluator's row is narrower than this frame's, so it gets a
       VIEW of this token rather than one of its own. *)
  let index_esc = Err.Escape.map (fun e -> (e :> error)) esc in
  let idx reducers i = eval_index index_esc ~output ~reducers i in
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
