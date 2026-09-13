(* The tail-call conversion's Stage 5 [eval_trampoline_delayed] candidate
   (see .ai/): CPS with periodic bounces, one of the four candidate
   evaluators for [Expr_internal]'s full [float Value.t]/[Bool.t] grammar. Split
   out of [eval_candidates.ml] (which keeps the shared [pending work] types
   and [eval_machine]) once the combined file passed the file-size cap --
   copied into [expr_internal_js]/[expr_internal_mel] the same way, from the
   same [js/probe/] origin, never linked natively or exposed through the
   public [Expr] API. [open Eval_candidates] for [pool_tag]'s constructors,
   the one thing this candidate borrows from there: its own [intrinsic] is
   otherwise a verbatim, self-contained copy of [eval_machine]'s.

   CPS: every transition takes an explicit continuation instead of returning
   through an OCaml call. A continuation is an arbitrary closure, so calling
   one is a call through an unknown target -- js_of_ocaml's own tail-call
   trampoline only covers a STATICALLY KNOWN (mutually) recursive function
   group (see .ai/ and [eval.ml]'s note on [eval_scan_at] staying out of
   [go]'s [and] group for exactly this reason), and Stage 4 found Melange
   rejects even that known-group case for a two-function mutual pair. This
   candidate therefore never trusts ANY backend to optimize a hop for it,
   whether the hop is a continuation call or a same-name recursive call:
   [go]/[guard] and every local recursive helper below check their own
   [depth] parameter on entry and, once it reaches [threshold], return
   [Bounce] -- a plain closure value, not a call -- instead of proceeding.
   Returning a value costs nothing extra to unwind (it is exactly what an
   un-optimized call chain does on the way back out), so control returns
   cleanly to [run_trampoline]'s own self-recursive loop (a single named
   function calling itself -- the one shape every backend has trampolined
   safely since Stage 0), which invokes the bounced closure at a fresh
   [depth = 0]. [resume] is the analogous gate for handing a value to an
   EXTERNALLY supplied continuation ([k]/a scan's row-completion callback):
   the same check, at the one place this file calls through such a closure
   instead of a named function it defined itself.

   [threshold = 1] bounces after every single hop -- maximally safe, and the
   closest in spirit to [eval_machine]'s per-node granularity, though
   [eval_trampoline_delayed] pays a closure allocation per hop rather than a
   frame-list cons. A higher [threshold] allows that many real (uncounted by
   any backend, just ordinary calls) hops in a row before bouncing, trading
   a documented bounded amount of host stack for fewer bounces -- the
   "maximum segment depth bounded by the configured threshold plus
   documented fixed overhead" the design record's stack-safety section
   requires as evidence for this candidate's shape. *)

open Eval_common
open Eval_candidates

type 'a bounce = Done of 'a | Bounce of (unit -> 'a bounce)

let rec run_trampoline : 'a. 'a bounce -> 'a = function
  | Done v -> v
  | Bounce f -> (run_trampoline [@tailcall]) (f ())

(* [skip_cleanup]: the cleanup protocol's private test-only leak-injection
   hook, same contract as [Eval_candidates.eval_machine]'s own (see its doc
   comment); normal callers never pass it. *)
let eval_trampoline_delayed ~threshold ?(local = fun _ -> None)
    ?(local_at = fun _ _ -> None) ?scan ?scan_meter ?(reducer = [])
    ?(on_reduction = fun () -> ()) ?skip_cleanup (env : Env.t) ~output e =
  if threshold < 1 then
    invalid_arg "eval_trampoline_delayed: threshold must be >= 1";
  Err.Escape.with_escape @@ fun esc ->
  let vchk r = vchk esc r in
  let init_reducers w =
    List.find_map
      (fun (v, p) -> if Reduce_var.equal w v then Some p else None)
      reducer
  in
  let scan : scan_reader =
    match scan with
    | Some reader -> reader
    | None -> fun id ~row:_ ~lane:_ -> Err.fail (Unknown_local id)
  in
  let idx reducers i =
    eval_index esc
      ~widen:(fun (e : index_error) -> (e :> error))
      ~output ~reducers ~resolve_data:env.Env.load_index i
  in
  let local_at_ref = ref local_at in
  let cleanups : (unit -> unit) list ref = ref [] in
  let run_top_cleanup () =
    match !cleanups with
    | f :: rest ->
        cleanups := rest;
        f ()
    | [] -> assert false
  in
  (* Verbatim copy of [eval_machine]'s [intrinsic]: already one self-recursive
     [loop] over a nullary tag (Stage 4), so it needs no [depth]/[Bounce]
     handling of its own -- it never hops through [go]/[guard]/a
     continuation. *)
  let intrinsic reducers (Intrinsic.Max_pool d as i) =
    let open Intrinsic.Max_pool in
    let at a = idx reducers (Coord.get d.out a) in
    let w = vchk (Intrinsic.window i ~out_h:(at Axis.H) ~out_w:(at Axis.W)) in
    let read ih iw =
      vchk
        (env.Env.load d.source
           (Coord.of_fn (fun a ->
                if a = Axis.H then ih else if a = Axis.W then iw else at a)))
    in
    let rec loop tag ih iw best best_ix =
      match tag with
      | Rows_tag ->
          if ih >= w.Intrinsic.Window.hhi then (best, best_ix)
          else
            (loop [@tailcall]) Cols_tag ih w.Intrinsic.Window.wlo best best_ix
      | Cols_tag ->
          if iw >= w.Intrinsic.Window.whi then
            (loop [@tailcall]) Rows_tag (ih + 1) iw best best_ix
          else
            let v = read ih iw in
            let best, best_ix =
              if Max_op.pool_better ~best ~value:v then
                (v, vchk (Intrinsic.flat_index i ~ih ~iw))
              else (best, best_ix)
            in
            (loop [@tailcall]) Cols_tag ih (iw + 1) best best_ix
    in
    let best, best_ix =
      loop Rows_tag w.Intrinsic.Window.hlo 0 Float.neg_infinity 0
    in
    match d.result with Value -> best | Index -> vchk (float_of_index best_ix)
  in
  (* The sole gate for handing a value to a caller-supplied continuation.
     Below [threshold], an ordinary (potentially stack-consuming) call;
     at [threshold], a returned [Bounce] value instead, unwinding back to
     [run_trampoline] before resuming at [depth = 0]. Polymorphic in the
     resumed payload: [go]'s [float] results, [guard]'s [bool] results, and a
     completed scan row's [float array] all resume through this one gate. *)
  let resume : 'a. int -> (int -> 'a -> float bounce) -> 'a -> float bounce =
   fun depth k v ->
    if depth >= threshold then Bounce (fun () -> k 0 v) else k (depth + 1) v
  in
  let rec go reducers depth (e : float Value.t) (k : int -> float -> float bounce) :
      float bounce =
    if depth >= threshold then
      Bounce (fun () -> (go [@tailcall]) reducers 0 e k)
    else
      let depth = depth + 1 in
      match e with
      | Value.Const x -> resume depth k x
      | Value.Local v -> (
          match local v with
          | Some x -> resume depth k x
          | None -> Err.Escape.throw esc (`Unbound_local v))
      | Value.Local_at (v, i) -> (
          match !local_at_ref v (idx reducers i) with
          | Some x -> resume depth k x
          | None -> Err.Escape.throw esc (`Unbound_local v))
      | Value.Local_scan_at (v, row_i, lane_i) ->
          let row = idx reducers row_i and lane = idx reducers lane_i in
          resume depth k (vchk (Err.map_error scan_error (scan v ~row ~lane)))
      | Value.Load (s, c) ->
          resume depth k (vchk (env.Env.load s (Coord.map (idx reducers) c)))
      | Value.Value_of_index i ->
          resume depth k (vchk (float_of_index (idx reducers i)))
      | Value.Intrinsic i -> resume depth k (intrinsic reducers i)
      (* Evaluation order is backend-measured (see [order_probe] goldens
         under .ai/): jsoo evaluates [b] before [a] at this call site's
         [eval.ml] original, Melange the reverse. [av]/[bv] keep their
         original-slot names regardless of which [go] call runs first, so
         the final [apply_binary] call needs no reordering of its own. *)
      | Value.Binary (op, a, b) ->
#if defined MELANGE_BACKEND
          go reducers depth a (fun depth av ->
              go reducers depth b (fun depth bv ->
                  resume depth k (Value.apply_binary op av bv)))
#else
          go reducers depth b (fun depth bv ->
              go reducers depth a (fun depth av ->
                  resume depth k (Value.apply_binary op av bv)))
#endif
      | Value.Unary (op, a) ->
          go reducers depth a (fun depth av ->
              resume depth k (Value.apply_unary op av))
      | Value.Round_f32 a ->
          go reducers depth a (fun depth av ->
              resume depth k (Int32.float_of_bits (Int32.bits_of_float av)))
      | Value.Select (c, a, b) ->
          guard reducers depth c (fun depth cond ->
              if cond then go reducers depth a k else go reducers depth b k)
      | Value.Reduce r -> (
          let lo = idx reducers r.Reduction.lo
          and hi = idx reducers r.Reduction.hi in
          match r.Reduction.kind with
          | Reduction.Max | Reduction.Sum ->
              let combine, init =
                match r.Reduction.kind with
                | Reduction.Max ->
                    (Max_op.apply Max_op.Float_max, Float.neg_infinity)
                | Reduction.Sum -> (( +. ), 0.)
                | Reduction.Argmax_index | Reduction.Argmax_value ->
                    assert false
              in
              if lo >= hi then resume depth k init
              else
                (* Replaces its own [i]/[acc] on every iteration -- an
                   [and]-bound sibling would make this a two-function mutual
                   pair (the exact shape Stage 4 removed from [intrinsic]); a
                   LOCAL [let rec], entered only through [go]'s continuation
                   for the reduction's body, keeps it self-recursive instead. *)
                let rec reduce_iterate depth i acc =
                  if depth >= threshold then
                    Bounce (fun () -> (reduce_iterate [@tailcall]) 0 i acc)
                  else
                    let depth = depth + 1 in
                    on_reduction ();
                    let bound v =
                      if Reduce_var.equal v r.Reduction.var then Some i
                      else reducers v
                    in
                    go bound depth r.Reduction.body (fun depth v ->
                        let acc = combine acc v in
                        if i + 1 >= hi then resume depth k acc
                        else (reduce_iterate [@tailcall]) depth (i + 1) acc)
                in
                reduce_iterate depth lo init
          | Reduction.Argmax_index | Reduction.Argmax_value ->
              (* One predicate advances value and index together
                 ([Max_op.pool_better], the same convention
                 [Intrinsic.Max_pool]'s own paired value/index output uses),
                 so the two outputs cannot fall out of step. *)
              if lo >= hi then
                resume depth k
                  (match r.Reduction.kind with
                  | Reduction.Argmax_index -> vchk (float_of_index lo)
                  | Reduction.Argmax_value | Reduction.Max | Reduction.Sum ->
                      Float.neg_infinity)
              else
                let rec reduce_iterate depth i best best_i =
                  if depth >= threshold then
                    Bounce
                      (fun () -> (reduce_iterate [@tailcall]) 0 i best best_i)
                  else
                    let depth = depth + 1 in
                    on_reduction ();
                    let bound v =
                      if Reduce_var.equal v r.Reduction.var then Some i
                      else reducers v
                    in
                    go bound depth r.Reduction.body (fun depth v ->
                        let best, best_i =
                          if Max_op.pool_better ~best ~value:v then (v, i)
                          else (best, best_i)
                        in
                        if i + 1 >= hi then
                          resume depth k
                            (match r.Reduction.kind with
                            | Reduction.Argmax_value -> best
                            | Reduction.Argmax_index ->
                                vchk (float_of_index best_i)
                            | Reduction.Max | Reduction.Sum -> assert false)
                        else
                          (reduce_iterate [@tailcall]) depth (i + 1) best
                            best_i)
                in
                reduce_iterate depth lo Float.neg_infinity lo)
      | Value.Scan_at (s, row_i, lane_i) ->
          let row = idx reducers row_i and lane = idx reducers lane_i in
          let projection = { Scan_projection.local = None; row; lane } in
          if row < 0 || row > s.Scan.steps then
            Err.Escape.throw esc
              (`Scan_projection
                 (Row_out_of_range
                    { Scan_bounds.projection; extent = s.Scan.steps + 1 }))
          else if lane < 0 || lane >= s.Scan.width then
            Err.Escape.throw esc
              (`Scan_projection
                 (Lane_out_of_range
                    { Scan_bounds.projection; extent = s.Scan.width }))
          else
            let meter =
              match scan_meter with
              | Some m -> m
              | None -> Err.Escape.throw esc `Scan_meter_required
            in
            vchk
              (Err.map_error scan_meter_error
                 (Scan_meter.reserve meter ~width:s.Scan.width));
            let saved_local_at = !local_at_ref in
            cleanups :=
              (fun () ->
                local_at_ref := saved_local_at;
                Scan_meter.release meter ~width:s.Scan.width)
              :: !cleanups;
            (* [scan_fill_row]/[scan_fill_lane] are LOCAL [let rec ... and
               ...], scoped to this one [Scan_at] evaluation and entered only
               through continuations -- the same reasoning as
               [reduce_iterate] above: each checks its own [depth] on entry
               rather than trusting a call from within a closure to stay
               cheap. *)
            let rec scan_fill_row depth filling_row =
              if depth >= threshold then
                Bounce (fun () -> (scan_fill_row [@tailcall]) 0 filling_row)
              else
                let depth = depth + 1 in
                let body, bound_extra =
                  if filling_row = 0 then (s.Scan.init, fun v -> reducers v)
                  else
                    let step = filling_row - 1 in
                    ( s.Scan.update,
                      fun v ->
                        if Reduce_var.equal v s.Scan.step then Some step
                        else reducers v )
                in
                let cur_row = Array.make s.Scan.width 0. in
                scan_fill_lane depth ~filling_row ~body ~bound_extra cur_row 0
                  (fun depth completed_row ->
                    if filling_row = row then begin
                      run_top_cleanup ();
                      resume depth k completed_row.(lane)
                    end
                    else begin
                      (local_at_ref :=
                         fun v pos ->
                           if Local_var.equal v s.Scan.prev then
                             Some completed_row.(pos)
                           else saved_local_at v pos);
                      (scan_fill_row [@tailcall]) depth (filling_row + 1)
                    end)
            and scan_fill_lane depth ~filling_row ~body ~bound_extra cur_row l
                row_done =
              if depth >= threshold then
                Bounce
                  (fun () ->
                    (scan_fill_lane [@tailcall]) 0 ~filling_row ~body
                      ~bound_extra cur_row l row_done)
              else
                let depth = depth + 1 in
                if filling_row > 0 then
                  vchk
                    (Err.map_error scan_meter_error
                       (Scan_meter.charge_update meter));
                let bound v =
                  if Reduce_var.equal v s.Scan.lane then Some l
                  else bound_extra v
                in
                go bound depth body (fun depth v ->
                    cur_row.(l) <- v;
                    if l + 1 < s.Scan.width then
                      (scan_fill_lane [@tailcall]) depth ~filling_row ~body
                        ~bound_extra cur_row (l + 1) row_done
                    else resume depth row_done cur_row)
            in
            scan_fill_row depth 0
  and guard reducers depth (b : Bool.t) (k : int -> bool -> float bounce) :
      float bounce =
    if depth >= threshold then
      Bounce (fun () -> (guard [@tailcall]) reducers 0 b k)
    else
      let depth = depth + 1 in
      match b with
      | Bool.Index_eq (a, b) ->
          resume depth k (Int.equal (idx reducers a) (idx reducers b))
      (* Same backend-measured order as [Binary] above. *)
      | Bool.Value_lt (a, b) ->
#if defined MELANGE_BACKEND
          go reducers depth a (fun depth av ->
              go reducers depth b (fun depth bv -> resume depth k (av < bv)))
#else
          go reducers depth b (fun depth bv ->
              go reducers depth a (fun depth av -> resume depth k (av < bv)))
#endif
  in
  try run_trampoline (go init_reducers 0 e (fun _ x -> Done x))
  with exn ->
    let bt = capture_backtrace () in
    (match skip_cleanup with
    | None -> List.iter (fun f -> f ()) !cleanups
    | Some skip ->
        List.iteri (fun i f -> if not (skip i) then f ()) !cleanups);
    cleanups := [];
    reraise exn bt
