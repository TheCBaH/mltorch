(* The tail-call conversion's Stage 5 [eval_hybrid] candidate (see .ai/): a
   complete DIRECT evaluator (ordinary OCaml recursion, [go]/[guard]/
   [eval_scan_at] shaped exactly like [eval.ml]'s native reference) with its
   own [cutoff], falling back to [Eval_machine_reuse.run] -- the fourth and
   last of the four candidates the design record calls for. Copied into
   [expr_internal_js]/[expr_internal_mel] the same way as the other three,
   from the same [js/probe/] origin, never linked natively or exposed
   through the public [Expr] API. [open Eval_candidates] for [pool_tag]'s
   constructors and [Eval_state]/[Guard_state]/[Float_result]/[Bool_result],
   the handoff protocol's own vocabulary.

   Below [cutoff], [go]/[guard] recurse exactly like the native reference:
   real, unbounded-looking OCaml recursion, deliberately -- this candidate's
   whole premise is that a SHALLOW subtree is cheaper evaluated directly
   than through any of the other three candidates' machinery, and every
   real backend can trampoline a bounded amount of ordinary recursion
   safely regardless of whether it is provably a tail call (the same
   "documented bounded overhead" argument [eval_trampoline_delayed]'s
   [threshold] makes, here expressed as a plain recursion-depth counter
   instead of a bounce budget). At [cutoff], [go]/[guard] stop recursing
   and hand the REMAINING subtree to [Eval_machine_reuse.run] instead,
   sharing this call's own [esc]/[cleanups]/[local_at_ref] rather than
   starting fresh ones -- "at any hybrid cutoff, transfer the current
   reducers and active local resolver into the machine; preserve active
   cleanup state across the handoff", per the design record. Once handed
   off, a subtree stays in the machine for the rest of its own evaluation:
   the machine is a complete evaluator of the full grammar, so there is
   never a reason to hand a partially-machine-evaluated subtree back to
   direct recursion. [run]'s own cleanup-on-exception handler is deliberately
   NOT re-entered here: this file's single top-level [try]/[with], wrapping
   the root [go] call, is the one place cleanups run, exactly as the design
   record's cleanup protocol requires -- a handoff exception unwinds through
   this candidate's own direct-recursion frames like any other exception and
   reaches that same handler. *)

open Eval_common
open Eval_candidates

(* [skip_cleanup]: same contract as [Eval_candidates.eval_machine]'s own
   (see its doc comment); normal callers never pass it. Applies only to
   THIS candidate's own top-level handler -- a handoff to [machine_run]
   never re-enters [Eval_machine_reuse.run]'s handler, per this file's own
   top comment. *)
let eval_hybrid ~cutoff ?(local = fun _ -> None) ?(local_at = fun _ _ -> None)
    ?(local_i64 = fun _ -> None) ?(local_at_i64 = fun _ _ -> None) ?scan
    ?scan_meter ?(reducer = []) ?(on_reduction = fun () -> ())
    ?skip_cleanup (env : Env.t) ~output e =
  if cutoff < 0 then invalid_arg "eval_hybrid: cutoff must be >= 0";
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
     [loop] over a nullary tag (Stage 4), needed regardless of [cutoff] --
     [Max_pool]'s own row/column sweep is never itself a candidate for a
     machine handoff, since it doesn't recurse through [go]/[guard]. *)
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
  let machine_run seed =
    Eval_machine_reuse.run ~esc ~env ~output ~scan ~scan_meter ~local
      ~local_at_ref ~local_i64 ~local_at_i64 ~cleanups ~run_top_cleanup
      ~on_reduction seed
  in
  let rec go depth reducers (e : float Value.t) : float =
    if depth >= cutoff then
      match machine_run (Eval_state (e, reducers)) with
      | Float_result v -> v
      | _ -> assert false
    else
      let depth = depth + 1 in
      match e with
      | Value.Binary (op, a, b) ->
          Value.apply_binary op (go depth reducers a) (go depth reducers b)
      | Value.Const x -> x
      | Value.I64_to_float a -> Int64.to_float (eval_i64 depth reducers a)
      | Value.Intrinsic i -> intrinsic reducers i
      | Value.Local v -> (
          match local v with
          | Some x -> x
          | None -> Err.Escape.throw esc (`Unbound_local v))
      | Value.Local_at (v, i) -> (
          match !local_at_ref v (idx reducers i) with
          | Some x -> x
          | None -> Err.Escape.throw esc (`Unbound_local v))
      | Value.Local_scan_at (v, row_i, lane_i) ->
          let row = idx reducers row_i and lane = idx reducers lane_i in
          vchk (Err.map_error scan_error (scan v ~row ~lane))
      | Value.Load (s, c) -> vchk (env.Env.load s (Coord.map (idx reducers) c))
      | Value.Reduce r ->
          let lo = idx reducers r.Reduction.lo
          and hi = idx reducers r.Reduction.hi in
          (match r.Reduction.kind with
          | Reduction.Max | Reduction.Sum ->
              let combine, init =
                match r.Reduction.kind with
                | Reduction.Max ->
                    (Max_op.apply Max_op.Float_max, Float.neg_infinity)
                | Reduction.Sum -> (( +. ), 0.)
                | Reduction.Argmax_index | Reduction.Argmax_value ->
                    assert false
              in
              let rec fold i acc =
                if i >= hi then acc
                else
                  let bound v =
                    if Reduce_var.equal v r.Reduction.var then Some i
                    else reducers v
                  in
                  on_reduction ();
                  (fold [@tailcall]) (i + 1)
                    (combine acc (go depth bound r.Reduction.body))
              in
              fold lo init
          | Reduction.Argmax_index | Reduction.Argmax_value ->
              (* One predicate advances value and index together
                 ([Max_op.pool_better], the same convention
                 [Intrinsic.Max_pool]'s own paired value/index output uses),
                 so the two outputs cannot fall out of step. *)
              let rec fold i best best_i =
                if i >= hi then (best, best_i)
                else
                  let bound v =
                    if Reduce_var.equal v r.Reduction.var then Some i
                    else reducers v
                  in
                  let value = go depth bound r.Reduction.body in
                  on_reduction ();
                  let best, best_i =
                    if Max_op.pool_better ~best ~value then (value, i)
                    else (best, best_i)
                  in
                  (fold [@tailcall]) (i + 1) best best_i
              in
              let best, best_i = fold lo Float.neg_infinity lo in
              (match r.Reduction.kind with
              | Reduction.Argmax_value -> best
              | Reduction.Argmax_index -> vchk (float_of_index best_i)
              | Reduction.Max | Reduction.Sum -> assert false))
      | Value.Round_f32 a ->
          Int32.float_of_bits (Int32.bits_of_float (go depth reducers a))
      | Value.Scan_at (s, row_i, lane_i) ->
          (eval_scan_at [@tailcall]) depth reducers s row_i lane_i
      | Value.Select (c, a, b) ->
          if guard depth reducers c then (go [@tailcall]) depth reducers a
          else (go [@tailcall]) depth reducers b
      | Value.Unary (op, a) -> Value.apply_unary op (go depth reducers a)
      | Value.Value_of_index i -> vchk (float_of_index (idx reducers i))
  and guard depth reducers (b : Bool.t) : bool =
    if depth >= cutoff then
      match machine_run (Guard_state (b, reducers)) with
      | Bool_result r -> r
      | _ -> assert false
    else
      let depth = depth + 1 in
      match b with
      | Bool.Index_eq (a, b) -> Int.equal (idx reducers a) (idx reducers b)
      | Bool.Value_eq (a, b) -> go depth reducers a = go depth reducers b
      | Bool.Value_lt (a, b) -> go depth reducers a < go depth reducers b
      | Bool.I64_eq (a, b) ->
          Int64.equal (eval_i64 depth reducers a) (eval_i64 depth reducers b)
      | Bool.I64_lt (a, b) ->
          Int64.compare (eval_i64 depth reducers a) (eval_i64 depth reducers b)
          < 0
  (* Not delegated to [Value.eval_i64]'s callback-based definition, unlike
     every other caller of it: that definition has no [depth] of its own (see
     its doc comment), so [I64_binary]/[Select]'s own nesting on the int64
     side needs the SAME cutoff/machine-handoff [go]/[guard] give the
     float/bool grammar, inlined here rather than threaded through a shared
     helper. [Float_to_i64]'s float operand still flows through [go], which
     is already cutoff-aware. *)
  and eval_i64 depth reducers (a : int64 Value.t) : int64 =
    if depth >= cutoff then
      match machine_run (Eval_i64_state (a, reducers)) with
      | I64_result v -> v
      | _ -> assert false
    else
      let depth = depth + 1 in
      match a with
      | Value.I64_const x -> x
      | Value.I64_load (s, c) ->
          vchk (env.Env.load_index s (Coord.map (idx reducers) c))
      | Value.I64_local v -> (
          match local_i64 v with
          | Some x -> x
          | None -> Err.Escape.throw esc (`Unbound_local v))
      | Value.I64_local_at (v, i) -> (
          match local_at_i64 v (idx reducers i) with
          | Some x -> x
          | None -> Err.Escape.throw esc (`Unbound_local v))
      | Value.I64_of_index i -> Int64.of_int (idx reducers i)
      | Value.I64_binary (op, x, y) ->
          Value.apply_i64_binary op (eval_i64 depth reducers x)
            (eval_i64 depth reducers y)
      | Value.Float_to_i64 x -> vchk (Value.i64_of_float (go depth reducers x))
      | Value.Select (c, x, y) ->
          if guard depth reducers c then (eval_i64 [@tailcall]) depth reducers x
          else (eval_i64 [@tailcall]) depth reducers y
  and eval_scan_at depth reducers s row_i lane_i =
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
           (Lane_out_of_range { Scan_bounds.projection; extent = s.Scan.width }))
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
      let init_row () =
        Array.init s.Scan.width (fun l ->
            let bound v =
              if Reduce_var.equal v s.Scan.lane then Some l else reducers v
            in
            go depth bound s.Scan.init)
      in
      let next_row ~step prev_row =
        (local_at_ref :=
           fun v pos ->
             if Local_var.equal v s.Scan.prev then Some prev_row.(pos)
             else saved_local_at v pos);
        Array.init s.Scan.width (fun l ->
            vchk
              (Err.map_error scan_meter_error (Scan_meter.charge_update meter));
            let bound v =
              if Reduce_var.equal v s.Scan.lane then Some l
              else if Reduce_var.equal v s.Scan.step then Some step
              else reducers v
            in
            go depth bound s.Scan.update)
      in
      let rec run_rows r prev_row =
        if r = row then prev_row.(lane)
        else (run_rows [@tailcall]) (r + 1) (next_row ~step:r prev_row)
      in
      let result = run_rows 0 (init_row ()) in
      run_top_cleanup ();
      result
  in
  try go 0 init_reducers e
  with exn ->
    let bt = capture_backtrace () in
    (match skip_cleanup with
    | None -> List.iter (fun f -> f ()) !cleanups
    | Some skip ->
        List.iteri (fun i f -> if not (skip i) then f ()) !cleanups);
    cleanups := [];
    reraise exn bt
