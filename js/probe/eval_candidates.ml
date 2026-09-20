(* Stack-safe candidate drivers for [Expr_internal]'s evaluator, developed for
   the tail-call conversion's JS backends (Stage 5; see .ai/). Copied
   verbatim into [expr_internal_js] and [expr_internal_mel] -- never linked
   natively, and never exposed through the public [Expr]/[Expr_api] surface:
   [Value.t]/[Bool.t] here are [Expr_internal]'s own non-private
   representations, the same ones [eval.ml] itself matches on. [open
   Eval_common] for [error]/[Env.t]/[vchk]/[eval_index]/[index_error]/
   [scan_error]/[scan_reader]/[scan_meter_error].

   This file holds the shared "pending work" types below and the first
   candidate, [eval_machine] (the explicit list-frame machine). The other
   three candidates -- [eval_trampoline_delayed] (CPS with periodic
   bounces), [eval_machine_reuse] (array/mutation-based storage reuse) and
   the still-unimplemented direct/machine hybrid -- live in sibling files
   ([eval_trampoline_delayed.ml], [eval_machine_reuse.ml]) once keeping them
   all in one file passed the repo's file-size cap; both [open
   Eval_candidates] for the types they share with [eval_machine] rather than
   duplicating them. *)

open Eval_common

(* Melange has no [caml_restore_raw_backtrace]: calling
   [Printexc.raise_with_backtrace] there throws "not polyfilled by Melange
   yet" at RUNTIME, not build time -- a distinct failure mode from jsoo,
   which implements it. Only reached once an exception genuinely propagates
   to a candidate's own top-level handler, which no test before the cleanup
   protocol's negative controls ever forced on Melange; see .ai/. Wrapping
   COMPLETE top-level definitions in a cppo conditional (never splitting one
   expression across the boundary, unlike an earlier attempt here) is what
   [ocamlformat] can parse -- the same shape [eval.ml]'s own JS_BACKEND
   conditional uses. [capture_backtrace] must still run as the
   very FIRST statement of an exception handler, before any cleanup code
   that could itself disturb the runtime's backtrace buffer; [reraise] is
   only ever the LAST. *)
#if defined MELANGE_BACKEND

type captured_backtrace = unit

let capture_backtrace () : captured_backtrace = ()
let reraise exn (_ : captured_backtrace) = raise exn

#else

type captured_backtrace = Printexc.raw_backtrace

let capture_backtrace () : captured_backtrace = Printexc.get_raw_backtrace ()
let reraise exn (bt : captured_backtrace) = Printexc.raise_with_backtrace exn bt

#endif

(* ---- pending work -----------------------------------------------------

   One list frame per unfinished parent, extending
   [experiments/tailcall/tailcall_cases.ml]'s [eval_machine] from its toy
   four-constructor AST to the real language's full node set. A LEAF
   constructor -- one that never calls [go]/[guard] on a child -- produces
   its [Float_result]/[Bool_result] immediately and pushes no frame: [Const],
   [Local], [Local_at], [Local_scan_at], [Load], [Value_of_index],
   [Intrinsic] (its own [rows]/[cols] sweep is already a self-recursive loop,
   ported unchanged below) and [Bool.Index_eq]. Only [Binary], [Unary],
   [Round_f32], [Select], [Bool.Value_lt], [Reduce] and [Scan_at] recurse
   into [go]/[guard] and so need frame handling. *)

type reducers = Reduce_var.t -> int option

(* Carried across one [Reduce]'s iterations. Replaced, not stacked, on every
   iteration -- the frame list only grows for a NESTED [Reduce], never for
   this one's own iteration count, which is exactly the O(depth) frame bound
   the design record requires. *)
(* [combine]/[acc] serve [Max]/[Sum]; [argmax]/[best_i] serve
   [Argmax_value]/[Argmax_index] instead (see the [Reduce_step] frame arm),
   which fold with [Max_op.pool_better] rather than [combine]. *)
type reduce_progress = {
  reduction : Reduction.t;
  outer_reducers : reducers;
  combine : float -> float -> float;
  argmax : bool;
  hi : int;
  i : int;
  acc : float;
  best_i : int;
}

(* Carried across one [Scan_at]'s row/lane fill. Fields are mutated in place
   (this is [eval_machine], not the reuse-optimized candidate, but a scan's
   own progress record is single-threaded through the loop regardless of
   which candidate hosts it, so mutating it saves a reallocation per lane
   with no safety cost). [saved_local_at] is the resolver active the instant
   this scan STARTED -- fixed for the scan's whole lifetime, so nested scans
   each capture their own immediately-enclosing resolver, not a moving
   target. *)
type scan_progress = {
  descriptor : Scan.t;
  requested_row : int;
  requested_lane : int;
  scan_outer_reducers : reducers;
  meter : Scan_meter.t;
  saved_local_at : Local_var.t -> int -> float option;
  mutable prev_row : float array option;
  mutable cur_row : float array;
  mutable lane_cursor : int;
  mutable filling_row : int; (* 0 = filling via [init]; r+1 via [update] *)
}

type frame =
  | Binary_left of Value.binary_op * Value.t * reducers
  | Binary_right of Value.binary_op * float
  | Unary_result of Value.unary_op
  | Round_f32_result
  | Select_result of Value.t * Value.t * reducers
  | Value_lt_left of Value.t * reducers
  | Value_lt_right of float
  | Reduce_step of reduce_progress
  | Scan_fill of scan_progress

type value_state =
  | Eval_state of Value.t * reducers
  | Guard_state of Bool.t * reducers
  | Float_result of float
  | Bool_result of bool

(* Row/column sweep for [Max_pool], unchanged from [eval.ml]'s [JS_BACKEND]
   conversion (Stage 4; see .ai/): already one self-recursive [loop] over a
   nullary tag, so it needs no frame handling of its own -- it never calls
   this module's [go]/[guard]. Duplicated here rather than shared, matching
   this whole file's policy of depending only on [Expr_internal], never on
   [eval.ml]. *)
type pool_tag = Cols_tag | Rows_tag

(* [skip_cleanup] is a PRIVATE, test-only hook for the cleanup protocol's
   negative controls (see .ai/): normal callers never pass it, so the
   exception-path sweep below always runs every pending cleanup, matching
   the design record. Passed [Some p], the sweep withholds a cleanup at
   index [i] (0 = most recently pushed, i.e. the innermost live scan, since
   [cleanups] is prepended on every reservation) whenever [p i] is [true] --
   letting a test deliberately leak a specific reservation and observe the
   resulting [Scan_meter.reserve] failure on reuse, proving the ordinary
   full-cleanup path's success is a real signal, not a vacuous one. Never
   exposed through the public [Expr] API. *)
let eval_machine ?(local = fun _ -> None) ?(local_at = fun _ _ -> None) ?scan
    ?scan_meter ?(reducer = []) ?(on_reduction = fun () -> ())
    ?skip_cleanup (env : Env.t) ~output e =
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
  (* Charges [Scan_meter.charge_update] for row [>0] before evaluating that
     lane's [update] body -- matching [eval.ml]'s per-element [Array.init]
     closure, which charges once per lane, never once per row. Row 0's
     [init] pass charges nothing. *)
  let fill_next_lane p =
    if p.filling_row > 0 then
      vchk (Err.map_error scan_meter_error (Scan_meter.charge_update p.meter));
    let body, bound =
      if p.filling_row = 0 then
        ( p.descriptor.Scan.init,
          fun v ->
            if Reduce_var.equal v p.descriptor.Scan.lane then Some p.lane_cursor
            else p.scan_outer_reducers v )
      else
        let step = p.filling_row - 1 in
        ( p.descriptor.Scan.update,
          fun v ->
            if Reduce_var.equal v p.descriptor.Scan.lane then Some p.lane_cursor
            else if Reduce_var.equal v p.descriptor.Scan.step then Some step
            else p.scan_outer_reducers v )
    in
    (Eval_state (body, bound), Scan_fill p)
  in
  let rec loop state frames =
    match (state, frames) with
    | Eval_state (Value.Const x, _), _ ->
        (loop [@tailcall]) (Float_result x) frames
    | Eval_state (Value.Local v, _), _ -> (
        match local v with
        | Some x -> (loop [@tailcall]) (Float_result x) frames
        | None -> Err.Escape.throw esc (`Unbound_local v))
    | Eval_state (Value.Local_at (v, i), reducers), _ -> (
        match !local_at_ref v (idx reducers i) with
        | Some x -> (loop [@tailcall]) (Float_result x) frames
        | None -> Err.Escape.throw esc (`Unbound_local v))
    | Eval_state (Value.Local_scan_at (v, row_i, lane_i), reducers), _ ->
        let row = idx reducers row_i and lane = idx reducers lane_i in
        (loop [@tailcall])
          (Float_result (vchk (Err.map_error scan_error (scan v ~row ~lane))))
          frames
    | Eval_state (Value.Load (s, c), reducers), _ ->
        (loop [@tailcall])
          (Float_result (vchk (env.Env.load s (Coord.map (idx reducers) c))))
          frames
    | Eval_state (Value.Value_of_index i, reducers), _ ->
        (loop [@tailcall])
          (Float_result (vchk (float_of_index (idx reducers i))))
          frames
    | Eval_state (Value.Intrinsic i, reducers), _ ->
        (loop [@tailcall]) (Float_result (intrinsic reducers i)) frames
    (* Evaluation order is backend-measured (see [order_probe] goldens under
       .ai/), not chosen here: jsoo evaluates [b] before [a] at this call
       site's [eval.ml] original, Melange the reverse. [Binary_left] holds
       whichever operand is NOT yet evaluated, regardless of its original
       slot; the slot only matters again once both values are in hand, at
       the matching [Binary_right] arm below. *)
    | Eval_state (Value.Binary (op, a, b), reducers), _ ->
#if defined MELANGE_BACKEND
        (loop [@tailcall])
          (Eval_state (a, reducers))
          (Binary_left (op, b, reducers) :: frames)
#else
        (loop [@tailcall])
          (Eval_state (b, reducers))
          (Binary_left (op, a, reducers) :: frames)
#endif
    | Eval_state (Value.Unary (op, a), reducers), _ ->
        (loop [@tailcall]) (Eval_state (a, reducers)) (Unary_result op :: frames)
    | Eval_state (Value.Round_f32 a, reducers), _ ->
        (loop [@tailcall])
          (Eval_state (a, reducers))
          (Round_f32_result :: frames)
    | Eval_state (Value.Select (c, a, b), reducers), _ ->
        (loop [@tailcall])
          (Guard_state (c, reducers))
          (Select_result (a, b, reducers) :: frames)
    | Eval_state (Value.Reduce r, reducers), _ ->
        let lo = idx reducers r.Reduction.lo
        and hi = idx reducers r.Reduction.hi in
        let argmax, combine, init =
          match r.Reduction.kind with
          | Reduction.Max ->
              (false, Max_op.apply Max_op.Float_max, Float.neg_infinity)
          | Reduction.Sum -> (false, ( +. ), 0.)
          | Reduction.Argmax_index | Reduction.Argmax_value ->
              (true, (fun _ _ -> assert false), Float.neg_infinity)
        in
        if lo >= hi then
          (loop [@tailcall])
            (Float_result
               (match r.Reduction.kind with
               | Reduction.Argmax_index -> vchk (float_of_index lo)
               | Reduction.Argmax_value | Reduction.Max | Reduction.Sum -> init
               ))
            frames
        else begin
          on_reduction ();
          let bound v =
            if Reduce_var.equal v r.Reduction.var then Some lo else reducers v
          in
          (loop [@tailcall])
            (Eval_state (r.Reduction.body, bound))
            (Reduce_step
               {
                 reduction = r;
                 outer_reducers = reducers;
                 combine;
                 argmax;
                 hi;
                 i = lo;
                 acc = init;
                 best_i = lo;
               }
            :: frames)
        end
    | Eval_state (Value.Scan_at (s, row_i, lane_i), reducers), _ ->
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
          let p =
            {
              descriptor = s;
              requested_row = row;
              requested_lane = lane;
              scan_outer_reducers = reducers;
              meter;
              saved_local_at;
              prev_row = None;
              cur_row = Array.make s.Scan.width 0.;
              lane_cursor = 0;
              filling_row = 0;
            }
          in
          let next_state, frame = fill_next_lane p in
          (loop [@tailcall]) next_state (frame :: frames)
    | Guard_state (Bool.Index_eq (a, b), reducers), _ ->
        (loop [@tailcall])
          (Bool_result (Int.equal (idx reducers a) (idx reducers b)))
          frames
    (* Same backend-measured order as [Binary] above. *)
    | Guard_state (Bool.Value_lt (a, b), reducers), _ ->
#if defined MELANGE_BACKEND
        (loop [@tailcall])
          (Eval_state (a, reducers))
          (Value_lt_left (b, reducers) :: frames)
#else
        (loop [@tailcall])
          (Eval_state (b, reducers))
          (Value_lt_left (a, reducers) :: frames)
#endif
    | Float_result first, Binary_left (op, second_expr, reducers) :: rest ->
        (loop [@tailcall])
          (Eval_state (second_expr, reducers))
          (Binary_right (op, first) :: rest)
    (* [first]/[second] are evaluation order, not the original [a]/[b] slots
       -- restore the slot order here, per backend, before applying [op]. *)
    | Float_result second, Binary_right (op, first) :: rest ->
#if defined MELANGE_BACKEND
        (loop [@tailcall])
          (Float_result (Value.apply_binary op first second))
          rest
#else
        (loop [@tailcall])
          (Float_result (Value.apply_binary op second first))
          rest
#endif
    | Float_result a, Unary_result op :: rest ->
        (loop [@tailcall]) (Float_result (Value.apply_unary op a)) rest
    | Float_result a, Round_f32_result :: rest ->
        (loop [@tailcall])
          (Float_result (Int32.float_of_bits (Int32.bits_of_float a)))
          rest
    | Float_result first, Value_lt_left (second_expr, reducers) :: rest ->
        (loop [@tailcall])
          (Eval_state (second_expr, reducers))
          (Value_lt_right first :: rest)
    | Float_result second, Value_lt_right first :: rest ->
#if defined MELANGE_BACKEND
        (loop [@tailcall]) (Bool_result (first < second)) rest
#else
        (loop [@tailcall]) (Bool_result (second < first)) rest
#endif
    | Bool_result cond, Select_result (a, b, reducers) :: rest ->
        (loop [@tailcall]) (Eval_state ((if cond then a else b), reducers)) rest
    | Float_result v, Reduce_step rs :: rest ->
        let acc, best_i =
          if rs.argmax then
            if Max_op.pool_better ~best:rs.acc ~value:v then (v, rs.i)
            else (rs.acc, rs.best_i)
          else (rs.combine rs.acc v, rs.best_i)
        in
        if rs.i + 1 >= rs.hi then
          (loop [@tailcall])
            (Float_result
               (match rs.reduction.Reduction.kind with
               | Reduction.Argmax_index -> vchk (float_of_index best_i)
               | Reduction.Argmax_value | Reduction.Max | Reduction.Sum -> acc))
            rest
        else begin
          on_reduction ();
          let i = rs.i + 1 in
          let bound w =
            if Reduce_var.equal w rs.reduction.Reduction.var then Some i
            else rs.outer_reducers w
          in
          (loop [@tailcall])
            (Eval_state (rs.reduction.Reduction.body, bound))
            (Reduce_step { rs with i; acc; best_i } :: rest)
        end
    | Float_result v, Scan_fill p :: rest ->
        p.cur_row.(p.lane_cursor) <- v;
        if p.lane_cursor + 1 < p.descriptor.Scan.width then begin
          p.lane_cursor <- p.lane_cursor + 1;
          let next_state, frame = fill_next_lane p in
          (loop [@tailcall]) next_state (frame :: rest)
        end
        else if p.filling_row = p.requested_row then begin
          let result = p.cur_row.(p.requested_lane) in
          run_top_cleanup ();
          (loop [@tailcall]) (Float_result result) rest
        end
        else begin
          let completed_row = p.cur_row in
          let saved = p.saved_local_at in
          p.prev_row <- Some completed_row;
          (local_at_ref :=
             fun v pos ->
               if Local_var.equal v p.descriptor.Scan.prev then
                 Some completed_row.(pos)
               else saved v pos);
          p.cur_row <- Array.make p.descriptor.Scan.width 0.;
          p.lane_cursor <- 0;
          p.filling_row <- p.filling_row + 1;
          let next_state, frame = fill_next_lane p in
          (loop [@tailcall]) next_state (frame :: rest)
        end
    | Float_result v, [] -> v
    | (Bool_result _ | Float_result _), _ -> assert false
  in
  try loop (Eval_state (e, init_reducers)) []
  with exn ->
    let bt = capture_backtrace () in
    (match skip_cleanup with
    | None -> List.iter (fun f -> f ()) !cleanups
    | Some skip ->
        List.iteri (fun i f -> if not (skip i) then f ()) !cleanups);
    cleanups := [];
    reraise exn bt
