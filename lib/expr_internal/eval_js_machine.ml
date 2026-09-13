(* Stack-safe machine fallback for [Eval.value]'s JS branch (tail-call
   conversion, Stage 6; see .ai/): the array/mutation-based frame machine
   selected as [eval_hybrid]'s fallback in Stage 5's benchmarking, installed
   from js/probe/eval_machine_reuse.ml's [run] and the "pending work" types
   it shares with js/probe/eval_candidates.ml (see .ai/'s design record for
   the full evaluation trail: four candidates measured, this one selected).
   [eval.ml]'s JS branch calls into [run] once its own direct recursion
   reaches the production cutoff -- see [eval.ml]'s own doc comment for the
   cutoff value and the handoff protocol.

   Wrapped entirely in [#if defined JS_BACKEND ... #endif] so this file
   compiles to an EMPTY module under native's own [expr_internal] library,
   which takes no [-D] (the same reason [eval.ml]'s own JS branch is dead
   code there) -- [eval.ml]'s native side must never reference anything
   here, and doesn't. Depends only on [Expr_internal] (via [open
   Eval_common]), never on [Eval], specifically to avoid a circular module
   dependency with [eval.ml], which depends on THIS file for its JS-branch
   handoff instead: [pool_tag] is therefore duplicated from [eval.ml]'s own
   JS branch rather than shared, the same policy js/probe/eval_candidates.ml
   documented for the candidates this file is ported from. [run] itself
   owns no [try]/[with] of its own -- the cleanup-on-exception handler
   belongs to [eval.ml], its only caller, matching
   js/probe/eval_machine_reuse.ml's own documented reasoning ("a handoff
   mid-evaluation must unwind through the direct evaluator's own frames
   first"). *)

#if defined JS_BACKEND

open Eval_common

type reducers = Reduce_var.t -> int option

(* Carried across one [Scan_at]'s row/lane fill. Fields are mutated in place
   -- see js/probe/eval_candidates.ml's own doc comment on [scan_progress]
   for why. [saved_local_at] is the resolver active the instant this scan
   STARTED, fixed for the scan's whole lifetime. *)
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

type value_state =
  | Eval_state of float Value.t * reducers
  | Guard_state of Bool.t * reducers
  | Float_result of float
  | Bool_result of bool

(* Row/column sweep for [Max_pool], duplicated from [eval.ml]'s own JS
   branch rather than shared (see this file's top comment) -- already one
   self-recursive [loop] over a nullary tag (Stage 4), so it never pushes a
   [reuse_frame] of its own. *)
type pool_tag = Cols_tag | Rows_tag

(* [combine]/[acc] serve [Max]/[Sum]; [argmax]/[best_i] serve
   [Argmax_value]/[Argmax_index] instead, which fold with [Max_op.pool_better]
   rather than [combine] (see [dispatch_frame]'s [Reduce_step] arm) -- kept as
   separate fields rather than folding [acc] into a pair so [Max]/[Sum]'s hot
   per-step update stays a plain unboxed float write, not a tuple allocation. *)
type reuse_reduce_progress = {
  reduction : Reduction.t;
  outer_reducers : reducers;
  combine : float -> float -> float;
  argmax : bool;
  hi : int;
  mutable i : int;
  mutable acc : float;
  mutable best_i : int;
}

type reuse_frame =
  | Binary_left of Value.binary_op * float Value.t * reducers
  | Binary_right of Value.binary_op * float
  | Unary_result of Value.unary_op
  | Round_f32_result
  | Select_result of float Value.t * float Value.t * reducers
  | Value_lt_left of float Value.t * reducers
  | Value_lt_right of float
  | Reduce_step of reuse_reduce_progress
  | Scan_fill of scan_progress

(* A growable array-backed stack, doubled on overflow and never shrunk
   within one call. [Round_f32_result] is an inert filler for slots beyond
   [top] -- never read, since every push writes its own slot before [top]
   passes it. Popping clears the vacated slot so a large frame doesn't
   outlive its logical pop just because the backing array hasn't shrunk. *)
type reuse_stack = { mutable slots : reuse_frame array; mutable top : int }

let reuse_stack_create () = { slots = Array.make 64 Round_f32_result; top = 0 }

let reuse_stack_push st f =
  if st.top >= Array.length st.slots then begin
    let bigger = Array.make (Array.length st.slots * 2) Round_f32_result in
    Array.blit st.slots 0 bigger 0 st.top;
    st.slots <- bigger
  end;
  st.slots.(st.top) <- f;
  st.top <- st.top + 1

let reuse_stack_pop st =
  if st.top = 0 then None
  else begin
    st.top <- st.top - 1;
    let f = st.slots.(st.top) in
    st.slots.(st.top) <- Round_f32_result;
    Some f
  end

let rec run ~esc ~(env : Env.t) ~output ~scan ~scan_meter ~local ~local_at_ref
    ~(cleanups : (unit -> unit) list ref) ~run_top_cleanup ~on_reduction
    (seed : value_state) : value_state =
  let vchk r = vchk esc r in
  let idx reducers i =
    eval_index esc
      ~widen:(fun (e : index_error) -> (e :> error))
      ~output ~reducers ~resolve_data:env.Env.load_index i
  in
  (* Shared by [I64_to_float]/[Float_to_i64]'s [eval] and [I64_eq]/[I64_lt]'s
     own operand evaluation below: [int64 Value.t] is not itself routed
     through this machine's cutoff/frame-stack (see [Value.eval_i64]'s own
     doc comment), but its [Float_to_i64]/[Select] children can embed the
     full float/bool grammar, so those re-enter [run] synchronously rather
     than assuming a closed leaf. *)
  let eval_i64_via reducers a =
    let eval_float e =
      match
        run ~esc ~env ~output ~scan ~scan_meter ~local ~local_at_ref ~cleanups
          ~run_top_cleanup ~on_reduction (Eval_state (e, reducers))
      with
      | Float_result v -> v
      | _ -> assert false
    in
    let eval_bool c =
      match
        run ~esc ~env ~output ~scan ~scan_meter ~local ~local_at_ref ~cleanups
          ~run_top_cleanup ~on_reduction (Guard_state (c, reducers))
      with
      | Bool_result v -> v
      | _ -> assert false
    in
    Value.eval_i64 ~eval_float ~eval_bool a
  in
  let st = reuse_stack_create () in
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
    Eval_state (body, bound)
  in
  let dispatch_frame (result : value_state) (frame : reuse_frame) : value_state
      =
    match (result, frame) with
    | Float_result first, Binary_left (op, second_expr, reducers) ->
        reuse_stack_push st (Binary_right (op, first));
        Eval_state (second_expr, reducers)
    (* [first]/[second] are evaluation order, not the original [a]/[b] slots
       -- restore the slot order here, per backend, before applying [op].
       Matches [eval.ml]'s JS branch's own [go]/[guard] order for [Binary]/
       [Value_lt] exactly, since both must agree with the SAME backend's
       measured native order (see .ai/'s design record). *)
    | Float_result second, Binary_right (op, first) ->
#if defined MELANGE_BACKEND
        Float_result (Value.apply_binary op first second)
#else
        Float_result (Value.apply_binary op second first)
#endif
    | Float_result a, Unary_result op -> Float_result (Value.apply_unary op a)
    | Float_result a, Round_f32_result ->
        Float_result (Int32.float_of_bits (Int32.bits_of_float a))
    | Float_result first, Value_lt_left (second_expr, reducers) ->
        reuse_stack_push st (Value_lt_right first);
        Eval_state (second_expr, reducers)
    | Float_result second, Value_lt_right first ->
#if defined MELANGE_BACKEND
        Bool_result (first < second)
#else
        Bool_result (second < first)
#endif
    | Bool_result cond, Select_result (a, b, reducers) ->
        Eval_state ((if cond then a else b), reducers)
    | Float_result v, Reduce_step rs ->
        let acc, best_i =
          if rs.argmax then
            if Max_op.pool_better ~best:rs.acc ~value:v then (v, rs.i)
            else (rs.acc, rs.best_i)
          else (rs.combine rs.acc v, rs.best_i)
        in
        if rs.i + 1 >= rs.hi then
          Float_result
            (match rs.reduction.Reduction.kind with
            | Reduction.Argmax_index -> vchk (float_of_index best_i)
            | Reduction.Argmax_value | Reduction.Max | Reduction.Sum -> acc)
        else begin
          on_reduction ();
          let i = rs.i + 1 in
          rs.i <- i;
          rs.acc <- acc;
          rs.best_i <- best_i;
          reuse_stack_push st (Reduce_step rs);
          let bound w =
            if Reduce_var.equal w rs.reduction.Reduction.var then Some i
            else rs.outer_reducers w
          in
          Eval_state (rs.reduction.Reduction.body, bound)
        end
    | Float_result v, Scan_fill p ->
        p.cur_row.(p.lane_cursor) <- v;
        if p.lane_cursor + 1 < p.descriptor.Scan.width then begin
          p.lane_cursor <- p.lane_cursor + 1;
          reuse_stack_push st (Scan_fill p);
          fill_next_lane p
        end
        else if p.filling_row = p.requested_row then begin
          let result = p.cur_row.(p.requested_lane) in
          run_top_cleanup ();
          Float_result result
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
          reuse_stack_push st (Scan_fill p);
          fill_next_lane p
        end
    | _ -> assert false
  in
  let rec loop state =
    match state with
    | Eval_state (Value.Const x, _) -> (loop [@tailcall]) (Float_result x)
    | Eval_state (Value.I64_to_float a, reducers) ->
        (loop [@tailcall])
          (Float_result (Int64.to_float (vchk (eval_i64_via reducers a))))
    | Eval_state (Value.Local v, _) -> (
        match local v with
        | Some x -> (loop [@tailcall]) (Float_result x)
        | None -> Err.Escape.throw esc (`Unbound_local v))
    | Eval_state (Value.Local_at (v, i), reducers) -> (
        match !local_at_ref v (idx reducers i) with
        | Some x -> (loop [@tailcall]) (Float_result x)
        | None -> Err.Escape.throw esc (`Unbound_local v))
    | Eval_state (Value.Local_scan_at (v, row_i, lane_i), reducers) ->
        let row = idx reducers row_i and lane = idx reducers lane_i in
        (loop [@tailcall])
          (Float_result (vchk (Err.map_error scan_error (scan v ~row ~lane))))
    | Eval_state (Value.Load (s, c), reducers) ->
        (loop [@tailcall])
          (Float_result (vchk (env.Env.load s (Coord.map (idx reducers) c))))
    | Eval_state (Value.Value_of_index i, reducers) ->
        (loop [@tailcall])
          (Float_result (vchk (float_of_index (idx reducers i))))
    | Eval_state (Value.Intrinsic i, reducers) ->
        (loop [@tailcall]) (Float_result (intrinsic reducers i))
    | Eval_state (Value.Binary (op, a, b), reducers) ->
#if defined MELANGE_BACKEND
        reuse_stack_push st (Binary_left (op, b, reducers));
        (loop [@tailcall]) (Eval_state (a, reducers))
#else
        reuse_stack_push st (Binary_left (op, a, reducers));
        (loop [@tailcall]) (Eval_state (b, reducers))
#endif
    | Eval_state (Value.Unary (op, a), reducers) ->
        reuse_stack_push st (Unary_result op);
        (loop [@tailcall]) (Eval_state (a, reducers))
    | Eval_state (Value.Round_f32 a, reducers) ->
        reuse_stack_push st Round_f32_result;
        (loop [@tailcall]) (Eval_state (a, reducers))
    | Eval_state (Value.Select (c, a, b), reducers) ->
        reuse_stack_push st (Select_result (a, b, reducers));
        (loop [@tailcall]) (Guard_state (c, reducers))
    | Eval_state (Value.Reduce r, reducers) ->
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
        else begin
          on_reduction ();
          let bound v =
            if Reduce_var.equal v r.Reduction.var then Some lo else reducers v
          in
          reuse_stack_push st
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
               });
          (loop [@tailcall]) (Eval_state (r.Reduction.body, bound))
        end
    | Eval_state (Value.Scan_at (s, row_i, lane_i), reducers) ->
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
          reuse_stack_push st (Scan_fill p);
          (loop [@tailcall]) (fill_next_lane p)
    | Guard_state (Bool.Index_eq (a, b), reducers) ->
        (loop [@tailcall])
          (Bool_result (Int.equal (idx reducers a) (idx reducers b)))
    | Guard_state (Bool.I64_eq (a, b), reducers) ->
        (loop [@tailcall])
          (Bool_result
             (Int64.equal
                (vchk (eval_i64_via reducers a))
                (vchk (eval_i64_via reducers b))))
    | Guard_state (Bool.I64_lt (a, b), reducers) ->
        (loop [@tailcall])
          (Bool_result
             (Int64.compare
                (vchk (eval_i64_via reducers a))
                (vchk (eval_i64_via reducers b))
             < 0))
    | Guard_state (Bool.Value_lt (a, b), reducers) ->
#if defined MELANGE_BACKEND
        reuse_stack_push st (Value_lt_left (b, reducers));
        (loop [@tailcall]) (Eval_state (a, reducers))
#else
        reuse_stack_push st (Value_lt_left (a, reducers));
        (loop [@tailcall]) (Eval_state (b, reducers))
#endif
    | Float_result v -> (
        match reuse_stack_pop st with
        | None -> Float_result v
        | Some frame ->
            (loop [@tailcall]) (dispatch_frame (Float_result v) frame))
    | Bool_result cond -> (
        match reuse_stack_pop st with
        | None -> Bool_result cond
        | Some frame ->
            (loop [@tailcall]) (dispatch_frame (Bool_result cond) frame))
  in
  loop seed

#endif
